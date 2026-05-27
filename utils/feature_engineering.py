"""Feature engineering for AQI time-series forecasting."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

RAW_AQI_COLS = ["aqi", "pm25", "pm10", "no2", "o3", "co", "so2"]
WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "weather_code",
    "visibility",
]


def _ensure_utc_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Normalize timestamp column to UTC datetime64.

    Args:
        df: Input DataFrame.
        col: Timestamp column name.

    Returns:
        DataFrame with UTC timestamp.
    """
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True)
    return out


def merge_aqi_weather(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge AQI and weather DataFrames on hourly timestamp.

    Args:
        aqi_df: Air quality readings.
        weather_df: Weather readings.

    Returns:
        Merged hourly DataFrame.
    """
    aqi = _ensure_utc_timestamp(aqi_df).sort_values("timestamp")
    weather = _ensure_utc_timestamp(weather_df).sort_values("timestamp")
    aqi["timestamp"] = aqi["timestamp"].dt.floor("h")
    weather["timestamp"] = weather["timestamp"].dt.floor("h")
    merged = pd.merge(aqi, weather, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def compute_features(
    aqi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    drop_targets: bool = True,
) -> pd.DataFrame:
    """Engineer features and multi-horizon targets from AQI and weather data.

    Args:
        aqi_df: Air quality DataFrame with ``timestamp`` and pollutant columns.
        weather_df: Weather DataFrame aligned hourly.
        drop_targets: If True, drop rows with NaN targets after shifting.

    Returns:
        Feature DataFrame with targets ``aqi_t24``, ``aqi_t48``, ``aqi_t72``.
    """
    df = merge_aqi_weather(aqi_df, weather_df)
    if df.empty:
        logger.warning("Empty merged DataFrame; no features computed.")
        return df

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for lag in config.LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

    df["aqi_rolling_mean_3h"] = df["aqi"].rolling(3, min_periods=1).mean()
    df["aqi_rolling_mean_6h"] = df["aqi"].rolling(6, min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
    df["aqi_rolling_std_6h"] = df["aqi"].rolling(6, min_periods=1).std().fillna(0)
    df["aqi_rolling_std_24h"] = df["aqi"].rolling(24, min_periods=1).std().fillna(0)
    df["aqi_rolling_max_24h"] = df["aqi"].rolling(24, min_periods=1).max()
    df["aqi_rolling_min_24h"] = df["aqi"].rolling(24, min_periods=1).min()

    df["aqi_change_rate"] = (df["aqi"] - df["aqi_lag_1h"]) / (df["aqi_lag_1h"] + 1e-6)
    df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-6)
    df["aqi_trend_3h"] = df["aqi"] - df["aqi_lag_3h"]

    df["aqi_t24"] = df["aqi"].shift(-24)
    df["aqi_t48"] = df["aqi"].shift(-48)
    df["aqi_t72"] = df["aqi"].shift(-72)

    if drop_targets:
        before = len(df)
        df = df.dropna(subset=config.TARGET_COLUMNS)
        logger.debug("Dropped %d rows with incomplete targets.", before - len(df))

    return df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return model feature column names (excludes timestamp and targets).

    Args:
        df: Feature DataFrame.

    Returns:
        List of feature column names.
    """
    exclude = {"timestamp", *config.TARGET_COLUMNS}
    return [c for c in df.columns if c not in exclude]


def build_single_row_features(
    current_aqi: dict,
    current_weather: dict,
    history_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Build feature row for real-time inference using historical context.

    Args:
        current_aqi: Current AQI reading dict.
        current_weather: Current weather dict.
        history_df: Recent feature store rows for lag computation.

    Returns:
        Single-row feature DataFrame, or ``None`` if insufficient history.
    """
    row = {**current_aqi, **current_weather}
    row.pop("timestamp", None)
    ts = pd.Timestamp(current_aqi.get("timestamp", pd.Timestamp.now(tz="UTC")))
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.floor("h")

    new_df = pd.DataFrame([{**row, "timestamp": ts}])

    if history_df is not None and not history_df.empty:
        hist = _ensure_utc_timestamp(history_df)
        combined = pd.concat([hist, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        weather_cols = [c for c in WEATHER_COLS if c in combined.columns]
        aqi_cols = [c for c in RAW_AQI_COLS if c in combined.columns]
        features = compute_features(
            combined[aqi_cols + ["timestamp"]],
            combined[weather_cols + ["timestamp"]],
            drop_targets=False,
        )
        last = features[features["timestamp"] == ts]
        if last.empty:
            last = features.tail(1)
        return last
    return None
