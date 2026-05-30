"""Hourly feature pipeline: fetch AQI + weather, engineer features, store in Hopsworks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from utils.aqi_source import get_current_aqi, get_historical_aqi
from utils.feature_engineering import compute_features, merge_aqi_weather
from utils.hopsworks_utils import insert_features, read_feature_group
from utils.logging_config import setup_logging
from utils.openmeteo_client import OpenMeteoClient

load_dotenv(ROOT / ".env")
logger = setup_logging(__name__, "feature_pipeline.log")


def fetch_current_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch current AQI and weather readings.

    Returns:
        Tuple of (aqi_df, weather_df) as single-row DataFrames.
    """
    weather_client = OpenMeteoClient()

    aqi = get_current_aqi()
    weather = weather_client.get_current()

    if aqi is None or weather is None:
        raise RuntimeError("Failed to fetch current AQI or weather data.")

    aqi_df = pd.DataFrame([aqi])
    weather_df = pd.DataFrame([{k: weather[k] for k in weather if k != "timestamp"}])
    weather_df["timestamp"] = aqi["timestamp"]
    return aqi_df, weather_df


def fetch_date_data(date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch AQI and weather for a specific historical date.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format.

    Returns:
        Tuple of (aqi_df, weather_df).
    """
    weather_client = OpenMeteoClient()

    aqi_hist = get_historical_aqi(date_str, date_str)
    weather_hist = weather_client.get_historical(date_str, date_str)

    if aqi_hist is None or aqi_hist.empty:
        raise RuntimeError(f"No AQI data for {date_str}")
    if weather_hist is None or weather_hist.empty:
        raise RuntimeError(f"No weather data for {date_str}")

    return aqi_hist, weather_hist


def run(date: str | None = None) -> None:
    """Execute the feature pipeline.

    Args:
        date: Optional ``YYYY-MM-DD`` for historical fetch instead of current.
    """
    if date:
        aqi_df, weather_df = fetch_date_data(date)
    else:
        aqi_df, weather_df = fetch_current_data()

    history = read_feature_group()
    if not history.empty:
        history = history.sort_values("timestamp").tail(100)
        combined_aqi = pd.concat(
            [history[[c for c in history.columns if c in aqi_df.columns]], aqi_df],
            ignore_index=True,
        )
        combined_weather = pd.concat(
            [
                history[[c for c in history.columns if c in weather_df.columns or c == "timestamp"]],
                weather_df,
            ],
            ignore_index=True,
        )
        features = compute_features(combined_aqi, combined_weather, drop_targets=False)
        ts = pd.to_datetime(aqi_df["timestamp"].iloc[0], utc=True).floor("h")
        row = features[features["timestamp"] == ts]
        if row.empty:
            row = features.tail(1)
    else:
        merged = merge_aqi_weather(aqi_df, weather_df)
        aqi_cols = [c for c in merged.columns if c in aqi_df.columns or c == "timestamp"]
        weather_cols = [c for c in merged.columns if c in weather_df.columns or c == "timestamp"]
        row = compute_features(
            merged[aqi_cols], merged[weather_cols], drop_targets=False
        ).tail(1)

    insert_features(row, upsert=True)
    aqi_val = row["aqi"].iloc[0] if "aqi" in row.columns else "N/A"
    logger.info(
        "Feature pipeline success | timestamp=%s | aqi=%s",
        row["timestamp"].iloc[0],
        aqi_val,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AQI feature pipeline")
    parser.add_argument("--date", type=str, help="YYYY-MM-DD for historical run")
    args = parser.parse_args()
    try:
        run(date=args.date)
    except Exception as exc:
        logger.exception("Feature pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
