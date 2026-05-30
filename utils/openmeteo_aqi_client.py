"""Open-Meteo Air Quality API client (primary AQI + pollutant source).

Provides hourly AQI and pollutant data for Karachi with no API key. Replaces
AQICN as the primary source because AQICN no longer has a live Karachi station.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

import config

logger = logging.getLogger(__name__)

# Request order must match the index order used when reading the response.
_REQUEST_VARS: List[str] = list(config.OPENMETEO_AQI_VARS.keys())
_INTERNAL_NAMES: List[str] = list(config.OPENMETEO_AQI_VARS.values())


class OpenMeteoAQIClient:
    """Client for the Open-Meteo Air Quality API (no authentication required)."""

    def __init__(self) -> None:
        """Initialize a cached, retry-enabled Open-Meteo client."""
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.lat = config.LATITUDE
        self.lon = config.LONGITUDE

    def _hourly_dataframe(self, hourly: Any) -> pd.DataFrame:
        """Convert an Open-Meteo hourly block into a UTC-indexed DataFrame.

        Args:
            hourly: ``Hourly()`` section of the API response.

        Returns:
            DataFrame with ``timestamp`` plus internal pollutant/AQI columns.
        """
        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        data: Dict[str, Any] = {"timestamp": timestamps}
        for i, internal in enumerate(_INTERNAL_NAMES):
            data[internal] = hourly.Variables(i).ValuesAsNumpy()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _fetch(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Execute an air-quality request and return an hourly DataFrame.

        Args:
            params: Query parameters (latitude/longitude added automatically).

        Returns:
            Hourly DataFrame, or ``None`` on failure.
        """
        try:
            base = {
                "latitude": self.lat,
                "longitude": self.lon,
                "hourly": _REQUEST_VARS,
                "timezone": "UTC",
            }
            responses = self.client.weather_api(
                config.OPENMETEO_AQI_URL, params={**base, **params}
            )
            return self._hourly_dataframe(responses[0].Hourly())
        except Exception as exc:
            logger.warning("Open-Meteo AQI request failed: %s", exc)
            return None

    def get_current(self) -> Optional[Dict[str, Any]]:
        """Return the most recent available hourly AQI reading.

        Returns:
            Dict with aqi, pollutants, and UTC timestamp, or ``None`` on failure.
        """
        df = self._fetch({"past_days": 2, "forecast_days": 1})
        if df is None or df.empty:
            return None
        now = pd.Timestamp.now(tz="UTC").floor("h")
        valid = df[(df["aqi"].notna()) & (df["timestamp"] <= now)]
        if valid.empty:
            valid = df[df["aqi"].notna()]
        if valid.empty:
            return None
        row = valid.iloc[-1]
        result: Dict[str, Any] = {"timestamp": row["timestamp"].to_pydatetime()}
        for name in _INTERNAL_NAMES:
            val = row[name]
            result[name] = None if pd.isna(val) else float(val)
        return result

    def get_historical(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch hourly AQI + pollutants for a date range.

        Args:
            start_date: Start date ``YYYY-MM-DD``.
            end_date: End date ``YYYY-MM-DD``.

        Returns:
            Hourly DataFrame, or ``None`` on failure.
        """
        df = self._fetch({"start_date": start_date, "end_date": end_date})
        if df is None or df.empty:
            return None
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_forecast(self, days: int = 3) -> Optional[pd.DataFrame]:
        """Fetch hourly AQI + pollutant forecast.

        Args:
            days: Number of forecast days (default 3 → 72 hours).

        Returns:
            Forecast DataFrame, or ``None`` on failure.
        """
        df = self._fetch({"forecast_days": days})
        if df is None or df.empty:
            return None
        return df.sort_values("timestamp").reset_index(drop=True)
