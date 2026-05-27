"""Open-Meteo API client for current, historical, and forecast weather data."""

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

HOURLY_VARS: List[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "weather_code",
    "visibility",
]


class OpenMeteoClient:
    """Client for Open-Meteo weather APIs (no authentication required)."""

    def __init__(self) -> None:
        """Initialize cached, retry-enabled Open-Meteo session."""
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.lat = config.LATITUDE
        self.lon = config.LONGITUDE
        self.timezone = config.TIMEZONE

    def _hourly_to_dataframe(
        self, hourly: Dict[str, Any], start_idx: int = 0, end_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """Convert Open-Meteo hourly block to a UTC-indexed DataFrame.

        Args:
            hourly: ``hourly`` section from API response.
            start_idx: Start slice index.
            end_idx: End slice index (exclusive).

        Returns:
            DataFrame with ``timestamp`` column in UTC.
        """
        times = hourly["time"][start_idx:end_idx]
        records: Dict[str, List[Any]] = {"timestamp": times}
        for var in HOURLY_VARS:
            if var in hourly:
                records[var] = hourly[var][start_idx:end_idx]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def get_current(self) -> Optional[Dict[str, Any]]:
        """Fetch current weather conditions for Karachi.

        Returns:
            Dict of current weather variables, or ``None`` on failure.
        """
        try:
            url = config.OPENMETEO_FORECAST_URL
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "current": HOURLY_VARS,
                "timezone": self.timezone,
            }
            responses = self.client.weather_api(url, params=params)
            response = responses[0]
            current = response.Current()
            result: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc),
            }
            for i, var in enumerate(HOURLY_VARS):
                result[var] = current.Variables(i).Value()
            return result
        except Exception as exc:
            logger.warning("Failed to fetch current weather: %s", exc)
            return None

    def get_historical(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch hourly historical weather for a date range.

        Args:
            start_date: Start date ``YYYY-MM-DD``.
            end_date: End date ``YYYY-MM-DD``.

        Returns:
            Hourly weather DataFrame, or ``None`` on failure.
        """
        try:
            url = config.OPENMETEO_ARCHIVE_URL
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(HOURLY_VARS),
                "timezone": "UTC",
            }
            responses = self.client.weather_api(url, params=params)
            hourly = responses[0].Hourly()
            payload = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                ).strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            }
            for i, var in enumerate(HOURLY_VARS):
                payload[var] = hourly.Variables(i).ValuesAsNumpy().tolist()
            return self._hourly_to_dataframe(payload)
        except Exception as exc:
            logger.warning(
                "Failed to fetch historical weather %s to %s: %s",
                start_date,
                end_date,
                exc,
            )
            return None

    def get_forecast(self, days: int = 3) -> Optional[pd.DataFrame]:
        """Fetch hourly weather forecast.

        Args:
            days: Number of forecast days (default 3 → 72 hours).

        Returns:
            Forecast DataFrame with UTC timestamps, or ``None`` on failure.
        """
        try:
            url = config.OPENMETEO_FORECAST_URL
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "hourly": ",".join(HOURLY_VARS),
                "forecast_days": days,
                "timezone": "UTC",
            }
            responses = self.client.weather_api(url, params=params)
            hourly = responses[0].Hourly()
            payload = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                ).strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            }
            for i, var in enumerate(HOURLY_VARS):
                payload[var] = hourly.Variables(i).ValuesAsNumpy().tolist()
            return self._hourly_to_dataframe(payload)
        except Exception as exc:
            logger.warning("Failed to fetch weather forecast: %s", exc)
            return None
