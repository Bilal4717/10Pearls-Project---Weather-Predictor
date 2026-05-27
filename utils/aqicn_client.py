"""AQICN API client for current and historical air quality data."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

logger = logging.getLogger(__name__)


class AQICNClient:
    """Client wrapper for the World Air Quality Index (AQICN) API."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the AQICN client.

        Args:
            token: AQICN API token. Falls back to ``AQICN_TOKEN`` env var.
        """
        self.token = token or os.getenv("AQICN_TOKEN")
        if not self.token:
            logger.warning("AQICN_TOKEN not set; API calls will fail.")
        self.base_url = config.AQICN_BASE_URL
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
        stop=stop_after_attempt(config.API_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=config.API_RETRY_WAIT_MIN,
            max=config.API_RETRY_WAIT_MAX,
        ),
        reraise=True,
    )
    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GET request against the AQICN API.

        Args:
            endpoint: API path (e.g. ``feed/@7064/``).
            params: Query parameters.

        Returns:
            Parsed JSON response body.

        Raises:
            requests.RequestException: On network or HTTP errors.
            ValueError: When API returns non-success status.
        """
        params = {**params, "token": self.token}
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            raise ValueError(f"AQICN API error: {data.get('data', data)}")
        return data

    def _parse_reading(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize AQICN feed payload into a flat dict.

        Args:
            payload: Raw ``data`` object from AQICN response.

        Returns:
            Dict with aqi, pollutants, and UTC timestamp.
        """
        iaqi = payload.get("iaqi", {}) or {}

        def _val(key: str) -> Optional[float]:
            entry = iaqi.get(key, {})
            if isinstance(entry, dict):
                return entry.get("v")
            return None

        ts = payload.get("time", {}).get("iso") or payload.get("time", {}).get("s")
        if ts:
            timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        return {
            "aqi": payload.get("aqi"),
            "pm25": _val("pm25"),
            "pm10": _val("pm10"),
            "no2": _val("no2"),
            "o3": _val("o3"),
            "co": _val("co"),
            "so2": _val("so2"),
            "timestamp": timestamp,
        }

    def get_current(self, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch current air quality for a city or station.

        Args:
            city: City slug or station ID (e.g. ``@7064``). Defaults to config.

        Returns:
            Normalized reading dict, or ``None`` on failure.
        """
        station = city or config.AQICN_STATION
        try:
            data = self._get(f"feed/{station}/", {})
            return self._parse_reading(data["data"])
        except Exception as exc:
            logger.warning("Failed to fetch current AQI for %s: %s", station, exc)
            return None

    def get_historical(
        self, city: Optional[str] = None, date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch historical air quality for a specific date.

        Args:
            city: City slug or station ID.
            date: Date string ``YYYY-MM-DD``.

        Returns:
            Normalized reading dict, or ``None`` on failure.
        """
        station = city or config.AQICN_STATION
        if not date:
            logger.warning("get_historical requires a date (YYYY-MM-DD).")
            return None
        try:
            data = self._get(f"feed/{station}/{date}/", {})
            return self._parse_reading(data["data"])
        except Exception as exc:
            logger.warning(
                "Failed to fetch historical AQI for %s on %s: %s", station, date, exc
            )
            return None

    def get_feed_json(self, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch raw feed JSON for the station (used by backfill).

        Args:
            city: Station ID.

        Returns:
            Full API response data block, or ``None`` on failure.
        """
        station = city or config.AQICN_STATION
        try:
            data = self._get(f"feed/{station}/", {})
            return data.get("data")
        except Exception as exc:
            logger.warning("Failed to fetch feed for %s: %s", station, exc)
            return None
