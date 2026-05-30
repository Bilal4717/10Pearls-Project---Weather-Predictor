"""Unified AQI data source with Open-Meteo (primary) and AQICN (fallback).

All pipelines and apps fetch AQI through these helpers so the underlying
provider can change without touching downstream code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

import config
from utils.aqicn_client import AQICNClient
from utils.openmeteo_aqi_client import OpenMeteoAQIClient

logger = logging.getLogger(__name__)

_POLLUTANT_KEYS = ["aqi", "pm25", "pm10", "no2", "o3", "co", "so2"]


def _is_usable(reading: Optional[Dict[str, Any]]) -> bool:
    """Check whether an AQI reading has a numeric AQI value.

    Args:
        reading: Candidate reading dict.

    Returns:
        True if ``aqi`` is present and numeric.
    """
    if not reading:
        return False
    aqi = reading.get("aqi")
    if aqi is None:
        return False
    try:
        float(aqi)
        return True
    except (TypeError, ValueError):
        return False


def get_current_aqi() -> Optional[Dict[str, Any]]:
    """Return the latest AQI reading from the primary source, else fallback.

    Returns:
        Reading dict (aqi, pollutants, timestamp), or ``None`` if all fail.
    """
    if config.USE_OPENMETEO_AQI:
        reading = OpenMeteoAQIClient().get_current()
        if _is_usable(reading):
            return reading
        logger.warning("Open-Meteo AQI unusable; trying AQICN fallback.")

    reading = AQICNClient().get_current(config.AQICN_STATION)
    if _is_usable(reading):
        return reading

    logger.warning("No usable current AQI from any source.")
    return None


def get_historical_aqi(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Return hourly historical AQI for a date range from the primary source.

    Args:
        start_date: ``YYYY-MM-DD`` start.
        end_date: ``YYYY-MM-DD`` end.

    Returns:
        Hourly DataFrame with aqi + pollutant columns, or ``None``.
    """
    if config.USE_OPENMETEO_AQI:
        df = OpenMeteoAQIClient().get_historical(start_date, end_date)
        if df is not None and not df.empty:
            return df
        logger.warning(
            "Open-Meteo AQI history empty for %s..%s; AQICN has no hourly history.",
            start_date,
            end_date,
        )
    return None


def get_aqi_forecast(days: int = 3) -> Optional[pd.DataFrame]:
    """Return hourly AQI forecast from the primary source.

    Args:
        days: Forecast horizon in days.

    Returns:
        Forecast DataFrame, or ``None``.
    """
    if config.USE_OPENMETEO_AQI:
        return OpenMeteoAQIClient().get_forecast(days)
    return None
