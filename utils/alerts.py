"""Hazardous AQI alert logic for dashboard and API consumers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import config


def get_aqi_category(aqi: float) -> str:
    """Map AQI value to EPA category label.

    Args:
        aqi: Air Quality Index value.

    Returns:
        Category label string.
    """
    if aqi <= config.AQI_THRESHOLDS["good"]:
        return "Good"
    if aqi <= config.AQI_THRESHOLDS["moderate"]:
        return "Moderate"
    if aqi <= config.AQI_THRESHOLDS["unhealthy_sensitive"]:
        return "Unhealthy for Sensitive Groups"
    if aqi <= config.AQI_THRESHOLDS["unhealthy"]:
        return "Unhealthy"
    if aqi <= config.AQI_THRESHOLDS["very_unhealthy"]:
        return "Very Unhealthy"
    return "Hazardous"


def check_alerts(
    current_aqi: float,
    forecast_24h: Optional[float] = None,
    forecast_48h: Optional[float] = None,
    forecast_72h: Optional[float] = None,
    aqi_change_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Evaluate alert conditions for current and forecast AQI.

    Args:
        current_aqi: Current AQI reading.
        forecast_24h: 24-hour ahead forecast.
        forecast_48h: 48-hour ahead forecast.
        forecast_72h: 72-hour ahead forecast.
        aqi_change_rate: Optional rate of change over 1 hour.

    Returns:
        List of alert dicts with keys: level, message, recommendation, color.
    """
    alerts: List[Dict[str, Any]] = []

    if current_aqi > config.AQI_THRESHOLDS["unhealthy_sensitive"]:
        alerts.append(
            {
                "level": "warning",
                "message": "Unhealthy — limit outdoor activity",
                "recommendation": (
                    "Sensitive groups should avoid prolonged outdoor exertion. "
                    "Consider wearing a mask if you must go outside."
                ),
                "color": "#FF0000",
            }
        )

    forecasts = [f for f in [forecast_24h, forecast_48h, forecast_72h] if f is not None]
    if any(f > config.AQI_THRESHOLDS["unhealthy"] for f in forecasts):
        alerts.append(
            {
                "level": "forecast",
                "message": "Forecast shows very unhealthy conditions",
                "recommendation": (
                    "Plan to limit outdoor activities in the next 1–3 days. "
                    "Monitor updates hourly."
                ),
                "color": "#8F3F97",
            }
        )

    if aqi_change_rate is not None and aqi_change_rate > 0.20:
        alerts.append(
            {
                "level": "trend",
                "message": "AQI rising rapidly",
                "recommendation": (
                    "Air quality is deteriorating quickly. Stay indoors if possible."
                ),
                "color": "#FF7E00",
            }
        )

    check_values = [current_aqi, *forecasts]
    if any(v > config.AQI_THRESHOLDS["very_unhealthy"] for v in check_values):
        alerts.append(
            {
                "level": "critical",
                "message": "HAZARDOUS — stay indoors, use air purifier",
                "recommendation": (
                    "Avoid all outdoor activity. Keep windows closed and use "
                    "HEPA air purifiers if available."
                ),
                "color": "#7E0023",
            }
        )

    return alerts
