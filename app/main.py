"""FastAPI backend for Karachi AQI prediction service."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

import config
from utils.aqicn_client import AQICNClient
from utils.alerts import check_alerts, get_aqi_category
from utils.hopsworks_utils import read_feature_group
from utils.inference import get_feature_importance, predict_forecast
from utils.logging_config import setup_logging
from utils.openmeteo_client import OpenMeteoClient

logger = setup_logging(__name__, "api.log")

app = FastAPI(
    title="Karachi AQI Forecast API",
    description="Serverless AQI prediction API for Karachi, Pakistan",
    version="1.0.0",
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Status and UTC timestamp.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/current")
def current() -> Dict[str, Any]:
    """Return current AQI and weather from live APIs.

    Returns:
        Current readings with category label.

    Raises:
        HTTPException: If data fetch fails.
    """
    aqi_client = AQICNClient()
    weather_client = OpenMeteoClient()
    aqi = aqi_client.get_current(config.AQICN_STATION)
    weather = weather_client.get_current()
    if aqi is None:
        raise HTTPException(status_code=503, detail="Failed to fetch current AQI")
    return {
        "aqi": aqi,
        "weather": weather,
        "category": get_aqi_category(aqi.get("aqi", 0) or 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/forecast")
def forecast() -> Dict[str, Any]:
    """Return 24/48/72-hour AQI forecasts.

    Returns:
        Predictions with confidence intervals and alerts.

    Raises:
        HTTPException: On prediction failure.
    """
    aqi_client = AQICNClient()
    weather_client = OpenMeteoClient()
    aqi = aqi_client.get_current(config.AQICN_STATION)
    weather = weather_client.get_current()
    if aqi is None or weather is None:
        raise HTTPException(status_code=503, detail="Failed to fetch live data")

    try:
        result = predict_forecast(aqi, weather)
    except Exception as exc:
        logger.exception("Forecast failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    preds = result["predictions"]
    alerts = check_alerts(
        current_aqi=aqi.get("aqi", 0) or 0,
        forecast_24h=preds.get("24h"),
        forecast_48h=preds.get("48h"),
        forecast_72h=preds.get("72h"),
    )
    result["alerts"] = alerts
    return result


@app.get("/history")
def history(days: int = Query(default=7, ge=1, le=90)) -> Dict[str, Any]:
    """Return historical AQI from the feature store.

    Args:
        days: Number of past days to return.

    Returns:
        List of timestamp/AQI records.

    Raises:
        HTTPException: If no data available.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = read_feature_group(
        start_time=start.isoformat(),
        end_time=end.isoformat(),
    )
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data found")

    records = df[["timestamp", "aqi", "pm25", "pm10"]].copy()
    records["timestamp"] = records["timestamp"].astype(str)
    return {
        "days": days,
        "count": len(records),
        "data": records.to_dict(orient="records"),
    }


@app.get("/feature_importance")
def feature_importance() -> Dict[str, Any]:
    """Return top SHAP feature importances for the deployed model.

    Returns:
        Ranked feature importance list.
    """
    items = get_feature_importance()
    return {"features": items, "count": len(items)}
