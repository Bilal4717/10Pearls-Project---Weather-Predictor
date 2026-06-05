"""Streamlit dashboard for Karachi AQI forecasting."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load credentials. Locally this reads .env; on Streamlit Cloud the values come
# from st.secrets, which we mirror into environment variables so the Hopsworks
# and AQICN clients (which use os.getenv) work in both environments.
load_dotenv(ROOT / ".env")


def _bridge_secrets_to_env() -> str:
    """Mirror Streamlit secrets into env vars and return the API base URL.

    Returns:
        API base URL (defaults to localhost).
    """
    api_base = "http://localhost:8000"
    try:
        secrets = st.secrets
    except Exception:
        return api_base
    for key in ("AQICN_TOKEN", "HOPSWORKS_API_KEY"):
        try:
            if key in secrets and secrets[key]:
                os.environ[key] = str(secrets[key])
        except Exception:
            continue
    try:
        if "API_BASE_URL" in secrets and secrets["API_BASE_URL"]:
            api_base = str(secrets["API_BASE_URL"])
    except Exception:
        pass
    return api_base


import config
from utils.alerts import check_alerts, get_aqi_category
from utils.aqi_source import get_current_aqi
from utils.hopsworks_utils import read_feature_group
from utils.inference import get_feature_importance, predict_forecast
from utils.openmeteo_client import OpenMeteoClient

API_BASE = _bridge_secrets_to_env()

# Page config
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .aqi-metric { font-size: 3rem; font-weight: bold; }
    .alert-banner {
        background: #7f1d1d; padding: 1rem; border-radius: 8px;
        border-left: 5px solid #ef4444; margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def aqi_color(aqi: float) -> str:
    """Return hex color for AQI value.

    Args:
        aqi: Air Quality Index.

    Returns:
        Hex color string.
    """
    for cat in config.AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return str(cat["color"])
    return "#7E0023"


@st.cache_data(ttl=3600)
def fetch_current_local() -> dict:
    """Fetch current AQI and weather (cached 1 hour).

    Returns:
        Dict with aqi and weather keys.
    """
    aqi = get_current_aqi()
    weather = OpenMeteoClient().get_current()
    return {"aqi": aqi, "weather": weather}


@st.cache_data(ttl=3600)
def fetch_forecast_local() -> dict:
    """Generate forecast using local inference (cached 1 hour).

    Returns:
        Forecast result dict.
    """
    data = fetch_current_local()
    if not data["aqi"] or not data["weather"]:
        return {}
    return predict_forecast(data["aqi"], data["weather"])


@st.cache_data(ttl=3600)
def fetch_history_local(days: int = 7) -> pd.DataFrame:
    """Load historical AQI from feature store (cached 1 hour).

    Args:
        days: Number of days of history.

    Returns:
        Historical DataFrame.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = read_feature_group(start_time=start.isoformat(), end_time=end.isoformat())
    if df.empty:
        try:
            resp = requests.get(f"{API_BASE}/history", params={"days": days}, timeout=15)
            if resp.ok:
                records = resp.json().get("data", [])
                return pd.DataFrame(records)
        except Exception:
            pass
    return df


@st.cache_data(ttl=3600)
def fetch_importance_local() -> pd.DataFrame:
    """Load SHAP feature importance (cached 1 hour).

    Returns:
        DataFrame with feature and importance columns.
    """
    items = get_feature_importance()
    return pd.DataFrame(items)


def render_header() -> None:
    """Render dashboard header with last updated time."""
    st.title("🌫️ Karachi AQI Forecast Dashboard")
    st.caption(
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Station {config.AQICN_STATION} | {config.TIMEZONE}"
    )


def render_current(data: dict) -> None:
    """Render current conditions metric row.

    Args:
        data: Dict with aqi and weather from fetch_current_local.
    """
    aqi_data = data.get("aqi") or {}
    weather = data.get("weather") or {}
    aqi_val = aqi_data.get("aqi") or 0
    category = get_aqi_category(aqi_val)
    color = aqi_color(aqi_val)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<p class="aqi-metric" style="color:{color}">{aqi_val}</p>',
            unsafe_allow_html=True,
        )
        st.write("Current AQI")
    with c2:
        st.markdown(f"**{category}**")
        st.write("AQI Category")
    with c3:
        pm25 = aqi_data.get("pm25", "N/A")
        st.metric("PM2.5", f"{pm25} µg/m³" if pm25 != "N/A" else "N/A")
    with c4:
        temp = weather.get("temperature_2m", "N/A")
        hum = weather.get("relative_humidity_2m", "N/A")
        st.metric("Temp / Humidity", f"{temp}°C / {hum}%" if temp != "N/A" else "N/A")


def render_alerts(aqi_val: float, forecast: dict) -> None:
    """Render alert banner when conditions warrant.

    Args:
        aqi_val: Current AQI.
        forecast: Forecast dict from predict_forecast.
    """
    preds = forecast.get("predictions", {})
    alerts = check_alerts(
        current_aqi=aqi_val,
        forecast_24h=preds.get("24h"),
        forecast_48h=preds.get("48h"),
        forecast_72h=preds.get("72h"),
    )
    if aqi_val > config.AQI_THRESHOLDS["unhealthy_sensitive"] or alerts:
        for alert in alerts:
            st.markdown(
                f'<div class="alert-banner"><strong>{alert["message"]}</strong><br>'
                f'{alert["recommendation"]}</div>',
                unsafe_allow_html=True,
            )
        if not alerts and aqi_val > 150:
            st.error("⚠️ Unhealthy — limit outdoor activity. Sensitive groups should stay indoors.")


def render_forecast_cards(forecast: dict) -> None:
    """Render 24/48/72h forecast cards.

    Args:
        forecast: Forecast result dict.
    """
    preds = forecast.get("predictions", {})
    horizons = [("24h", "+24 Hours"), ("48h", "+48 Hours"), ("72h", "+72 Hours")]
    cols = st.columns(3)
    for col, (key, label) in zip(cols, horizons):
        val = preds.get(key, 0)
        with col:
            st.markdown(
                f"""
                <div style="background:#1a1f2e;padding:1.5rem;border-radius:12px;
                border-top:4px solid {aqi_color(val)};">
                <h3 style="margin:0;color:{aqi_color(val)}">{val:.0f}</h3>
                <p style="margin:0.5rem 0 0 0;">{label}</p>
                <small>{get_aqi_category(val)}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history_chart(history: pd.DataFrame, forecast: dict) -> None:
    """Render historical + forecast Plotly chart with AQI bands.

    Args:
        history: Historical AQI DataFrame.
        forecast: Forecast dict.
    """
    fig = go.Figure()

    for cat in config.AQI_CATEGORIES:
        fig.add_hrect(
            y0=cat["min"],
            y1=cat["max"],
            fillcolor=cat["color"],
            opacity=0.08,
            line_width=0,
            annotation_text=cat["label"],
        )

    if not history.empty and "timestamp" in history.columns:
        hist = history.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
        fig.add_trace(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["aqi"],
                mode="lines",
                name="Actual AQI",
                line=dict(color="#60a5fa", width=2),
                hovertemplate="%{x}<br>AQI: %{y}<extra></extra>",
            )
        )

    preds = forecast.get("predictions", {})
    if preds:
        now = pd.Timestamp.now(tz="UTC")
        fc_times = [now + pd.Timedelta(hours=h) for h in [24, 48, 72]]
        fc_vals = [preds.get("24h"), preds.get("48h"), preds.get("72h")]
        fig.add_trace(
            go.Scatter(
                x=fc_times,
                y=fc_vals,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#f97316", width=2, dash="dash"),
                hovertemplate="%{x}<br>Forecast AQI: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="AQI History & 3-Day Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title="AQI",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pollutants(aqi_data: dict) -> None:
    """Render pollutant breakdown bar chart.

    Args:
        aqi_data: Current AQI dict.
    """
    pollutants = ["pm25", "pm10", "no2", "o3", "co", "so2"]
    labels = ["PM2.5", "PM10", "NO₂", "O₃", "CO", "SO₂"]
    values = [aqi_data.get(p) or 0 for p in pollutants]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#60a5fa", "#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6"],
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Current Pollutant Levels",
        yaxis_title="Concentration",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(importance_df: pd.DataFrame) -> None:
    """Render horizontal SHAP importance bar chart.

    Args:
        importance_df: DataFrame with feature and importance columns.
    """
    if importance_df.empty:
        st.info("Feature importance will appear after the training pipeline runs.")
        return
    top = importance_df.head(10).sort_values("importance")
    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color="#34d399",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Top 10 Feature Importances (SHAP)",
        xaxis_title="Mean |SHAP|",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_legend() -> None:
    """Render AQI category legend table."""
    st.subheader("AQI Category Guide")
    legend_df = pd.DataFrame(
        [
            {
                "Category": c["label"],
                "Range": f"{c['min']}–{c['max']}",
                "Health Implications": c["health"],
            }
            for c in config.AQI_CATEGORIES
        ]
    )
    st.dataframe(legend_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Run Streamlit dashboard."""
    render_header()

    try:
        current = fetch_current_local()
        forecast = fetch_forecast_local()
        history = fetch_history_local(days=7)
        importance = fetch_importance_local()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    aqi_val = (current.get("aqi") or {}).get("aqi") or 0

    st.subheader("Current Conditions")
    render_current(current)

    if forecast:
        render_alerts(aqi_val, forecast)

    st.subheader("3-Day Forecast")
    if forecast:
        render_forecast_cards(forecast)
    else:
        st.warning("Forecast unavailable. Run feature + training pipelines first.")

    st.subheader("Historical Trend")
    render_history_chart(history, forecast or {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pollutant Breakdown")
        render_pollutants(current.get("aqi") or {})
    with col2:
        st.subheader("Feature Importance")
        render_feature_importance(importance)

    render_legend()


if __name__ == "__main__":
    main()
