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

# Theme + custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(1200px 600px at 15% -10%, #16203a 0%, #0b0f1a 45%, #080a12 100%);
        color: #e7ecf3;
    }

    /* Tighten default top padding */
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* Hide Streamlit chrome for a cleaner look */
    #MainMenu, footer, header [data-testid="stHeader"] { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    /* ---------- Hero header ---------- */
    .hero {
        display: flex; align-items: center; gap: 1rem;
        padding: 1.6rem 1.8rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(56,189,248,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
        margin-bottom: 0.4rem;
    }
    .hero-icon { font-size: 2.8rem; line-height: 1; }
    .hero-title {
        font-size: 2.1rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #c7d2fe, #7dd3fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-sub { color: #9aa6b8; font-size: 0.9rem; margin-top: 0.2rem; }
    .hero-sub .chip {
        display: inline-block; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 2px 10px; border-radius: 999px; margin-right: 6px; font-size: 0.78rem;
    }

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #f1f5f9;
        margin: 1.8rem 0 0.9rem 0; display: flex; align-items: center; gap: 0.5rem;
    }
    .section-title::before {
        content: ""; width: 4px; height: 1.05rem; border-radius: 4px;
        background: linear-gradient(180deg, #818cf8, #38bdf8);
    }

    /* ---------- Metric cards ---------- */
    .card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.2rem 1.3rem; height: 100%;
        backdrop-filter: blur(6px);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .card:hover { transform: translateY(-3px); border-color: rgba(255,255,255,0.16); }
    .card-label { color: #9aa6b8; font-size: 0.82rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.45rem; }
    .card-value { font-size: 2.1rem; font-weight: 800; line-height: 1.1; }
    .card-unit { font-size: 0.95rem; font-weight: 500; color: #9aa6b8; margin-left: 4px; }
    .card-badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600;
    }

    /* ---------- Forecast cards ---------- */
    .fc-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.4rem; text-align: center;
        transition: transform 0.15s ease;
    }
    .fc-card:hover { transform: translateY(-3px); }
    .fc-value { font-size: 2.6rem; font-weight: 800; line-height: 1; }
    .fc-label { color: #cbd5e1; font-weight: 600; margin-top: 0.5rem; }
    .fc-cat { color: #9aa6b8; font-size: 0.82rem; margin-top: 0.2rem; }
    .fc-ring {
        width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-bottom: 0.6rem;
    }

    .alert-banner {
        background: linear-gradient(135deg, rgba(127,29,29,0.85), rgba(153,27,27,0.65));
        padding: 1rem 1.2rem; border-radius: 14px;
        border-left: 4px solid #ef4444; margin: 1rem 0;
        box-shadow: 0 6px 24px rgba(239,68,68,0.15);
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


def section_title(label: str) -> None:
    """Render a styled section heading.

    Args:
        label: Section title text.
    """
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)


def render_header() -> None:
    """Render dashboard hero header with last updated time."""
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">🌫️</div>
            <div>
                <p class="hero-title">Karachi AQI Forecast</p>
                <div class="hero-sub">
                    <span class="chip">🕒 {updated}</span>
                    <span class="chip">📍 Station {config.AQICN_STATION}</span>
                    <span class="chip">🌐 {config.TIMEZONE}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

    pm25 = aqi_data.get("pm25")
    pm25_str = f"{pm25:.2f}" if pm25 is not None else "N/A"
    temp = weather.get("temperature_2m")
    hum = weather.get("relative_humidity_2m")
    temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"
    hum_str = f"{hum:.0f}%" if hum is not None else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Current AQI</div>
                <div class="card-value" style="color:{color}">{aqi_val:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Air Quality</div>
                <div style="margin-top:0.4rem">
                    <span class="card-badge" style="background:{color}22;color:{color};
                    border:1px solid {color}55">{category}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">PM2.5</div>
                <div class="card-value">{pm25_str}<span class="card-unit">µg/m³</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Temp / Humidity</div>
                <div class="card-value">{temp_str}
                    <span class="card-unit">/ {hum_str}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
    horizons = [("24h", "Tomorrow", "+24h"), ("48h", "In 2 Days", "+48h"), ("72h", "In 3 Days", "+72h")]
    cols = st.columns(3)
    for col, (key, label, sub) in zip(cols, horizons):
        val = preds.get(key, 0)
        color = aqi_color(val)
        with col:
            st.markdown(
                f"""
                <div class="fc-card" style="border-top:3px solid {color}">
                    <span class="fc-ring" style="background:{color};box-shadow:0 0 12px {color}"></span>
                    <div class="fc-value" style="color:{color}">{val:.0f}</div>
                    <div class="fc-label">{label} <span style="color:#64748b">· {sub}</span></div>
                    <div class="fc-cat">{get_aqi_category(val)}</div>
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
        title=None,
        xaxis_title="Time (UTC)",
        yaxis_title="AQI",
        height=420,
        margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
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
        title=None,
        yaxis_title="Concentration",
        height=340,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
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
        title=None,
        xaxis_title="Mean |SHAP|",
        height=340,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_legend() -> None:
    """Render AQI category legend table."""
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

    section_title("Current Conditions")
    render_current(current)

    if forecast:
        render_alerts(aqi_val, forecast)

    section_title("3-Day Forecast")
    if forecast:
        render_forecast_cards(forecast)
    else:
        st.warning("Forecast unavailable. Run feature + training pipelines first.")

    section_title("Historical Trend")
    render_history_chart(history, forecast or {})

    col1, col2 = st.columns(2)
    with col1:
        section_title("Pollutant Breakdown")
        render_pollutants(current.get("aqi") or {})
    with col2:
        section_title("Feature Importance")
        render_feature_importance(importance)

    section_title("AQI Category Guide")
    render_legend()


if __name__ == "__main__":
    main()
