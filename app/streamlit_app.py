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

# ---------------------------------------------------------------------------
# Design system — refined palette mirroring the revamped UI.
# Browsers render oklch fine, but we use hex so Plotly and CSS stay consistent.
# ---------------------------------------------------------------------------
PRIMARY = "#2dd4cf"
ACCENT = "#7c73f0"

# Softer AQI palette aligned to config.AQI_CATEGORIES boundaries (by index).
REFINED_AQI_COLORS = [
    "#3ddc84",  # Good
    "#ffd93d",  # Moderate
    "#ff9f45",  # Unhealthy for Sensitive Groups
    "#f0524a",  # Unhealthy
    "#c45cc7",  # Very Unhealthy
    "#b23b32",  # Hazardous
]

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #2dd4cf;
        --accent: #7c73f0;
        --muted: #93a1b5;
        --fg: #eef2f7;
        --radius: 1rem;
        --shadow-card: 0 10px 40px rgba(0,0,0,0.40);
        --shadow-elevated: 0 20px 60px rgba(0,0,0,0.50);
        --gradient-card: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0c1018;
        background-image:
            radial-gradient(1200px 600px at 15% -10%, rgba(45,212,207,0.16) 0%, rgba(12,16,24,0) 55%),
            radial-gradient(900px 500px at 100% 0%, rgba(124,115,240,0.16) 0%, rgba(12,16,24,0) 50%);
        background-attachment: fixed;
        color: var(--fg);
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
    #MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; height: 0; }
    ::selection { background: rgba(45,212,207,0.3); }

    .mono { font-family: 'JetBrains Mono', monospace; }
    .display { font-family: 'Space Grotesk', sans-serif; }
    .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
        text-transform: uppercase; letter-spacing: 0.28em; color: var(--muted);
    }

    /* ---------- Glass cards ---------- */
    .glass {
        background: var(--gradient-card);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius);
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        box-shadow: var(--shadow-card);
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden;
    }
    .glass.hoverable:hover {
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.18);
        box-shadow: var(--shadow-elevated);
    }

    /* ---------- Hero ---------- */
    .hero { padding: 2.2rem; }
    .hero-icon {
        width: 64px; height: 64px; border-radius: 18px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 2rem;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 800;
        letter-spacing: -1px; margin: 0.25rem 0 0 0; line-height: 1.05;
    }
    .text-gradient {
        background: linear-gradient(90deg, #8fe5ff, #b9b1ff);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
    }
    .hero-desc { color: var(--muted); font-size: 0.92rem; margin-top: 0.6rem; max-width: 34rem; }
    .chip {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.32rem 0.8rem; border-radius: 999px; margin: 0.2rem 0.4rem 0.2rem 0;
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.76rem; color: #cdd7e5;
    }
    .glow {
        position: absolute; border-radius: 50%; filter: blur(60px); pointer-events: none;
    }

    /* ---------- Section titles ---------- */
    .section { display: flex; align-items: center; gap: 0.8rem; margin: 2.4rem 0 1.1rem 0; }
    .section .bar {
        width: 4px; height: 2.6rem; border-radius: 4px;
        background: linear-gradient(180deg, var(--primary), var(--accent));
    }
    .section h2 {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700;
        margin: 0.1rem 0 0 0; color: var(--fg); letter-spacing: -0.3px;
    }

    /* ---------- Metric cards ---------- */
    .grid { display: grid; gap: 1rem; }
    .grid-4 { grid-template-columns: repeat(4, 1fr); }
    .grid-3 { grid-template-columns: repeat(3, 1fr); }
    .grid-6 { grid-template-columns: repeat(6, 1fr); }
    @media (max-width: 900px) {
        .grid-4, .grid-3, .grid-6 { grid-template-columns: repeat(2, 1fr); }
    }
    .metric { padding: 1.5rem; }
    .metric .top { display: flex; justify-content: space-between; align-items: flex-start; }
    .metric .value {
        font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 800;
        line-height: 1.05; margin-top: 0.9rem; letter-spacing: -1px;
    }
    .metric .unit { font-size: 1rem; font-weight: 500; color: var(--muted); margin-left: 4px; }
    .metric .sub { color: var(--muted); font-size: 0.78rem; margin-top: 0.35rem; }
    .badge {
        display: inline-flex; padding: 0.4rem 0.9rem; border-radius: 999px;
        font-size: 0.9rem; font-weight: 600;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    @keyframes pulse-glow {
        0%,100% { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
        50% { box-shadow: 0 0 0 7px transparent; opacity: 0.65; }
    }
    .pulse { animation: pulse-glow 2s ease-in-out infinite; }

    /* ---------- Forecast cards ---------- */
    .fc { padding: 1.7rem; }
    .fc .value {
        font-family: 'Space Grotesk', sans-serif; font-size: 3.6rem; font-weight: 800;
        line-height: 1; margin: 1.1rem 0 0.9rem 0; letter-spacing: -1.5px;
    }
    .fc .row { display: flex; justify-content: space-between; align-items: center; }

    /* ---------- Bars (pollutants + SHAP) ---------- */
    .bar-track { height: 6px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width 0.7s ease; }
    .bar-row { margin-bottom: 1.15rem; }
    .bar-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.45rem; }

    /* ---------- Legend grid ---------- */
    .legend-cell { padding: 1.25rem; position: relative; }
    .legend-cell .accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; }

    /* ---------- Alert ---------- */
    .alert {
        margin-top: 1.2rem; padding: 1.2rem 1.4rem; border-radius: 16px;
        border: 1px solid rgba(240,82,74,0.40);
        background: linear-gradient(135deg, rgba(150,40,35,0.55), rgba(40,30,55,0.40));
        display: flex; gap: 1rem; align-items: flex-start;
    }
    .alert .ic {
        width: 42px; height: 42px; border-radius: 12px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
        background: rgba(240,82,74,0.20);
    }

    .footer {
        margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }
    .footer p {
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
        text-transform: uppercase; letter-spacing: 0.3em; color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def category_index(aqi: float) -> int:
    """Return index into config.AQI_CATEGORIES for an AQI value.

    Args:
        aqi: Air Quality Index.

    Returns:
        Category index (0-5).
    """
    for i, cat in enumerate(config.AQI_CATEGORIES):
        if cat["min"] <= aqi <= cat["max"]:
            return i
    return len(config.AQI_CATEGORIES) - 1


def aqi_color(aqi: float) -> str:
    """Return refined hex color for an AQI value.

    Args:
        aqi: Air Quality Index.

    Returns:
        Hex color string.
    """
    return REFINED_AQI_COLORS[category_index(aqi)]


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


def _md(html: str) -> None:
    """Render an HTML block, stripping per-line indentation.

    Streamlit's Markdown parser turns lines indented 4+ spaces into code
    blocks, so we compact the HTML to a single line before rendering.

    Args:
        html: HTML markup to render.
    """
    compact = "".join(line.strip() for line in html.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


def section_title(title: str, eyebrow: str = "") -> None:
    """Render a styled section heading with optional eyebrow.

    Args:
        title: Section title text.
        eyebrow: Small uppercase label above the title.
    """
    eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    _md(f'<div class="section"><span class="bar"></span><div>{eb}<h2>{title}</h2></div></div>')


def render_header() -> None:
    """Render dashboard hero header with last updated time."""
    updated = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    _md(
        f"""
        <div class="glass hero">
            <div class="glow" style="width:320px;height:320px;top:-130px;right:-80px;
                background:radial-gradient(circle,rgba(45,212,207,0.35),transparent 70%);"></div>
            <div class="glow" style="width:280px;height:280px;bottom:-120px;left:-60px;
                background:radial-gradient(circle,rgba(124,115,240,0.32),transparent 70%);"></div>
            <div style="position:relative;display:flex;flex-wrap:wrap;gap:1.4rem;
                justify-content:space-between;align-items:center;">
                <div style="display:flex;gap:1.2rem;align-items:flex-start;">
                    <div class="hero-icon">🌬️</div>
                    <div>
                        <div class="eyebrow">Air Quality Intelligence</div>
                        <h1 class="hero-title"><span class="text-gradient">Karachi</span> AQI Forecast</h1>
                        <p class="hero-desc">Real-time particulate monitoring and 72-hour predictive
                        modeling for Pakistan's largest metropolitan area.</p>
                    </div>
                </div>
                <div style="text-align:right;">
                    <span class="chip">🕒 {updated}</span>
                    <span class="chip">📍 Station {config.AQICN_STATION}</span>
                    <span class="chip">🌐 {config.TIMEZONE}</span>
                </div>
            </div>
        </div>
        """
    )


def render_current(data: dict) -> None:
    """Render current conditions metric cards.

    Args:
        data: Dict with aqi and weather from fetch_current_local.
    """
    aqi_data = data.get("aqi") or {}
    weather = data.get("weather") or {}
    aqi_val = aqi_data.get("aqi") or 0
    idx = category_index(aqi_val)
    category = config.AQI_CATEGORIES[idx]["label"]
    health = config.AQI_CATEGORIES[idx]["health"]
    color = REFINED_AQI_COLORS[idx]

    pm25 = aqi_data.get("pm25")
    pm10 = aqi_data.get("pm10")
    pm25_str = f"{pm25:.1f}" if pm25 is not None else "N/A"
    pm10_sub = f"PM10 · {pm10:.0f} µg/m³" if pm10 is not None else "Particulate matter"

    temp = weather.get("temperature_2m")
    hum = weather.get("relative_humidity_2m")
    wind = weather.get("wind_speed_10m")
    temp_str = f"{temp:.1f}°" if temp is not None else "N/A"
    weather_sub_parts = []
    if hum is not None:
        weather_sub_parts.append(f"Humidity {hum:.0f}%")
    if wind is not None:
        weather_sub_parts.append(f"Wind {wind:.0f} km/h")
    weather_sub = " · ".join(weather_sub_parts) or "Live conditions"

    _md(
        f"""
        <div class="grid grid-4">
            <div class="glass metric hoverable">
                <div class="glow" style="width:130px;height:130px;top:-40px;right:-40px;
                    background:radial-gradient(circle,{color}55,transparent 70%);"></div>
                <div class="top"><span class="eyebrow">Current AQI</span><span>📈</span></div>
                <div class="value" style="color:{color}">{aqi_val:.0f}</div>
                <div class="sub">US EPA scale</div>
            </div>
            <div class="glass metric hoverable">
                <div class="top"><span class="eyebrow">Air Quality</span>
                    <span class="dot pulse" style="background:{color};color:{color}"></span></div>
                <div style="margin-top:1.1rem;">
                    <span class="badge" style="background:{color}22;border:1px solid {color}66;color:{color}">{category}</span>
                </div>
                <div class="sub" style="margin-top:0.8rem;line-height:1.5;">{health}</div>
            </div>
            <div class="glass metric hoverable">
                <div class="top"><span class="eyebrow">PM2.5</span><span>🌫️</span></div>
                <div class="value">{pm25_str}<span class="unit">µg/m³</span></div>
                <div class="sub">{pm10_sub}</div>
            </div>
            <div class="glass metric hoverable">
                <div class="top"><span class="eyebrow">Temperature</span><span>🌡️</span></div>
                <div class="value">{temp_str}</div>
                <div class="sub">{weather_sub}</div>
            </div>
        </div>
        """
    )


def render_alerts(aqi_val: float, forecast: dict) -> None:
    """Render styled alert banner when conditions warrant.

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
    if not alerts and aqi_val <= config.AQI_THRESHOLDS["unhealthy_sensitive"]:
        return

    if alerts:
        title = alerts[0].get("message", "Air quality advisory")
        rec = alerts[0].get("recommendation", "")
    else:
        title = "Unhealthy air quality detected"
        rec = (
            "Sensitive groups should remain indoors. Wear an N95 mask outdoors and "
            "avoid prolonged exertion. Air purifiers recommended in living areas."
        )
    _md(
        f"""
        <div class="alert">
            <div class="ic">⚠️</div>
            <div>
                <p class="display" style="font-size:1.05rem;font-weight:600;margin:0;color:var(--fg);">{title}</p>
                <p style="margin:0.4rem 0 0 0;font-size:0.9rem;color:var(--muted);line-height:1.55;">{rec}</p>
            </div>
        </div>
        """
    )


def render_forecast_cards(forecast: dict, current_aqi: float) -> None:
    """Render 24/48/72h forecast cards with trend deltas.

    Args:
        forecast: Forecast result dict.
        current_aqi: Current AQI for the first delta comparison.
    """
    preds = forecast.get("predictions", {})
    horizons = [("24h", "Tomorrow", "+24 hours"), ("48h", "In 2 Days", "+48 hours"),
                ("72h", "In 3 Days", "+72 hours")]

    cards = ""
    prev = current_aqi
    for key, label, sub in horizons:
        val = preds.get(key) or 0
        color = aqi_color(val)
        category = get_aqi_category(val)
        delta = val - prev
        arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "→")
        sign = "+" if delta > 0 else ""
        cards += f"""
            <div class="glass fc hoverable" style="border-top:2px solid {color};">
                <div class="glow" style="width:160px;height:160px;top:-60px;right:-60px;
                    background:radial-gradient(circle,{color}55,transparent 70%);"></div>
                <div class="row">
                    <div><div class="eyebrow">{sub}</div>
                        <div class="display" style="font-size:1.05rem;font-weight:600;margin-top:0.2rem;">{label}</div></div>
                    <span class="dot pulse" style="background:{color};color:{color}"></span>
                </div>
                <div class="value" style="color:{color}">{val:.0f}</div>
                <div class="row">
                    <span class="badge" style="background:{color}22;color:{color};font-size:0.78rem;padding:0.25rem 0.7rem;">{category}</span>
                    <span style="font-size:0.76rem;color:var(--muted);">{arrow} {sign}{delta:.0f} vs prev</span>
                </div>
            </div>
        """
        prev = val

    _md(f'<div class="grid grid-3">{cards}</div>')


def render_history_chart(history: pd.DataFrame, forecast: dict) -> None:
    """Render historical + forecast Plotly chart with AQI bands.

    Args:
        history: Historical AQI DataFrame.
        forecast: Forecast dict.
    """
    _md(
        '<div class="eyebrow" style="margin-bottom:0.2rem;">7 days actual · 3 days forecast</div>'
        '<div class="display" style="font-size:1.1rem;font-weight:600;margin-bottom:0.6rem;">AQI Trend Analysis</div>'
    )

    fig = go.Figure()

    for i, cat in enumerate(config.AQI_CATEGORIES):
        fig.add_hrect(
            y0=cat["min"],
            y1=cat["max"],
            fillcolor=REFINED_AQI_COLORS[i],
            opacity=0.06,
            line_width=0,
        )

    last_actual_time = None
    last_actual_val = None
    if not history.empty and "timestamp" in history.columns:
        hist = history.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
        hist = hist.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["aqi"],
                mode="lines",
                name="Observed",
                line=dict(color=PRIMARY, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(45,212,207,0.12)",
                hovertemplate="%{x}<br>AQI %{y:.0f}<extra></extra>",
            )
        )
        if len(hist):
            last_actual_time = hist["timestamp"].iloc[-1]
            last_actual_val = hist["aqi"].iloc[-1]

    preds = forecast.get("predictions", {})
    if preds:
        anchor_time = last_actual_time or pd.Timestamp.now(tz="UTC")
        fc_times = [anchor_time] + [anchor_time + pd.Timedelta(hours=h) for h in [24, 48, 72]]
        fc_vals = [
            last_actual_val if last_actual_val is not None else preds.get("24h"),
            preds.get("24h"),
            preds.get("48h"),
            preds.get("72h"),
        ]
        fig.add_trace(
            go.Scatter(
                x=fc_times,
                y=fc_vals,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#ffb04a", width=2.5, dash="dash"),
                marker=dict(size=7, color="#ffb04a"),
                hovertemplate="%{x}<br>Forecast AQI %{y:.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=None,
        xaxis_title=None,
        yaxis_title="AQI",
        height=400,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pollutants(aqi_data: dict) -> None:
    """Render pollutant breakdown as glowing progress bars.

    Args:
        aqi_data: Current AQI dict.
    """
    pollutants = [
        ("pm25", "PM2.5", "#2dd4cf", 250.0),
        ("pm10", "PM10", "#48b4ff", 430.0),
        ("no2", "NO₂", "#7c8cff", 200.0),
        ("o3", "O₃", "#a874ff", 240.0),
        ("co", "CO", "#d76bd0", 15000.0),
        ("so2", "SO₂", "#ff9f6b", 350.0),
    ]
    rows = ""
    for key, label, color, ref in pollutants:
        val = aqi_data.get(key)
        if val is None:
            disp, pct = "N/A", 0.0
        else:
            disp = f"{val:.1f}"
            pct = max(0.0, min(100.0, (val / ref) * 100.0))
        rows += f"""
            <div class="bar-row">
                <div class="bar-head">
                    <span style="font-size:0.88rem;font-weight:500;">{label}</span>
                    <span class="mono" style="font-size:0.85rem;color:{color};">{disp}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}aa);box-shadow:0 0 12px {color}88;"></div>
                </div>
            </div>
        """
    _md(
        f"""
        <div class="glass" style="padding:1.7rem;height:100%;">
            <div class="eyebrow">Concentration · µg/m³</div>
            <div class="display" style="font-size:1.1rem;font-weight:600;margin:0.2rem 0 1.4rem 0;">Pollutant Breakdown</div>
            {rows}
        </div>
        """
    )


def render_feature_importance(importance_df: pd.DataFrame) -> None:
    """Render ranked SHAP feature importance bars.

    Args:
        importance_df: DataFrame with feature and importance columns.
    """
    if importance_df.empty:
        _md(
            """
            <div class="glass" style="padding:1.7rem;height:100%;">
                <div class="eyebrow">Mean |SHAP| · model explainability</div>
                <div class="display" style="font-size:1.1rem;font-weight:600;margin:0.2rem 0 1.4rem 0;">Feature Importance</div>
                <p style="color:var(--muted);font-size:0.88rem;">Feature importance will appear after the training pipeline runs.</p>
            </div>
            """
        )
        return

    top = importance_df.head(8)
    max_imp = float(top["importance"].max()) or 1.0
    rows = ""
    for i, (_, row) in enumerate(top.iterrows()):
        feat = str(row["feature"])
        imp = float(row["importance"])
        pct = (imp / max_imp) * 100.0
        rows += f"""
            <div class="bar-row" style="margin-bottom:0.95rem;">
                <div class="bar-head">
                    <span class="mono" style="font-size:0.78rem;color:var(--muted);">
                        <span style="opacity:0.5;margin-right:0.5rem;">{i + 1:02d}</span>{feat}</span>
                    <span class="mono" style="font-size:0.78rem;">{imp:.3f}</span>
                </div>
                <div class="bar-track" style="height:5px;">
                    <div class="bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{PRIMARY},{ACCENT});"></div>
                </div>
            </div>
        """
    _md(
        f"""
        <div class="glass" style="padding:1.7rem;height:100%;">
            <div class="eyebrow">Mean |SHAP| · model explainability</div>
            <div class="display" style="font-size:1.1rem;font-weight:600;margin:0.2rem 0 1.4rem 0;">Feature Importance</div>
            {rows}
        </div>
        """
    )


def render_legend() -> None:
    """Render AQI category legend as a 6-cell grid."""
    cells = ""
    for i, cat in enumerate(config.AQI_CATEGORIES):
        color = REFINED_AQI_COLORS[i]
        cells += f"""
            <div class="glass legend-cell" style="border-radius:0;border:none;box-shadow:none;
                border-right:1px solid rgba(255,255,255,0.05);">
                <div class="accent" style="background:{color};box-shadow:0 0 12px {color};"></div>
                <div class="mono" style="font-size:0.66rem;letter-spacing:0.2em;color:var(--muted);">{cat['min']}–{cat['max']}</div>
                <div class="display" style="font-size:0.98rem;font-weight:600;margin-top:0.5rem;color:{color};">{cat['label']}</div>
                <div style="margin-top:0.5rem;font-size:0.74rem;line-height:1.5;color:var(--muted);">{cat['health']}</div>
            </div>
        """
    _md(f'<div class="glass" style="overflow:hidden;"><div class="grid grid-6" style="gap:0;">{cells}</div></div>')


def render_footer() -> None:
    """Render dashboard footer."""
    _md('<div class="footer"><p>Data · AQICN · Open-Meteo · Hopsworks Feature Store</p></div>')


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

    section_title("Current Conditions", "Live readings")
    render_current(current)

    if forecast:
        render_alerts(aqi_val, forecast)

    section_title("3-Day Forecast", "Predictive model")
    if forecast:
        render_forecast_cards(forecast, aqi_val)
    else:
        st.warning("Forecast unavailable. Run feature + training pipelines first.")

    section_title("Historical Trend", "Time series")
    render_history_chart(history, forecast or {})

    col1, col2 = st.columns(2)
    with col1:
        section_title("Pollutant Breakdown", "Atmospheric chemistry")
        render_pollutants(current.get("aqi") or {})
    with col2:
        section_title("Feature Importance", "Model insights")
        render_feature_importance(importance)

    section_title("AQI Category Guide", "Reference")
    render_legend()

    render_footer()


if __name__ == "__main__":
    main()
