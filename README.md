# Karachi AQI Forecast — Serverless ML System

End-to-end Air Quality Index (AQI) prediction system for **Karachi, Pakistan**. Predicts AQI **24, 48, and 72 hours** ahead using live pollutant data, weather features, and automated ML pipelines on Hopsworks.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://10pearls-project---weather-predictor-phjmr4gzi5sbu98dzuqksa.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-1EB182)](https://www.hopsworks.ai/)

> **🔴 Live dashboard:** **https://10pearls-project---weather-predictor-phjmr4gzi5sbu98dzuqksa.streamlit.app/**

## Highlights

- **3-day AQI forecasts** (24/48/72h) for Karachi with confidence intervals and health alerts.
- **Fully serverless & automated** — GitHub Actions run the hourly feature ingestion and daily retraining; no servers to manage.
- **Model selection** across Ridge, Random Forest, XGBoost, and an LSTM, with the best model (lowest avg RMSE) auto-registered.
- **Explainability** via SHAP feature importances surfaced in the dashboard.
- **Modern glassmorphism dashboard** built in Streamlit, deployed free on Streamlit Cloud.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Open-Meteo AQ   │     │  Open-Meteo API  │     │  GitHub Actions     │
│ (AQI/pollut.)   │     │  (weather)       │     │  (hourly + daily)   │
└────────┬────────┘     └────────┬─────────┘     └──────────┬──────────┘
         │                       │                          │
         └───────────┬───────────┘                          │
                     ▼                                      ▼
            ┌─────────────────┐              ┌─────────────────────────┐
            │ Feature Pipeline │◄─────────────│  Backfill Pipeline      │
            │ (hourly)         │              │  (one-time / manual)    │
            └────────┬─────────┘              └─────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Hopsworks       │
            │ Feature Store   │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Training        │
            │ Pipeline        │
            │ Ridge/RF/XGB/   │
            │ LSTM + SHAP     │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Hopsworks       │
            │ Model Registry  │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ FastAPI         │    │ Streamlit       │
│ /forecast       │    │ Dashboard       │
│ /current        │    │ (Streamlit Cloud)│
└─────────────────┘    └─────────────────┘
```

## Repository Structure

```
aqi-predictor/
├── .github/workflows/     # CI/CD (hourly features, daily training)
├── pipelines/             # Feature, backfill, training scripts
├── app/                   # FastAPI + Streamlit
├── notebooks/eda.ipynb    # Exploratory analysis
├── utils/                 # API clients, FE, alerts, inference
├── config.py              # Constants and thresholds
├── requirements.txt
└── .env.example
```

## Data Sources

| Source | Role | Auth |
|--------|------|------|
| [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api) | **Primary** AQI + pollutants (PM2.5/PM10/NO₂/O₃/CO/SO₂, US AQI) for Karachi — hourly history + forecast | None |
| [Open-Meteo Weather](https://open-meteo.com/) | Weather features (temp, humidity, wind, etc.) | None |
| [AQICN](https://aqicn.org/) | **Optional fallback** AQI source (`@11790`, Karachi US Consulate) | Token |

> **Why Open-Meteo for AQI?** AQICN no longer has a live Karachi station (the US Consulate station `@11790` stopped reporting in 2025), so Open-Meteo's Air Quality API is used as the reliable primary source. Set `USE_OPENMETEO_AQI = False` in `config.py` to prefer AQICN.

## Prerequisites

- Python 3.11+ (3.12 also works locally; **avoid 3.14** — no wheels yet for `torch`/`hopsworks`)
- [Hopsworks](https://www.hopsworks.ai/) account (free tier)
- (Optional) [AQICN API token](https://aqicn.org/data-platform/token/) for the fallback source
- GitHub repository (for Actions)

> **Windows note:** `hopsworks` requires Microsoft C++ Build Tools to install locally (its `twofish` dependency is compiled). It installs cleanly on GitHub Actions (Linux), so the recommended workflow is to run the backfill/training pipelines via Actions.

## Setup

### 1. Clone and install

```bash
cd aqi-predictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
AQICN_TOKEN=your_aqicn_token
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

### 3. Create Hopsworks project

1. Log in to [Hopsworks](https://cloud.hopsworks.ai/)
2. Create project named **`aqi_karachi4717`** (must match `config.py`)
3. Copy API key from **User Settings → API Keys**

### 4. Backfill historical data

```bash
python pipelines/backfill_pipeline.py
```

Default: last 90 days. Open-Meteo Air Quality only retains ~92 days of history, so keep the range within the last 90 days. (On Windows, prefer running the **Backfill Pipeline** GitHub Action instead — see below.)

### 5. Run feature pipeline (manual test)

```bash
python pipelines/feature_pipeline.py
```

### 6. Train models

```bash
python pipelines/training_pipeline.py
```

Artifacts are saved to `artifacts/` and registered in Hopsworks Model Registry.

## Running Locally

### FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Streamlit

```bash
streamlit run app/streamlit_app.py
```

Set `API_BASE_URL` in Streamlit secrets if using remote API.

## Dashboard

The Streamlit dashboard (**[live here](https://10pearls-project---weather-predictor-phjmr4gzi5sbu98dzuqksa.streamlit.app/)**) presents:

- **Hero header** with last-updated time, station, and timezone.
- **Current conditions** — live AQI, category, PM2.5, temperature/humidity/wind.
- **Health alerts** that appear automatically when AQI is unhealthy.
- **3-day forecast cards** (24/48/72h) with category and trend vs. previous horizon.
- **Historical trend chart** — 7 days observed + 3-day forecast with AQI category bands.
- **Pollutant breakdown** — PM2.5/PM10/NO₂/O₃/CO/SO₂ concentration bars.
- **Feature importance** — top SHAP features from the registered model.
- **AQI category guide** — US EPA reference table.

> Data is cached for 1 hour and refreshes automatically. The dashboard runs on live Open-Meteo data plus the registered model, so it stays available even if a pipeline run fails.

## API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/current` | GET | Live AQI + weather from APIs |
| `/forecast` | GET | 24/48/72h predictions + confidence |
| `/history?days=7` | GET | Historical AQI from feature store |
| `/feature_importance` | GET | Top SHAP features |

### Example: Forecast

```bash
curl http://localhost:8000/forecast
```

```json
{
  "predictions": {"24h": 142.0, "48h": 155.0, "72h": 168.0},
  "timestamp": "2025-05-27T12:00:00+00:00",
  "confidence_low": {"24h": 127.0, "48h": 140.0, "72h": 153.0},
  "confidence_high": {"24h": 157.0, "48h": 170.0, "72h": 183.0},
  "alerts": []
}
```

## GitHub Actions Setup

1. Push repo to GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add secrets:
   - `AQICN_TOKEN`
   - `HOPSWORKS_API_KEY`
4. Workflows:
   - **Backfill Pipeline**: manual (`workflow_dispatch`) — run this **first** to populate history. Optional `start_date` / `end_date` inputs (defaults to last 90 days).
   - **Feature Pipeline**: every hour (`0 * * * *`)
   - **Training Pipeline**: daily at 02:00 UTC (`0 2 * * *`)
5. Use **Actions → workflow → Run workflow** for manual runs

**Recommended first run order (all in GitHub Actions):**
1. **Backfill Pipeline** → populates ~90 days of hourly features
2. **Feature Pipeline** → confirms hourly ingestion works
3. **Training Pipeline** → trains and registers the best model

## Deploy Streamlit Cloud

**Live deployment:** https://10pearls-project---weather-predictor-phjmr4gzi5sbu98dzuqksa.streamlit.app/

To deploy your own instance:

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set **Main file path**: `app/streamlit_app.py`
5. Under **Advanced settings**, choose **Python 3.11** (3.14 lacks wheels for `hopsworks`/`confluent-kafka`)
6. Add secrets in Streamlit Cloud:
   - `AQICN_TOKEN`
   - `HOPSWORKS_API_KEY`
7. Deploy

## Models

| Model | Description |
|-------|-------------|
| Ridge | Baseline linear multi-output |
| Random Forest | 200 trees, max_depth=15 |
| XGBoost | 300 estimators, early stopping |
| LSTM | 2-layer PyTorch, seq_len=24 |

Best model (lowest average RMSE across t24/t48/t72) is registered with SHAP explainability plot.

## Karachi Notes

- Primary AQI source: Open-Meteo Air Quality API (coordinates-based, no station needed)
- AQICN fallback station: `@11790` (Karachi US Consulate, currently inactive)
- Coordinates: 24.8607°N, 67.0011°E
- Timezone: Asia/Karachi (UTC+5)
- Typical AQI: 100–200 (traffic + industrial)
- Monsoon (Jul–Sep) affects humidity and dispersion

## License

MIT — Internship / educational use.
