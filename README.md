# Karachi AQI Forecast — Serverless ML System

End-to-end Air Quality Index (AQI) prediction system for **Karachi, Pakistan**. Predicts AQI **24, 48, and 72 hours** ahead using live pollutant data, weather features, and automated ML pipelines on Hopsworks.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  AQICN API      │     │  Open-Meteo API  │     │  GitHub Actions     │
│  (AQI/pollut.)  │     │  (weather)       │     │  (hourly + daily)   │
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

## Prerequisites

- Python 3.11+
- [AQICN API token](https://aqicn.org/data-platform/token/)
- [Hopsworks](https://www.hopsworks.ai/) account (free tier)
- GitHub repository (for Actions)

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
python pipelines/backfill_pipeline.py --start-date 2025-02-01 --end-date 2025-05-27
```

Default: last 90 days.

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
4. Workflows run automatically:
   - **Feature Pipeline**: every hour (`0 * * * *`)
   - **Training Pipeline**: daily at 02:00 UTC (`0 2 * * *`)
5. Use **Actions → workflow → Run workflow** for manual runs

## Deploy Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set **Main file path**: `app/streamlit_app.py`
5. Add secrets in Streamlit Cloud:
   - `AQICN_TOKEN`
   - `HOPSWORKS_API_KEY`
6. Deploy

## Models

| Model | Description |
|-------|-------------|
| Ridge | Baseline linear multi-output |
| Random Forest | 200 trees, max_depth=15 |
| XGBoost | 300 estimators, early stopping |
| LSTM | 2-layer PyTorch, seq_len=24 |

Best model (lowest average RMSE across t24/t48/t72) is registered with SHAP explainability plot.

## Karachi Notes

- Station: `@7064` on AQICN
- Coordinates: 24.8607°N, 67.0011°E
- Timezone: Asia/Karachi (UTC+5)
- Typical AQI: 100–200 (traffic + industrial)
- Monsoon (Jul–Sep) affects humidity and dispersion

## License

MIT — Internship / educational use.
