"""Central configuration for the Karachi AQI prediction system."""

from __future__ import annotations

from typing import Dict, List

# Location
CITY: str = "karachi"
# AQICN no longer has a live Karachi station (US Consulate @11790 stopped
# reporting in 2025). Open-Meteo Air Quality is the primary AQI source; AQICN
# (@11790) is kept only as an optional fallback.
AQICN_STATION: str = "@11790"
LATITUDE: float = 24.8607
LONGITUDE: float = 67.0011
TIMEZONE: str = "Asia/Karachi"

# Primary AQI source toggle
USE_OPENMETEO_AQI: bool = True

# Forecast horizons (hours ahead)
FORECAST_HORIZONS: List[int] = [24, 48, 72]

# AQI category thresholds (US EPA scale)
AQI_THRESHOLDS: Dict[str, int] = {
    "good": 50,
    "moderate": 100,
    "unhealthy_sensitive": 150,
    "unhealthy": 200,
    "very_unhealthy": 300,
    "hazardous": 500,
}

# AQI category labels and colors for visualization
AQI_CATEGORIES: List[Dict[str, object]] = [
    {"label": "Good", "min": 0, "max": 50, "color": "#00E400", "health": "Air quality is satisfactory."},
    {"label": "Moderate", "min": 51, "max": 100, "color": "#FFFF00", "health": "Acceptable for most; sensitive groups may experience minor issues."},
    {"label": "Unhealthy for Sensitive Groups", "min": 101, "max": 150, "color": "#FF7E00", "health": "Sensitive groups should reduce prolonged outdoor exertion."},
    {"label": "Unhealthy", "min": 151, "max": 200, "color": "#FF0000", "health": "Everyone may experience health effects; sensitive groups more serious."},
    {"label": "Very Unhealthy", "min": 201, "max": 300, "color": "#8F3F97", "health": "Health alert — everyone may experience serious effects."},
    {"label": "Hazardous", "min": 301, "max": 500, "color": "#7E0023", "health": "Health warnings of emergency conditions; entire population likely affected."},
]

# Hopsworks
HOPSWORKS_PROJECT: str = "aqi_karachi4717"
FEATURE_GROUP_NAME: str = "aqi_features"
FEATURE_GROUP_VERSION: int = 1
MODEL_NAME: str = "aqi_forecaster"
SCALER_MODEL_NAME: str = "aqi_feature_scaler"

# Targets
TARGET_COLUMNS: List[str] = ["aqi_t24", "aqi_t48", "aqi_t72"]

# Feature engineering
LAG_HOURS: List[int] = [1, 3, 6, 12, 24, 48]
ROLLING_WINDOWS: List[int] = [3, 6, 24]

# Training
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1
LSTM_SEQUENCE_LENGTH: int = 24
SHAP_SAMPLE_SIZE: int = 200
BACKFILL_BATCH_SIZE: int = 500
BACKFILL_DEFAULT_DAYS: int = 90

# API
AQICN_BASE_URL: str = "https://api.waqi.info"
OPENMETEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_AQI_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo air-quality hourly variables → internal column names
OPENMETEO_AQI_VARS: Dict[str, str] = {
    "us_aqi": "aqi",
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "carbon_monoxide": "co",
    "sulphur_dioxide": "so2",
}
# Open-Meteo air-quality history is limited to the most recent ~92 days
OPENMETEO_AQI_MAX_HISTORY_DAYS: int = 90

# Logging
LOG_DIR: str = "logs"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Retry settings
API_MAX_RETRIES: int = 3
API_RETRY_WAIT_MIN: int = 2
API_RETRY_WAIT_MAX: int = 30
