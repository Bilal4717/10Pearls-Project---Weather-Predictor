"""Model loading and prediction utilities for API and dashboard."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from utils.feature_engineering import build_single_row_features, get_feature_columns
from utils.hopsworks_utils import get_model_registry, read_feature_group

logger = logging.getLogger(__name__)

# torch/LSTM support is optional at inference time. The deployed dashboard and
# API only need it if the registered best model is an LSTM; otherwise (e.g.
# XGBoost) we avoid importing torch so lightweight deployments don't need it.
try:
    from utils.model_utils import LSTMRegressor, predict_lstm

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch not installed in slim envs
    LSTMRegressor = None
    predict_lstm = None
    _TORCH_AVAILABLE = False

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _load_local_artifacts() -> Tuple[Any, StandardScaler, Dict]:
    """Load model, scaler, and metadata from local artifacts directory.

    Returns:
        Tuple of (model, scaler, metadata dict).
    """
    with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def load_model_from_registry() -> Tuple[Any, StandardScaler, Dict]:
    """Load model from Hopsworks registry, falling back to local artifacts.

    Returns:
        Tuple of (model, scaler, metadata).
    """
    try:
        mr = get_model_registry()
        model_meta = mr.get_model(name=config.MODEL_NAME, version=None)
        scaler_meta = mr.get_model(name=config.SCALER_MODEL_NAME, version=None)
        model_dir = model_meta.download()
        scaler_dir = scaler_meta.download()
        model_path = list(Path(model_dir).glob("*.pkl"))[0]
        scaler_path = list(Path(scaler_dir).glob("*.pkl"))[0]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        metadata = {"feature_names": [], "shap_importance": {}, "metrics": {}}
        if (ARTIFACTS_DIR / "metadata.json").exists():
            with open(ARTIFACTS_DIR / "metadata.json", encoding="utf-8") as f:
                metadata = json.load(f)
        return model, scaler, metadata
    except Exception as exc:
        logger.warning("Registry load failed (%s); using local artifacts.", exc)
        return _load_local_artifacts()


def predict_forecast(
    current_aqi: dict,
    current_weather: dict,
) -> Dict[str, Any]:
    """Generate 24/48/72h AQI forecasts with confidence intervals.

    Args:
        current_aqi: Current AQI reading.
        current_weather: Current weather reading.

    Returns:
        Dict with predictions, timestamp, and confidence bounds.
    """
    history = read_feature_group()
    features = build_single_row_features(current_aqi, current_weather, history)
    if features is None or features.empty:
        raise ValueError("Insufficient historical data for lag features.")

    model, scaler, metadata = load_model_from_registry()
    feature_names = metadata.get("feature_names") or get_feature_columns(features)
    X = features[feature_names].values

    X_scaled = scaler.transform(X)

    if _TORCH_AVAILABLE and LSTMRegressor is not None and isinstance(model, LSTMRegressor):
        preds = predict_lstm(model, X_scaled)
        pred = preds[-1] if len(preds) else np.zeros(3)
    elif hasattr(model, "predict"):
        pred = model.predict(X_scaled)[0]
    else:
        raise ValueError("Unsupported model type for inference.")

    metrics = metadata.get("metrics", {})
    avg_rmse = np.mean(
        [
            metrics.get(t, {}).get("rmse", 15.0)
            for t in config.TARGET_COLUMNS
        ]
    )

    result = {
        "predictions": {
            "24h": float(pred[0]),
            "48h": float(pred[1]),
            "72h": float(pred[2]),
        },
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "confidence_low": {
            "24h": float(max(0, pred[0] - avg_rmse)),
            "48h": float(max(0, pred[1] - avg_rmse)),
            "72h": float(max(0, pred[2] - avg_rmse)),
        },
        "confidence_high": {
            "24h": float(pred[0] + avg_rmse),
            "48h": float(pred[1] + avg_rmse),
            "72h": float(pred[2] + avg_rmse),
        },
    }
    return result


def get_feature_importance() -> List[Dict[str, Any]]:
    """Return top SHAP feature importances from metadata.

    Returns:
        List of {feature, importance} dicts sorted descending.
    """
    try:
        _, _, metadata = load_model_from_registry()
        shap_imp = metadata.get("shap_importance", {})
        items = [
            {"feature": k, "importance": v}
            for k, v in shap_imp.items()
        ]
        items.sort(key=lambda x: x["importance"], reverse=True)
        return items[:10]
    except Exception as exc:
        logger.warning("Could not load feature importance: %s", exc)
        return []
