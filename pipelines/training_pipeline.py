"""Training pipeline: train models, evaluate, SHAP, register best model in Hopsworks."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from utils.feature_engineering import get_feature_columns
from utils.hopsworks_utils import get_model_registry, read_feature_group
from utils.logging_config import setup_logging
from utils.model_utils import (
    evaluate_predictions,
    predict_lstm,
    set_seeds,
    time_based_split,
    train_lstm,
)

load_dotenv(ROOT / ".env")
logger = setup_logging(__name__, "training_pipeline.log")

ARTIFACTS_DIR = ROOT / "artifacts"
SHAP_PLOT_PATH = ARTIFACTS_DIR / "shap_summary.png"


def _print_metrics_table(results: dict) -> None:
    """Log comparison table of model metrics.

    Args:
        results: Dict mapping model name to metrics dict.
    """
    rows = []
    for name, metrics in results.items():
        for target in config.TARGET_COLUMNS:
            m = metrics[target]
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "rmse": f"{m['rmse']:.2f}",
                    "mae": f"{m['mae']:.2f}",
                    "r2": f"{m['r2']:.3f}",
                }
            )
        rows.append(
            {
                "model": name,
                "target": "AVG",
                "rmse": f"{metrics['avg_rmse']:.2f}",
                "mae": "-",
                "r2": "-",
            }
        )
    table = pd.DataFrame(rows)
    logger.info("\n%s", table.to_string(index=False))


def compute_shap(
    model,
    model_name: str,
    X_sample: np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """Compute SHAP values and save summary plot.

    Args:
        model: Fitted model (sklearn or torch LSTM wrapper).
        model_name: Model identifier for explainer selection.
        X_sample: Sample of scaled features (max 200 rows).
        feature_names: Feature column names.

    Returns:
        Mean absolute SHAP values per feature.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = X_sample[: config.SHAP_SAMPLE_SIZE]

    if model_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model.estimators_[0] if hasattr(model, "estimators_") else model)
        if hasattr(model, "estimators_"):
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = np.abs(shap_values)
        else:
            shap_values = np.abs(explainer.shap_values(sample))
    else:
        def predict_fn(x):
            return model.predict(x)

        explainer = shap.KernelExplainer(predict_fn, sample[:50])
        shap_values = explainer.shap_values(sample[:100], nsamples=50)
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

    mean_shap = np.mean(shap_values, axis=0)
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.mean(axis=1)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values if not isinstance(shap_values, list) else shap_values[0],
        sample,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s", SHAP_PLOT_PATH)
    return mean_shap


def register_models(
    model,
    scaler: StandardScaler,
    metrics: dict,
    feature_names: list,
    shap_values: np.ndarray,
) -> None:
    """Register best model and scaler in Hopsworks Model Registry.

    Args:
        model: Best fitted model.
        scaler: Fitted StandardScaler.
        metrics: Test metrics dict.
        feature_names: Feature column list.
        shap_values: Mean SHAP importance array.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.pkl"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    meta_path = ARTIFACTS_DIR / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "metrics": metrics,
        "feature_names": feature_names,
        "shap_importance": {
            feature_names[i]: float(shap_values[i])
            for i in range(min(len(feature_names), len(shap_values)))
        },
        "targets": config.TARGET_COLUMNS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    mr = get_model_registry()
    model_obj = mr.python.create_model(
        name=config.MODEL_NAME,
        metrics=metrics,
        description="Best AQI forecaster for Karachi (24/48/72h)",
    )
    model_obj.save(str(model_path))

    scaler_obj = mr.python.create_model(
        name=config.SCALER_MODEL_NAME,
        description="StandardScaler for AQI features",
    )
    scaler_obj.save(str(scaler_path))
    logger.info("Registered model and scaler in Hopsworks Model Registry.")


def run() -> None:
    """Execute full training pipeline."""
    set_seeds()
    df = read_feature_group()
    if df.empty or len(df) < 200:
        raise RuntimeError(
            f"Insufficient training data ({len(df)} rows). Run backfill first."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop feature columns that are entirely NaN (e.g. ``visibility`` is not
    # provided by the Open-Meteo historical archive API).
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        logger.warning("Dropping all-NaN feature columns: %s", all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    # Targets must be present; remaining feature NaNs (early lag/rolling rows)
    # are filled rather than dropping the entire dataset.
    df = df.dropna(subset=config.TARGET_COLUMNS).reset_index(drop=True)
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].fillna(0)

    if len(df) < 200:
        raise RuntimeError(
            f"Insufficient usable training data ({len(df)} rows after cleaning)."
        )

    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(df, feature_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}
    models = {}

    ridge = MultiOutputRegressor(Ridge(alpha=1.0, random_state=config.RANDOM_SEED))
    ridge.fit(X_train_s, y_train)
    pred = ridge.predict(X_test_s)
    results["ridge"] = evaluate_predictions(y_test, pred)
    models["ridge"] = ridge

    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
    )
    rf.fit(X_train_s, y_train)
    pred = rf.predict(X_test_s)
    results["random_forest"] = evaluate_predictions(y_test, pred)
    models["random_forest"] = rf

    xgb_estimators = []
    for i in range(len(config.TARGET_COLUMNS)):
        est = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=config.RANDOM_SEED,
            early_stopping_rounds=20,
            eval_metric="rmse",
        )
        est.fit(
            X_train_s,
            y_train[:, i],
            eval_set=[(X_val_s, y_val[:, i])],
            verbose=False,
        )
        xgb_estimators.append(est)
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor())
    xgb_model.estimators_ = xgb_estimators
    pred = np.column_stack([e.predict(X_test_s) for e in xgb_estimators])
    results["xgboost"] = evaluate_predictions(y_test, pred)
    models["xgboost"] = xgb_model

    lstm = train_lstm(X_train_s, y_train, X_val_s, y_val)
    pred_lstm = predict_lstm(lstm, X_test_s)
    offset = config.LSTM_SEQUENCE_LENGTH - 1
    y_test_lstm = y_test[offset : offset + len(pred_lstm)]
    if len(pred_lstm) > 0:
        results["lstm"] = evaluate_predictions(y_test_lstm, pred_lstm)
        models["lstm"] = lstm
    else:
        logger.warning("LSTM produced no predictions; skipping.")

    _print_metrics_table(results)

    best_name = min(results.keys(), key=lambda k: results[k]["avg_rmse"])
    best_metrics = results[best_name]
    best_model = models[best_name]
    logger.info("Best model: %s (avg RMSE=%.2f)", best_name, best_metrics["avg_rmse"])

    X_shap = X_test_s[: config.SHAP_SAMPLE_SIZE]
    shap_vals = compute_shap(best_model, best_name, X_shap, feature_cols)
    register_models(best_model, scaler, best_metrics, feature_cols, shap_vals)


def main() -> None:
    """CLI entry point."""
    try:
        run()
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
