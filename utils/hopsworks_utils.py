"""Shared Hopsworks connection and feature group helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional

import hopsworks
import pandas as pd

import config

logger = logging.getLogger(__name__)


def get_project():
    """Log in to Hopsworks and return the project.

    ``hopsworks.login`` returns a ``Project`` object directly, so the project
    name is passed to ``login`` rather than fetched from a separate connection.

    Returns:
        Hopsworks Project instance.

    Raises:
        ValueError: If ``HOPSWORKS_API_KEY`` is not set.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable is required.")
    return hopsworks.login(
        api_key_value=api_key,
        project=config.HOPSWORKS_PROJECT,
    )


def get_feature_store():
    """Return the project's feature store handle.

    Returns:
        Hopsworks FeatureStore instance.
    """
    project = get_project()
    return project.get_feature_store()


def get_or_create_feature_group(fg=None):
    """Get existing feature group or create with standard schema.

    Args:
        fg: Optional pre-fetched feature group.

    Returns:
        Feature group object.
    """
    fs = get_feature_store()
    if fg is not None:
        return fg
    try:
        return fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
    except Exception:
        logger.info("Creating feature group %s v%s", config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION)
        return fs.create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            description="Hourly AQI + weather features for Karachi forecasting",
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=True,
        )


def read_feature_group(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """Read features from Hopsworks as a pandas DataFrame.

    Args:
        start_time: Optional ISO start filter.
        end_time: Optional ISO end filter.

    Returns:
        Feature DataFrame (may be empty).
    """
    fg = get_or_create_feature_group()
    kwargs = {}
    if start_time:
        kwargs["start_time"] = start_time
    if end_time:
        kwargs["end_time"] = end_time
    try:
        df = fg.read(**kwargs)
        if df is not None and not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.warning("Failed to read feature group: %s", exc)
        return pd.DataFrame()


def insert_features(df: pd.DataFrame, upsert: bool = True) -> None:
    """Insert or upsert rows into the feature group.

    Args:
        df: Feature DataFrame with UTC timestamp.
        upsert: Whether to upsert on primary key.
    """
    if df.empty:
        logger.warning("No rows to insert.")
        return
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    fg = get_or_create_feature_group()
    fg.insert(out, write_options={"wait_for_job": True}, upsert=upsert)
    logger.info("Inserted %d rows into feature group.", len(out))


def get_model_registry():
    """Return model registry for the project.

    Returns:
        Hopsworks ModelRegistry instance.
    """
    project = get_project()
    return project.get_model_registry()
