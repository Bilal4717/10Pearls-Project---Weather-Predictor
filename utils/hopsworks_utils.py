"""Shared Hopsworks connection and feature group helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional

import hopsworks
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Module-level caches to avoid repeated logins during a single run.
_PROJECT = None
_FEATURE_STORE = None
_FEATURE_GROUP = None


def get_project():
    """Log in to Hopsworks and return the (cached) project.

    ``hopsworks.login`` returns a ``Project`` object directly, so the project
    name is passed to ``login`` rather than fetched from a separate connection.

    Returns:
        Hopsworks Project instance.

    Raises:
        ValueError: If ``HOPSWORKS_API_KEY`` is not set.
    """
    global _PROJECT
    if _PROJECT is None:
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            raise ValueError("HOPSWORKS_API_KEY environment variable is required.")
        _PROJECT = hopsworks.login(
            api_key_value=api_key,
            project=config.HOPSWORKS_PROJECT,
        )
    return _PROJECT


def get_feature_store():
    """Return the project's (cached) feature store handle.

    Returns:
        Hopsworks FeatureStore instance.
    """
    global _FEATURE_STORE
    if _FEATURE_STORE is None:
        _FEATURE_STORE = get_project().get_feature_store()
    return _FEATURE_STORE


def get_or_create_feature_group():
    """Get the existing feature group or create it with the standard schema.

    The current Hopsworks SDK returns ``None`` from ``get_feature_group`` when
    the group does not exist (rather than raising), so the ``None`` case is
    handled explicitly before creating.

    Returns:
        Feature group object (never ``None``).
    """
    global _FEATURE_GROUP
    if _FEATURE_GROUP is not None:
        return _FEATURE_GROUP

    fs = get_feature_store()
    fg = None
    try:
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
    except Exception as exc:
        logger.info("get_feature_group raised (%s); will create.", exc)
        fg = None

    if fg is None:
        logger.info(
            "Creating feature group %s v%s",
            config.FEATURE_GROUP_NAME,
            config.FEATURE_GROUP_VERSION,
        )
        fg = fs.create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            description="Hourly AQI + weather features for Karachi forecasting",
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=False,
        )

    _FEATURE_GROUP = fg
    return fg


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
