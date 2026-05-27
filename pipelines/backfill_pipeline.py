"""Backfill pipeline: populate historical training data into Hopsworks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from utils.aqicn_client import AQICNClient
from utils.feature_engineering import compute_features
from utils.hopsworks_utils import insert_features, read_feature_group
from utils.logging_config import setup_logging
from utils.openmeteo_client import OpenMeteoClient

load_dotenv(ROOT / ".env")
logger = setup_logging(__name__, "backfill_pipeline.log")


def _existing_timestamps() -> set:
    """Load timestamps already present in the feature store.

    Returns:
        Set of UTC timestamp strings (hour-floored).
    """
    df = read_feature_group()
    if df.empty:
        return set()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    return set(ts.astype(str).tolist())


def build_hourly_aqi_frame(
    start_date: datetime,
    end_date: datetime,
    aqi_client: AQICNClient,
) -> pd.DataFrame:
    """Build hourly AQI frame by looping daily AQICN fetches.

    Args:
        start_date: Start datetime (UTC).
        end_date: End datetime (UTC).
        aqi_client: AQICN client instance.

    Returns:
        Hourly AQI DataFrame (forward-filled from daily snapshots).
    """
    rows = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        reading = aqi_client.get_historical(config.AQICN_STATION, date_str)
        if reading:
            for hour in range(24):
                ts = current.replace(hour=hour, minute=0, second=0, microsecond=0)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                row = {**reading, "timestamp": ts}
                rows.append(row)
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp")


def run(start_date: str, end_date: str) -> None:
    """Execute backfill for a date range.

    Args:
        start_date: ``YYYY-MM-DD`` start.
        end_date: ``YYYY-MM-DD`` end.
    """
    existing = _existing_timestamps()
    weather_client = OpenMeteoClient()
    aqi_client = AQICNClient()

    weather_df = weather_client.get_historical(start_date, end_date)
    if weather_df is None or weather_df.empty:
        raise RuntimeError("Failed to fetch historical weather.")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, tzinfo=timezone.utc
    )

    aqi_df = build_hourly_aqi_frame(start_dt, end_dt, aqi_client)
    if aqi_df.empty:
        raise RuntimeError("No AQI data retrieved for backfill range.")

    features = compute_features(aqi_df, weather_df)
    if features.empty:
        raise RuntimeError("Feature computation produced no rows.")

    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True).dt.floor("h")
    mask = ~features["timestamp"].astype(str).isin(existing)
    features = features[mask]

    if features.empty:
        logger.info("All dates in range already exist in feature store; skipping.")
        return

    batches = [
        features.iloc[i : i + config.BACKFILL_BATCH_SIZE]
        for i in range(0, len(features), config.BACKFILL_BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc="Uploading to Hopsworks"):
        insert_features(batch, upsert=True)

    logger.info("Backfill complete: %d rows uploaded.", len(features))


def main() -> None:
    """CLI entry point."""
    default_end = datetime.now(timezone.utc).date()
    default_start = default_end - timedelta(days=config.BACKFILL_DEFAULT_DAYS)

    parser = argparse.ArgumentParser(description="AQI backfill pipeline")
    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start.isoformat(),
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end.isoformat(),
        help="End date YYYY-MM-DD",
    )
    args = parser.parse_args()

    try:
        run(args.start_date, args.end_date)
    except Exception as exc:
        logger.exception("Backfill pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
