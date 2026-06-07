

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hopsworks
import pandas as pd
from typing import Any
from dotenv import load_dotenv

from fetch_data import fetch_current
from engineer_features import engineer_features
from config import get_logger, HOPSWORKS_PROJECT, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, CITY

load_dotenv()

logger = get_logger(__name__)

def connect_hopsworks() -> hopsworks.project.Project:
    try:
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        fs = project.get_feature_store()
        logger.info("Connected to Hopsworks project: %s", project.name)
        return fs
    except Exception as e:
        logger.error("Failed to connect to Hopsworks: %s", e)
        raise

def get_or_create_feature_group(fs) -> Any:
    try:
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["city", "unix_time"],
            description=f"Hourly AQI features for {CITY.capitalize()} (OpenWeatherMap)",
            online_enabled=True,
            event_time="unix_time"
        )
        return fg
    except Exception as e:
        logger.error("Failed to get or create feature group: %s", e)
        raise

def store_features(df: pd.DataFrame, fs) -> None:
    try:
        fg = get_or_create_feature_group(fs)
        fg.insert(df, write_options={"wait_for_job": False})
        logger.info("Inserted %d row(s) into Hopsworks Feature Store!", len(df))
    except Exception as e:
        logger.error("Failed to store features: %s", e)
        raise

if __name__ == "__main__":
    logger.info("─── Hourly Feature Pipeline ───")
    try:
        logger.info("Step 1: Fetching current data ...")
        df_raw = fetch_current()

        logger.info("Step 2: Engineering features ...")
        df_eng = engineer_features(df_raw)

        logger.info("Step 3: Connecting to Hopsworks ...")
        fs = connect_hopsworks()

        logger.info("Step 4: Storing features ...")
        store_features(df_eng, fs)

        logger.info("Hourly pipeline complete!")
    except Exception as e:
        logger.error("Hourly feature pipeline failed: %s", e)
