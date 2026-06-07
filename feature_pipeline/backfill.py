import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from fetch_data import fetch_historical
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

def backfill(days: int = 365) -> None:
    logger.info("BACKFILL: Last %d days for %s", days, CITY.capitalize())
    
    try:
        logger.info("[1/4] Fetching %d days of historical data ...", days)
        df_raw = fetch_historical(days=days)
        logger.info("Raw rows fetched: %d", len(df_raw))

        logger.info("[2/4] Engineering features ...")
        df_eng = engineer_features(df_raw)
        logger.info("Engineered shape: %s", df_eng.shape)

        logger.info("[3/4] Connecting to Hopsworks ...")
        fs = connect_hopsworks()

        logger.info("[4/4] Inserting into Feature Store ...")
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["city", "unix_time"],
            description=f"Hourly AQI features for {CITY.capitalize()} (OpenWeatherMap)",
            online_enabled=True,
            event_time="unix_time"
        )
        fg.insert(df_eng, write_options={"wait_for_job": False})

        logger.info("Backfill complete! %d rows stored in Hopsworks.", len(df_eng))
        logger.info("Date range: %s  ->  %s", df_eng['timestamp'].min(), df_eng['timestamp'].max())
        logger.info("AQI Distribution:\n%s", df_eng["aqi_category"].value_counts().to_string())

    except Exception as e:
        logger.error("Backfill failed: %s", e)
        raise

if __name__ == "__main__":
    backfill(days=365)
