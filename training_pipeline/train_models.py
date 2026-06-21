

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hopsworks
import pandas as pd
import numpy as np
from typing import Tuple, Any
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from config import get_logger, FEATURE_COLS, TARGET_COL, HOPSWORKS_PROJECT, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

load_dotenv()
logger = get_logger(__name__)

FEATURE_VIEW_NAME = "aqi_features_view"
FEATURE_VIEW_VERSION = 1

def get_training_data() -> Tuple[pd.DataFrame, Any, int]:
    try:
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

        # Create or get the Feature View with selected features
        selected_features = fg.select(FEATURE_COLS + [TARGET_COL, "timestamp"])
        fv = fs.get_or_create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=selected_features,
            labels=[TARGET_COL],
            description="Selected features from correlation matrix for AQI prediction"
        )
        logger.info("Feature View '%s' ready!", FEATURE_VIEW_NAME)

        # Create a Training Dataset to establish full provenance
        td_version, _ = fv.create_training_data(
            description="AQI training dataset from correlation-selected features",
            write_options={"wait_for_job": True}
        )
        logger.info("Training Dataset version %d created!", td_version)

        df = fg.read()
        logger.info("Loaded %d rows from Feature Store", len(df))
        return df, project, td_version
    except Exception as e:
        logger.error("Failed to fetch training data: %s", e)
        raise

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=required).copy()
    
    df = df.sort_values("timestamp").reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Random Forest trained!")
    return model

def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting trained!")
    return model

def train_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> Ridge:
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    logger.info("Ridge Regression trained!")
    return model

if __name__ == "__main__":
    try:
        df, project, _ = get_training_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        train_random_forest(X_train, y_train)
        train_gradient_boosting(X_train, y_train)
        train_ridge(X_train, y_train)
        logger.info("All models trained!")
    except Exception as e:
        logger.error("Training failed: %s", e)
