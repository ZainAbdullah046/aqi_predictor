

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

from config import get_logger, FEATURE_COLS, TARGET_COL, HOPSWORKS_PROJECT

load_dotenv()
logger = get_logger(__name__)

def get_training_data() -> Tuple[pd.DataFrame, Any]:
    try:
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features_delhi", version=1)
        df = fg.read()
        logger.info("Loaded %d rows from Feature Store", len(df))
        return df, project
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
        df, project = get_training_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        train_random_forest(X_train, y_train)
        train_gradient_boosting(X_train, y_train)
        train_ridge(X_train, y_train)
        logger.info("All models trained!")
    except Exception as e:
        logger.error("Training failed: %s", e)
