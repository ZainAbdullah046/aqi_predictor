"""
save_model.py
-------------
Full training pipeline entry point (triggered by GitHub Actions daily).

Steps:
  1. Load data from Hopsworks Feature Store
  2. Train all models
  3. Evaluate and pick best
  4. Generate SHAP plot
  5. Save best model to Hopsworks Model Registry
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hopsworks
import joblib
from typing import Dict, Any
from dotenv import load_dotenv

from training_pipeline.train_models import (
    get_training_data, prepare_data,
    train_random_forest, train_gradient_boosting, train_ridge
)
from training_pipeline.evaluate_models import (
    evaluate_model, pick_best_model, generate_shap_plot
)
from config import get_logger, MODEL_NAME

load_dotenv()
logger = get_logger(__name__)

def save_best_model(model: Any, metrics: Dict[str, float], model_algo_name: str, project: Any) -> None:
    """Upload the best model + SHAP plot to Hopsworks Model Registry."""
    try:
        mr = project.get_model_registry()

        local_path = "best_model.pkl"
        joblib.dump(model, local_path)
        logger.info("Model saved locally -> %s", local_path)

        aqi_model = mr.python.create_model(
            name=MODEL_NAME,
            metrics={
                "rmse": round(metrics["rmse"], 4),
                "mae": round(metrics["mae"], 4),
                "r2": round(metrics["r2"], 4),
            },
            description=f"Best AQI model for Delhi: {model_algo_name}"
        )

        files_to_save = [local_path]
        if os.path.exists("shap_plot.png"):
            files_to_save.append("shap_plot.png")

        for file_path in files_to_save:
            aqi_model.save(file_path)
        logger.info("Model uploaded to Hopsworks Registry!")
        logger.info("Name : %s", MODEL_NAME)
        logger.info("Type : %s", model_algo_name)
        logger.info("RMSE : %.4f", metrics['rmse'])
        logger.info("MAE  : %.4f", metrics['mae'])
        logger.info("R²   : %.4f", metrics['r2'])
    except Exception as e:
        logger.error("Failed to save best model: %s", e)
        raise

if __name__ == "__main__":
    logger.info("═══════════════════════════════════════")
    logger.info("      Daily Training Pipeline          ")
    logger.info("═══════════════════════════════════════")

    try:
        logger.info("[1/5] Loading data from Feature Store ...")
        df, project = get_training_data()

        logger.info("[2/5] Preparing train/test split ...")
        X_train, X_test, y_train, y_test = prepare_data(df)

        logger.info("[3/5] Training models ...")
        rf = train_random_forest(X_train, y_train)
        gb = train_gradient_boosting(X_train, y_train)
        rdg = train_ridge(X_train, y_train)

        logger.info("[4/5] Evaluating models ...")
        results = [
            evaluate_model(rf, X_test, y_test, "RandomForest"),
            evaluate_model(gb, X_test, y_test, "GradientBoosting"),
            evaluate_model(rdg, X_test, y_test, "Ridge"),
        ]
        best_model, best_info = pick_best_model(
            {"RandomForest": rf, "GradientBoosting": gb, "Ridge": rdg},
            results
        )

        generate_shap_plot(best_model, X_test, best_info["model_name"])

        logger.info("[5/5] Saving best model to Hopsworks ...")
        save_best_model(best_model, best_info, best_info["model_name"], project)

        logger.info("Daily training pipeline complete!")
    except Exception as e:
        logger.error("Training pipeline failed: %s", e)
