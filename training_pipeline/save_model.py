

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
from config import get_logger, MODEL_NAME, FEATURE_COLS, TARGET_COL, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from training_pipeline.train_models import FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION

load_dotenv()
logger = get_logger(__name__)

def save_best_model(model: Any, metrics: Dict[str, float], model_algo_name: str, project: Any, td_version: int = None) -> None:
    try:
        mr = project.get_model_registry()
        fs = project.get_feature_store()

        # Get the Feature View to link to the model
        fv = fs.get_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION
        )

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
            feature_view=fv,
            training_dataset_version=td_version,
            description=f"Best AQI model for Delhi: {model_algo_name}"
        )

        import shutil
        os.makedirs("model_dir", exist_ok=True)
        shutil.copy(local_path, "model_dir/")
        if os.path.exists("shap_plot.png"):
            shutil.copy("shap_plot.png", "model_dir/")
        
        aqi_model.save("model_dir")
        
        # Clean up the temporary folder after successful upload
        shutil.rmtree("model_dir")
        
        logger.info("Model and SHAP plot uploaded to Hopsworks Registry!")
        logger.info("Name : %s", MODEL_NAME)
        logger.info("Type : %s", model_algo_name)
        logger.info("RMSE : %.4f", metrics['rmse'])
        logger.info("MAE  : %.4f", metrics['mae'])
        logger.info("R²   : %.4f", metrics['r2'])
    except Exception as e:
        logger.error("Failed to save best model: %s", e)
        raise

if __name__ == "__main__":
    logger.info("Daily Training Pipeline")

    try:
        logger.info("Loading data from Feature Store ...")
        df, project, td_version = get_training_data()

        logger.info("Preparing train/test split ...")
        X_train, X_test, y_train, y_test = prepare_data(df)

        logger.info("Training models ...")
        rf = train_random_forest(X_train, y_train)
        gb = train_gradient_boosting(X_train, y_train)
        rdg = train_ridge(X_train, y_train)

        logger.info("Evaluating models ...")
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

        logger.info("Saving best model to Hopsworks ...")
        save_best_model(best_model, best_info, best_info["model_name"], project, td_version)

        logger.info("Daily training pipeline complete!")
    except Exception as e:
        logger.error("Training pipeline failed: %s", e)
