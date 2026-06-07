

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — skipping SHAP plots (pip install shap)")

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    logger.info("%s", model_name)
    logger.info("   RMSE : %.4f", rmse)
    logger.info("   MAE  : %.4f", mae)
    logger.info("   R²   : %.4f", r2)

    return {"model_name": model_name, "rmse": rmse, "mae": mae, "r2": r2}

def pick_best_model(models_dict: Dict[str, Any], results: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
    best_info = min(results, key=lambda x: x["rmse"])
    best_name = best_info["model_name"]
    logger.info("Best Model: %s  (RMSE: %.4f, R²: %.4f)", best_name, best_info['rmse'], best_info['r2'])
    return models_dict[best_name], best_info

def generate_shap_plot(model: Any, X_test: pd.DataFrame, model_name: str, save_path: str = "shap_plot.png") -> None:
    if not SHAP_AVAILABLE:
        logger.warning("Skipping SHAP — library not installed")
        return

    logger.info("Generating SHAP values for %s ...", model_name)
    try:
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            sample = X_test.sample(min(100, len(X_test)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, sample)
            shap_values = explainer.shap_values(sample)
            X_test = sample

        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        logger.info("SHAP plot saved -> %s", save_path)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)

if __name__ == "__main__":
    from train_models import get_training_data, prepare_data
    from train_models import train_random_forest, train_gradient_boosting, train_ridge

    try:
        df, project = get_training_data()
        X_train, X_test, y_train, y_test = prepare_data(df)

        rf = train_random_forest(X_train, y_train)
        gb = train_gradient_boosting(X_train, y_train)
        rdg = train_ridge(X_train, y_train)

        results = []
        results.append(evaluate_model(rf, X_test, y_test, "RandomForest"))
        results.append(evaluate_model(gb, X_test, y_test, "GradientBoosting"))
        results.append(evaluate_model(rdg, X_test, y_test, "Ridge"))

        best_model, best_info = pick_best_model(
            {"RandomForest": rf, "GradientBoosting": gb, "Ridge": rdg},
            results
        )
        generate_shap_plot(best_model, X_test, best_info["model_name"])
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
