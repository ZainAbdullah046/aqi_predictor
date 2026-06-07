

import os
import sys
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import hopsworks
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from config import get_logger, FEATURE_COLS, AQI_LABELS, HOPSWORKS_PROJECT, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, MODEL_NAME, MODEL_VERSION, CITY

load_dotenv()
logger = get_logger(__name__)

# Global state for model and df
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Initializing Hopsworks connection...")
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        mr = project.get_model_registry()
        model_obj = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
        model_dir = model_obj.download()
        ml_models["model"] = joblib.load(os.path.join(model_dir, "best_model.pkl"))

        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ml_models["df"] = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Model and data loaded successfully.")
    except Exception as e:
        logger.error("Failed to load model/data on startup: %s", e)
        # Depending on how strict we want to be, we could raise here to prevent startup
        # But we'll just log it for now in case hopsworks is temporarily unavailable.
    
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(
    title=f"{CITY.capitalize()} AQI Predictor API",
    description="API for predicting Air Quality Index.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models ---

class HealthCheck(BaseModel):
    status: str
    service: str
    endpoints: List[str]

class DayForecast(BaseModel):
    aqi_value: float
    category: str

class ForecastResponse(BaseModel):
    city: str
    current_aqi: int
    forecast: Dict[str, DayForecast]

class CurrentDataResponse(BaseModel):
    city: str
    timestamp: str
    aqi: int
    category: str
    pm25: float
    pm10: float
    no2: float
    o3: float
    co: float
    so2: float
    nh3: float

class HistoryDataResponse(BaseModel):
    timestamp: str
    aqi: int
    pm25: float
    pm10: float
    no2: float
    o3: float

# --- Endpoints ---

@app.get("/", response_model=HealthCheck)
def home() -> Any:
    return {
        "status": "running",
        "service": f"{CITY.capitalize()} AQI Predictor API",
        "endpoints": ["/predict", "/current", "/history"]
    }

@app.get("/predict", response_model=ForecastResponse)
def predict() -> Any:
    if "model" not in ml_models or "df" not in ml_models:
        raise HTTPException(status_code=503, detail="Model or data not loaded.")
        
    try:
        model = ml_models["model"]
        df = ml_models["df"]
        latest = df.tail(1).copy()

        preds = []
        for day_offset in range(1, 4):
            row = latest[FEATURE_COLS].copy().fillna(0)
            if preds:
                row["rolling_avg_24"] = float(np.mean(preds))
            pred = float(np.clip(model.predict(row)[0], 1, 5))
            preds.append(round(pred, 2))

        forecast = {}
        for i, p in enumerate(preds, 1):
            aqi_int = max(1, min(5, int(round(p))))
            forecast[f"day_{i}"] = {
                "aqi_value": p,
                "category": AQI_LABELS.get(aqi_int, "Unknown")
            }

        return {
            "city": CITY.capitalize(),
            "current_aqi": int(latest["aqi"].values[0]),
            "forecast": forecast
        }
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current", response_model=CurrentDataResponse)
def current() -> Any:
    if "df" not in ml_models:
        raise HTTPException(status_code=503, detail="Data not loaded.")
        
    try:
        df = ml_models["df"]
        row = df.tail(1).iloc[0]
        aqi = int(row["aqi"])
        return {
            "city": CITY.capitalize(),
            "timestamp": str(row["timestamp"]),
            "aqi": aqi,
            "category": AQI_LABELS.get(aqi, "Unknown"),
            "pm25": float(row.get("pm25", 0)),
            "pm10": float(row.get("pm10", 0)),
            "no2": float(row.get("no2", 0)),
            "o3": float(row.get("o3", 0)),
            "co": float(row.get("co", 0)),
            "so2": float(row.get("so2", 0)),
            "nh3": float(row.get("nh3", 0)),
        }
    except Exception as e:
        logger.error("Failed to get current data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=List[HistoryDataResponse])
def history(hours: int = Query(24, description="Number of hours to return history for")) -> Any:
    if "df" not in ml_models:
        raise HTTPException(status_code=503, detail="Data not loaded.")
        
    try:
        df = ml_models["df"]
        rows = df.tail(hours)[["timestamp", "aqi", "pm25", "pm10", "no2", "o3"]].copy()
        rows["timestamp"] = rows["timestamp"].astype(str)
        return rows.to_dict(orient="records")
    except Exception as e:
        logger.error("Failed to get history data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_api:app", host="127.0.0.1", port=5000, reload=True)
