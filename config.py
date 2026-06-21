import logging
import sys

# Location Settings
LAT = 28.6139
LON = 77.2090
CITY = "delhi"

# OpenWeather API Settings
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Hopsworks Settings
HOPSWORKS_PROJECT = "aqi_10pearlproject"
FEATURE_GROUP_NAME = "aqi_features_delhi"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_predictor_delhi"
MODEL_VERSION = 15

# Features for training, SHAP, dashboard, and API by seeing the  correlation matrix
FEATURE_COLS = [
    "co", "no2", "so2", "pm25", "pm10",
    "pm25_pm10_ratio", "rolling_avg_24"
]
TARGET_COL = "aqi"

# AQI Standards
AQI_LABELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

# Logging configuration
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
