

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from config import get_logger, AQI_LABELS

logger = get_logger(__name__)

def aqi_category(aqi: int) -> str:
    return AQI_LABELS.get(int(aqi), "Unknown")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Parse & sort timestamp ─────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── 2. Time-based features (assignment required) ──────────────────
    df["hour"]  = df["timestamp"].dt.hour        # 0-23
    df["day"]   = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month       # 1-12

    # ── 3. Fill pollutants (all 8 from API) ───────────────────────────
    pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm25", "pm10", "nh3"]
    df[pollutant_cols] = df[pollutant_cols].fillna(0.0)

    # ── 4. AQI change rate (assignment required derived feature) ──────
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

    # ── 5. Rolling average 24h (daily trend) ─────────────────────────
    df["rolling_avg_24"] = df["aqi"].rolling(24, min_periods=1).mean()

    # ── 6. PM2.5 / PM10 ratio (combustion vs dust source) ────────────
    df["pm25_pm10_ratio"] = df.apply(
        lambda r: round(r["pm25"] / r["pm10"], 4) if r["pm10"] > 0 else 0.0,
        axis=1
    )

    # ── 7. AQI category label (display only, not used in training) ────
    df["aqi_category"] = df["aqi"].apply(aqi_category)

    # ── 8. Convert timestamp back to string for Hopsworks ────────────
    df["timestamp"] = df["timestamp"].astype(str)

    logger.info(
        "Features engineered! Shape: %s | Columns: %s",
        df.shape, list(df.columns)
    )
    return df


if __name__ == "__main__":
    from fetch_data import fetch_historical
    try:
        df_raw = fetch_historical(days=7)
        df_eng = engineer_features(df_raw)
        logger.info(
            "Engineered data sample:\n%s",
            df_eng[[
                "timestamp", "aqi", "pm25", "pm10",
                "hour", "day", "month",
                "aqi_change_rate", "rolling_avg_24",
                "pm25_pm10_ratio", "aqi_category"
            ]].tail(10).to_string(index=False)
        )
    except Exception as e:
        logger.error("Error during execution: %s", e)
