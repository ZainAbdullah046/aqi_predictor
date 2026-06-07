"""
dashboard/app.py
----------------
Streamlit dashboard for Delhi AQI predictions.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from typing import Tuple, List, Any
from dotenv import load_dotenv

from config import get_logger, FEATURE_COLS, AQI_LABELS, HOPSWORKS_PROJECT, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, MODEL_NAME, MODEL_VERSION, CITY

load_dotenv()
logger = get_logger(__name__)

st.set_page_config(
    page_title=f"{CITY.capitalize()} AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

AQI_COLORS = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🟣"}
AQI_HEALTH = {
    1: "Air quality is satisfactory. No health risk.",
    2: "Air quality is acceptable. Sensitive individuals should be cautious.",
    3: "Sensitive groups may experience health effects.",
    4: "Everyone may experience health effects.",
    5: "Health alert! Avoid outdoor activities."
}

@st.cache_resource(show_spinner="Loading model from Hopsworks ...")
def load_model_and_data() -> Tuple[Any, pd.DataFrame]:
    try:
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        mr = project.get_model_registry()
        model_obj = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
        model_dir = model_obj.download()
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))

        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return model, df
    except Exception as e:
        logger.error("Error loading model and data: %s", e)
        st.error(f"Failed to load data: {e}")
        st.stop()

def predict_next_3_days(model: Any, df: pd.DataFrame) -> List[float]:
    latest = df.tail(1).copy()
    preds = []
    for day_offset in range(1, 4):
        row = latest[FEATURE_COLS].copy().fillna(0)
        if preds:
            row["rolling_avg_24"] = float(np.mean(preds))
        pred = float(np.clip(model.predict(row)[0], 1, 5))
        preds.append(round(pred, 2))
    return preds

st.title(f"🌫️ {CITY.capitalize()} AQI Predictor")
st.caption("Powered by OpenWeatherMap · Hopsworks Feature Store · Machine Learning")

try:
    model, df = load_model_and_data()

    latest = df.tail(1).iloc[0]
    current_aqi = int(round(latest["aqi"]))
    current_aqi = max(1, min(5, current_aqi))

    if current_aqi >= 4:
        st.error(f"⚠️ **POOR AIR QUALITY ALERT** — {AQI_HEALTH.get(current_aqi, '')}")
    elif current_aqi == 3:
        st.warning(f"⚠️ {AQI_HEALTH.get(current_aqi, '')}")
    else:
        st.success(f"✅ {AQI_HEALTH.get(current_aqi, '')}")

    st.subheader(f"📍 Current Reading — {CITY.capitalize()}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AQI Level", f"{current_aqi}  {AQI_COLORS.get(current_aqi, '')}")
    col2.metric("Category", AQI_LABELS.get(current_aqi, 'Unknown'))
    col3.metric("PM2.5", f"{latest.get('pm25', 0):.1f} µg/m³")
    col4.metric("PM10", f"{latest.get('pm10', 0):.1f} µg/m³")

    st.caption(f"Last updated: {latest['timestamp']}")
    st.divider()

    st.subheader("📅 3-Day AQI Forecast")
    forecasts = predict_next_3_days(model, df)

    day_labels = ["Tomorrow", "Day 2", "Day 3"]
    fcols = st.columns(3)
    for i, col in enumerate(fcols):
        pred_aqi = int(round(forecasts[i]))
        pred_aqi = max(1, min(5, pred_aqi))
        col.metric(
            day_labels[i],
            f"{forecasts[i]}  {AQI_COLORS.get(pred_aqi, '⚪')}",
            delta=f"{AQI_LABELS.get(pred_aqi, 'Unknown')}"
        )

    st.divider()

    st.subheader("📈 24-Hour AQI Trend")
    trend_df = df.tail(24)[["timestamp", "aqi"]].copy()
    trend_df["timestamp"] = trend_df["timestamp"].astype(str)
    st.line_chart(trend_df.set_index("timestamp")["aqi"])

    st.divider()

    st.subheader("🧪 Current Pollutant Levels")
    pollutants = {
        "CO (µg/m³)": latest.get("co", 0),
        "NO (µg/m³)": latest.get("no", 0),
        "NO₂ (µg/m³)": latest.get("no2", 0),
        "O₃ (µg/m³)": latest.get("o3", 0),
        "SO₂ (µg/m³)": latest.get("so2", 0),
        "PM2.5 (µg/m³)": latest.get("pm25", 0),
        "PM10 (µg/m³)": latest.get("pm10", 0),
        "NH₃ (µg/m³)": latest.get("nh3", 0),
    }
    pol_df = pd.DataFrame(
        {"Pollutant": list(pollutants.keys()), "Value": list(pollutants.values())}
    )
    st.bar_chart(pol_df.set_index("Pollutant"))

    st.divider()

    st.subheader("🧠 Model Feature Importance (SHAP)")
    st.caption("This chart explains which factors most influenced the model's recent predictions.")
    try:
        sample_data = df.tail(100)[FEATURE_COLS].fillna(0)
        
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_data)
        else:
            explainer = shap.KernelExplainer(model.predict, sample_data)
            shap_values = explainer.shap_values(sample_data)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_data, show=False)
        st.pyplot(fig)
    except Exception as e:
        logger.warning("Failed to render SHAP plot in Streamlit: %s", e)
        st.info("SHAP feature importance plot could not be generated.")

    st.divider()

    with st.expander("🔍 View Recent Data"):
        st.dataframe(df.tail(24)[[
            "timestamp", "aqi", "pm25", "pm10", "no2", "o3", "co", "so2", "nh3"
        ]], use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Make sure you have:\n1. Run `backfill.py`\n2. Run `save_model.py`\n3. Set `.env` credentials correctly")
