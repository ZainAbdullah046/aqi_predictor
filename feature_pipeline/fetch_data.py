"""
fetch_data.py
-------------
Fetches Air Pollution data for Delhi from OpenWeatherMap API.

Two modes:
  1. fetch_current()
  2. fetch_historical(days)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from config import get_logger, LAT, LON, CITY, OPENWEATHER_BASE_URL

load_dotenv()

logger = get_logger(__name__)
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def _parse_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    comp = entry.get("components", {})
    dt = entry.get("dt")
    aqi = entry.get("main", {}).get("aqi", 0)

    return {
        "city": CITY,
        "timestamp": datetime.utcfromtimestamp(dt).strftime("%Y-%m-%d %H:%M:%S") if dt else "",
        "unix_time": dt,
        "aqi": aqi,
        "co": comp.get("co", 0.0),
        "no": comp.get("no", 0.0),
        "no2": comp.get("no2", 0.0),
        "o3": comp.get("o3", 0.0),
        "so2": comp.get("so2", 0.0),
        "pm25": comp.get("pm2_5", 0.0),
        "pm10": comp.get("pm10", 0.0),
        "nh3": comp.get("nh3", 0.0),
    }

def fetch_current() -> pd.DataFrame:
    try:
        url = f"{OPENWEATHER_BASE_URL}?lat={LAT}&lon={LON}&appid={API_KEY}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        entries = data.get("list", [])
        if not entries:
            raise ValueError("No data returned from OpenWeather current API")

        row = _parse_entry(entries[0])
        logger.info("Fetched current AQI for %s: %s", CITY, row['aqi'])
        return pd.DataFrame([row])
    except requests.RequestException as e:
        logger.error("Failed to fetch current data: %s", e)
        raise

def fetch_historical(days: int = 365) -> pd.DataFrame:
    try:
        end_dt        = datetime.utcnow()
        start_dt      = end_dt - timedelta(days=days)
        all_rows      = []
        chunk_days    = 7
        current_start = start_dt
        total_chunks  = (days // chunk_days) + 1
        chunk_num     = 0

        logger.info("Fetching %d days in %d chunks of 7 days each...", days, total_chunks)

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)

            start_unix = int(current_start.timestamp())
            end_unix   = int(current_end.timestamp())

            url = (
                f"{OPENWEATHER_BASE_URL}/history"
                f"?lat={LAT}&lon={LON}"
                f"&start={start_unix}&end={end_unix}"
                f"&appid={API_KEY}"
            )

            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                entries = resp.json().get("list", [])
                rows    = [_parse_entry(e) for e in entries]
                all_rows.extend(rows)
                chunk_num += 1
                logger.info(
                    "Chunk %d/%d done → %d records (total: %d)",
                    chunk_num, total_chunks, len(rows), len(all_rows)
                )
            except Exception as e:
                logger.warning("Chunk %d failed: %s — skipping", chunk_num, e)

            current_start = current_end

        if not all_rows:
            raise ValueError("No historical data returned from OpenWeather API")

        df = pd.DataFrame(all_rows)
        df = (df.drop_duplicates(subset=["unix_time"])
                .sort_values("timestamp")
                .reset_index(drop=True))

        logger.info("Total: %d hourly records fetched", len(df))
        return df

    except requests.RequestException as e:
        logger.error("Failed to fetch historical data: %s", e)
        raise

if __name__ == "__main__":
    try:
        df_cur = fetch_current()
        logger.info("Current Reading DataFrame shape: %s", df_cur.shape)

        df_hist = fetch_historical(days=7)
        logger.info("Historical Reading DataFrame shape: %s", df_hist.shape)
    except Exception as e:
        logger.error("Error during execution: %s", e)
