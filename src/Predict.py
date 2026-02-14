# src/Predict.py

import os
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv
from collections import deque

load_dotenv()

# --------------------------------------------------
# Config
# --------------------------------------------------

LAT, LON = 24.8607, 67.0011
FORECAST_HOURS = 72

# CORRECTED: Added weather features
FEATURE_COLUMNS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidity", "wind_speed",  # ADDED
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_24h",
]

MODEL_NAME = "AQI_NextHour_Model"

# --------------------------------------------------
# Load Production Model
# --------------------------------------------------

def load_model():
    """Load the production model from MLflow"""
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Loaded model: {MODEL_NAME}/Production")
    return model


# --------------------------------------------------
# Load latest feature row
# --------------------------------------------------

def load_latest_features():
    """Load the most recent feature row from MongoDB"""
    col = MongoClient(
        os.getenv("MONGODB_URI")
    )[os.getenv("MONGODB_DB")][os.getenv("MONGODB_COLLECTION")]

    doc = col.find_one(
        {"dataset_type": "online"},
        sort=[("timestamp", -1)]
    )

    if not doc:
        raise ValueError("❌ No online features found in MongoDB")

    doc.pop("_id", None)

    df = pd.DataFrame([doc])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print(f"📊 Loaded latest features from: {df['timestamp'].iloc[0]}")

    return df


# --------------------------------------------------
# Recursive Hourly Forecast - CORRECTED
# --------------------------------------------------

def hourly_forecast():
    """
    Generate 72-hour recursive forecast with proper lag handling
    """

    model = load_model()
    last_row = load_latest_features()

    forecasts = []

    # Initialize state
    last = last_row.iloc[0].to_dict()
    current_time = last["timestamp"]

    # FIXED: Maintain a proper 24-hour lag history
    aqi_history = deque(maxlen=24)
    
    # Populate with current and lagged values
    aqi_history.append(last["aqi_lag_24"])  # 24 hours ago
    for _ in range(22):  # Fill intermediate hours (we don't have them)
        aqi_history.append(last["aqi_lag_24"])  # Use same value
    aqi_history.append(last["aqi_lag_1"])  # 1 hour ago

    print(f"🔮 Generating {FORECAST_HOURS}-hour forecast...")

    for i in range(FORECAST_HOURS):

        # Move time forward
        current_time = current_time + timedelta(hours=1)

        # Update time-based features
        last["hour"] = current_time.hour
        last["day"] = current_time.day
        last["month"] = current_time.month
        last["day_of_week"] = current_time.dayofweek

        # Predict next AQI
        input_df = pd.DataFrame([{k: last[k] for k in FEATURE_COLUMNS}])
        pred_aqi = float(model.predict(input_df)[0])
        
        # Ensure non-negative AQI
        pred_aqi = max(0, pred_aqi)

        # Store prediction
        forecasts.append({
            "timestamp": current_time.isoformat(),
            "predicted_aqi": round(pred_aqi, 2)
        })

        # ----------------------------
        # CORRECTED: Recursive feature updates
        # ----------------------------

        # Update AQI history
        aqi_history.append(pred_aqi)

        # Update lag features CORRECTLY
        last["aqi_lag_1"] = pred_aqi
        last["aqi_lag_24"] = aqi_history[0]  # Correct 24-hour lag

        # Update change rates
        last["aqi_change_1h"] = pred_aqi - aqi_history[-2]  # Compare to 1h ago
        last["aqi_change_24h"] = pred_aqi - aqi_history[0]  # Compare to 24h ago

        # ----------------------------
        # CORRECTED: Realistic pollutant & weather drift
        # ----------------------------
        
        # Pollutant drift (based on AQI change)
        aqi_delta = pred_aqi - aqi_history[-2]
        
        for col in ["pm2_5", "pm10", "no2", "o3", "co", "so2"]:
            # Drift proportional to AQI change with some noise
            drift = aqi_delta * 0.03 + np.random.normal(0, 0.5)
            last[col] = max(0, last[col] + drift)

        # ADDED: Weather feature drift (gradual changes)
        # Temperature: small random walk
        last["temperature"] += np.random.normal(0, 0.3)
        
        # Humidity: bounded random walk
        last["humidity"] += np.random.normal(0, 1.0)
        last["humidity"] = np.clip(last["humidity"], 0, 100)
        
        # Wind speed: small variations
        last["wind_speed"] = max(0, last["wind_speed"] + np.random.normal(0, 0.2))

    print(f"✅ Generated {len(forecasts)} hourly predictions")

    return forecasts


# --------------------------------------------------
# Public function
# --------------------------------------------------

def get_72h_forecast():
    """
    Public API function to get 72-hour forecast
    Returns: List of dicts with timestamp and predicted_aqi
    """
    return hourly_forecast()


# --------------------------------------------------
# Run standalone test
# --------------------------------------------------

if __name__ == "__main__":
    print("Testing forecast generation...\n")
    
    result = get_72h_forecast()
    
    print(f"\nFirst 5 predictions:")
    for pred in result[:5]:
        print(f"   {pred['timestamp']}: AQI = {pred['predicted_aqi']}")
    
    print(f"\nLast 5 predictions:")
    for pred in result[-5:]:
        print(f"   {pred['timestamp']}: AQI = {pred['predicted_aqi']}")
    
    # Statistics
    aqis = [p['predicted_aqi'] for p in result]
    print(f"\nForecast statistics:")
    print(f"   Mean AQI: {np.mean(aqis):.2f}")
    print(f"   Min AQI: {np.min(aqis):.2f}")
    print(f"   Max AQI: {np.max(aqis):.2f}")
    print(f"   Std Dev: {np.std(aqis):.2f}")
