# src/Predict.py

import os
from datetime import timedelta
import pandas as pd
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Config
# --------------------------------------------------
CITY = "Karachi"
FORECAST_HOURS = 72
MODEL_PATH = "models/aqi_model.pkl"

FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "no2",
    "o3",
    "co",
    "so2",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_change_1h",
    "aqi_change_24h",
]

# --------------------------------------------------
# Load latest feature rows from Feature Store
# --------------------------------------------------
def load_latest_features() -> pd.DataFrame:
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db["training_features"]

    # Need at least 24h history for lags
    df = pd.DataFrame(
        list(
            collection.find(
                {"dataset_type": "training", "city": CITY}
            ).sort("timestamp", -1).limit(48)
        )
    )

    if df.empty:
        raise ValueError("❌ No feature data found in MongoDB")

    df = df.drop(columns=["_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Recursive AQI forecast (next 72 hours)
# --------------------------------------------------
def hourly_forecast() -> list:
    history = load_latest_features()
    model = load_model()

    current = history.copy()
    forecasts = []

    for step in range(FORECAST_HOURS):
        last = current.iloc[-1]
        next_time = last["timestamp"] + timedelta(hours=1)

        X = last[FEATURE_COLUMNS].to_frame().T
        pred_aqi = float(model.predict(X)[0])

        # Safety clamp (prevents explosion)
        pred_aqi = max(last["aqi"] - 30, min(last["aqi"] + 30, pred_aqi))

        # Build next row
        new = last.copy()
        new["timestamp"] = next_time
        new["aqi"] = pred_aqi

        # Update lag features
        new["aqi_lag_24"] = current.iloc[-24]["aqi"]
        new["aqi_lag_1"] = last["aqi"]

        # Update change features
        new["aqi_change_1h"] = pred_aqi - last["aqi"]
        new["aqi_change_24h"] = pred_aqi - current.iloc[-24]["aqi"]

        # Update time features
        new["hour"] = next_time.hour
        new["day"] = next_time.day
        new["month"] = next_time.month
        new["day_of_week"] = next_time.dayofweek

        current = pd.concat([current, pd.DataFrame([new])], ignore_index=True)

        forecasts.append({
            "timestamp": next_time.isoformat(),
            "aqi": round(pred_aqi),
        })

    return forecasts


# --------------------------------------------------
# Public API helper
# --------------------------------------------------
def get_72h_forecast() -> dict:
    forecast = hourly_forecast()
    max_aqi = max(f["aqi"] for f in forecast)

    return {
        "city": CITY,
        "hours": 72,
        "max_aqi": max_aqi,
        "risk_level": (
            "Hazardous" if max_aqi > 200 else
            "Unhealthy" if max_aqi > 150 else
            "Moderate"
        ),
        "forecast": forecast,
    }


# --------------------------------------------------
# Local test
# --------------------------------------------------
if __name__ == "__main__":
    result = get_72h_forecast()
    print(f"Max AQI next 72h: {result['max_aqi']}")
    print("First 5 predictions:")
    for row in result["forecast"][:5]:
        print(row)
