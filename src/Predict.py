import os
import joblib
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Config
# -----------------------
MODEL_PATH = "models/best_aqi_model.pkl"
FORECAST_HOURS = 72

FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_change_rate",
    "rolling_24h_mean"
]

# -----------------------
# Load latest features
# -----------------------
def load_latest_features():
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    df = pd.DataFrame(
        list(collection.find().sort("timestamp", -1).limit(24))
    ).sort_values("timestamp")

    return df.drop(columns=["_id"]).reset_index(drop=True)

# -----------------------
# Recursive forecast
# -----------------------
def recursive_forecast(model, history_df):
    predictions = []
    current_df = history_df.copy()

    for _ in range(FORECAST_HOURS):
        last_row = current_df.iloc[-1]
        last_time = last_row["timestamp"]
        next_time = last_time + timedelta(hours=1)

        X = last_row[FEATURE_COLUMNS].to_frame().T
        pred_aqi = float(model.predict(X)[0])

        # Create next row
        new_row = last_row.copy()
        new_row["timestamp"] = next_time
        new_row["aqi"] = pred_aqi

        # Shift lag features
        new_row["aqi_lag_24"] = last_row["aqi_lag_23"] if "aqi_lag_23" in last_row else last_row["aqi_lag_24"]
        new_row["aqi_lag_12"] = last_row["aqi_lag_11"] if "aqi_lag_11" in last_row else last_row["aqi_lag_12"]
        new_row["aqi_lag_6"] = last_row["aqi_lag_5"] if "aqi_lag_5" in last_row else last_row["aqi_lag_6"]
        new_row["aqi_lag_3"] = last_row["aqi_lag_2"] if "aqi_lag_2" in last_row else last_row["aqi_lag_3"]
        new_row["aqi_lag_1"] = last_row["aqi"]

        # Derived features
        new_row["aqi_change_rate"] = pred_aqi - last_row["aqi"]
        new_row["rolling_24h_mean"] = (
            current_df["aqi"].tail(23).sum() + pred_aqi
        ) / 24

        # Time features
        new_row["hour"] = next_time.hour
        new_row["day"] = next_time.day
        new_row["month"] = next_time.month
        new_row["day_of_week"] = next_time.dayofweek

        predictions.append({
            "timestamp": next_time,
            "predicted_aqi": round(pred_aqi, 2)
        })

        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(predictions)

# -----------------------
# API hook
# -----------------------
def get_72h_forecast():
    model = joblib.load(MODEL_PATH)
    history_df = load_latest_features()
    return recursive_forecast(model, history_df)
