import requests
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
CITY = "Karachi"
TIMEZONE = "Asia/Karachi"

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

LAGS = [1, 3, 6, 12, 24]

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
    "rolling_24h_mean",
]

def fetch_raw_aqi_data():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,us_aqi",
        "timezone": TIMEZONE,
    }

    r = requests.get(AQI_API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "pm2_5": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "us_aqi": data["hourly"]["us_aqi"],
    })

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    df["aqi"] = df["us_aqi"]
    df["city"] = CITY

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Lag features
    for lag in LAGS:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    df["aqi_change_rate"] = df["aqi"] - df["aqi"].shift(1)
    df["rolling_24h_mean"] = df["aqi"].rolling(24).mean()

    # Target
    df["target_aqi"] = df["aqi"].shift(-1)

    # 🔑 Drop NaNs for ALL required columns
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_aqi"]
    ).reset_index(drop=True)

    return df

def save_to_mongodb(df: pd.DataFrame):
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    timestamps = df["timestamp"].tolist()
    collection.delete_many({"timestamp": {"$in": timestamps}})

    records = df.to_dict(orient="records")
    for r in records:
        r["schema_version"] = 1

    if records:
        collection.insert_many(records)

    print(f"Inserted {len(records)} rows into MongoDB")

def run_feature_pipeline():
    raw = fetch_raw_aqi_data()
    features = engineer_features(raw)
    save_to_mongodb(features)
    return features

if __name__ == "__main__":
    df = run_feature_pipeline()
    print(df.head())
    print(f"Generated {len(df)} feature rows")
    print(f"Generated {len(df)} feature rows")
