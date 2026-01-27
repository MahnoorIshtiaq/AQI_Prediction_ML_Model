import requests
import pandas as pd
from datetime import datetime
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

def fetch_raw_aqi_data():
    """
    Fetch latest hourly AQI data for Karachi from Open-Meteo
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,us_aqi",
        "timezone": TIMEZONE,
    }

    response = requests.get(AQI_API_URL, params=params)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "pm2_5": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "us_aqi": data["hourly"]["us_aqi"],
    })

    return df
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based, lag-based, and rolling features
    """

    df = df.copy()

    # Rename target
    df["aqi"] = df["us_aqi"]

    # Metadata
    df["city"] = CITY

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Lag features
    for lag in LAGS:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # AQI change rate
    df["aqi_change_rate"] = df["aqi"] - df["aqi"].shift(1)

    # Rolling features
    df["rolling_24h_mean"] = df["aqi"].rolling(24).mean()

    # Drop rows with NaNs caused by lagging
    df = df.dropna().reset_index(drop=True)

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

    collection.insert_many(records)

    print(f"Inserted {len(records)} feature rows into MongoDB")


def run_feature_pipeline():
    raw_df = fetch_raw_aqi_data()
    feature_df = engineer_features(raw_df)
    save_to_mongodb(feature_df)
    return feature_df


if __name__ == "__main__":
    df = run_feature_pipeline()
    print(df.head())
    print(f"Generated {len(df)} feature rows")



