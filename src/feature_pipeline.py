# src\feature_pipeline.py
import requests
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
CITY = "Karachi"

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidity", "wind_speed",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_24h",
]

# --------------------------------------------------
# AQI + pollutant data
# --------------------------------------------------
def fetch_raw_aqi_data() -> pd.DataFrame:
    r = requests.get(
        AQI_API_URL,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": (
                "pm2_5,pm10,"
                "nitrogen_dioxide,ozone,"
                "carbon_monoxide,sulphur_dioxide,"
                "us_aqi"
            ),
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "pm2_5": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "no2": data["hourly"]["nitrogen_dioxide"],
        "o3": data["hourly"]["ozone"],
        "co": data["hourly"]["carbon_monoxide"],
        "so2": data["hourly"]["sulphur_dioxide"],
        "aqi": data["hourly"]["us_aqi"],
        "city": CITY,
    })

# --------------------------------------------------
# Weather data
# --------------------------------------------------
def fetch_weather_data() -> pd.DataFrame:
    r = requests.get(
        WEATHER_API_URL,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "wind_speed_10m"
            ),
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })

# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    df["aqi_change_1h"] = df["aqi"] - df["aqi"].shift(1)
    df["aqi_change_24h"] = df["aqi"] - df["aqi"].shift(24)

    df["target_aqi"] = df["aqi"].shift(-1)

    return df.dropna(
        subset=FEATURE_COLUMNS + ["target_aqi"]
    ).reset_index(drop=True)

# --------------------------------------------------
# MongoDB write (safe)
# --------------------------------------------------
def save_to_mongodb(df: pd.DataFrame) -> None:
    col = MongoClient(
        os.getenv("MONGODB_URI")
    )[os.getenv("MONGODB_DB")][os.getenv("MONGODB_COLLECTION")]

    timestamps = df["timestamp"].tolist()
    col.delete_many({
        "timestamp": {"$in": timestamps},
        "dataset_type": "online",
        "city": CITY,
    })

    records = df.to_dict("records")
    for r in records:
        r["dataset_type"] = "online"
        r["schema_version"] = 2

    if records:
        col.insert_many(records)

    print(f"✅ Inserted {len(records)} online feature rows")

# --------------------------------------------------
# Runner
# --------------------------------------------------
def run_feature_pipeline():
    aqi = fetch_raw_aqi_data()
    weather = fetch_weather_data()

    raw = aqi.merge(weather, on="timestamp", how="left")
    features = engineer_features(raw)

    save_to_mongodb(features)
    return features

if __name__ == "__main__":
    df = run_feature_pipeline()
    print(df.head())
