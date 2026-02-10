# backfill.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

from feature_pipeline import engineer_features

load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
CITY = "Karachi"
TIMEZONE = "Asia/Karachi"

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

HISTORY_DAYS = 90

def fetch_historical_data(start_date, end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm2_5", "pm10",
            "nitrogen_dioxide", "ozone",
            "carbon_monoxide", "sulphur_dioxide",
            "us_aqi",
        ],
        "timezone": TIMEZONE,
    }

    r = requests.get(AQI_API_URL, params=params, timeout=60)
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



def fetch_weather(start_date, end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ],
        "timezone": "UTC",
    }

    r = requests.get(WEATHER_API_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })


def save_training_data(df):
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    col = db["training_features"]

    col.delete_many({
        "city": CITY,
        "dataset_type": "training"
    })

    records = df.to_dict("records")
    for r in records:
        r["dataset_type"] = "training"
        r["schema_version"] = 2

    if records:
        col.insert_many(records)

    print(f"Inserted {len(records)} training rows")

def run_backfill():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    aqi = fetch_historical_data(start_date.isoformat(), end_date.isoformat())
    weather = fetch_weather(start_date.isoformat(), end_date.isoformat())

    raw = aqi.merge(weather, on="timestamp", how="left")
    features = engineer_features(raw)

    save_training_data(features)
    return features

if __name__ == "__main__":
    df = run_backfill()
    print(df.head())
