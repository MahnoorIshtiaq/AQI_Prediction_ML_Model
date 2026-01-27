import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

from feature_pipeline import engineer_features

load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
CITY = "Karachi"
TIMEZONE = "Asia/Karachi"

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

HISTORY_DAYS = 90


def fetch_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
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


def save_training_data(df: pd.DataFrame):
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db["training_features"]

    collection.delete_many({"city": CITY})

    records = df.to_dict(orient="records")
    for r in records:
        r["schema_version"] = 1
        r["dataset_type"] = "training"

    collection.insert_many(records)

    print(f"Inserted {len(records)} training rows")


def run_backfill():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    print(f"Backfilling from {start_date} to {end_date}")

    raw_df = fetch_historical_data(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )

    feature_df = engineer_features(raw_df)
    save_training_data(feature_df)

    return feature_df


if __name__ == "__main__":
    df = run_backfill()
    print(df.head())
