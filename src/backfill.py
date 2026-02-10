import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

from feature_pipeline import engineer_features

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Location & API config
# --------------------------------------------------
LATITUDE = 24.8607
LONGITUDE = 67.0011
CITY = "Karachi"
TIMEZONE = "Asia/Karachi"

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Number of past days to backfill
HISTORY_DAYS = 90

# --------------------------------------------------
# MongoDB config
# --------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = "training_features"

# --------------------------------------------------
# Fetch historical AQI + pollutant data
# --------------------------------------------------
def fetch_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone",
            "carbon_monoxide",
            "sulphur_dioxide",
            "us_aqi",
        ],
        "timezone": TIMEZONE,
    }

    response = requests.get(AQI_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "pm2_5": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "no2": data["hourly"]["nitrogen_dioxide"],
        "o3": data["hourly"]["ozone"],
        "co": data["hourly"]["carbon_monoxide"],
        "so2": data["hourly"]["sulphur_dioxide"],
        "aqi": data["hourly"]["us_aqi"],
    })

    df["city"] = CITY
    return df


# --------------------------------------------------
# Save engineered features to Feature Store
# --------------------------------------------------
def save_training_data(df: pd.DataFrame) -> None:
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]

    # Remove old training data for this city
    collection.delete_many({
        "city": CITY,
        "dataset_type": "training"
    })

    records = df.to_dict(orient="records")

    for r in records:
        r["schema_version"] = 1
        r["dataset_type"] = "training"

    if records:
        collection.insert_many(records)

    print(f"✅ Inserted {len(records)} training rows into MongoDB")


# --------------------------------------------------
# Backfill runner
# --------------------------------------------------
def run_backfill() -> pd.DataFrame:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    print(f"📦 Backfilling AQI data from {start_date} to {end_date}")

    # 1. Fetch raw historical data
    raw_df = fetch_historical_data(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    # 2. Engineer features + target
    feature_df = engineer_features(raw_df)

    # 3. Save to Feature Store
    save_training_data(feature_df)

    return feature_df


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    df = run_backfill()
    print("Sample rows:")
    print(df.head())
