# feature_pipeline.py
import requests
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

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
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# --------------------------------------------------
# Feature configuration (MUST MATCH TRAINING)
# --------------------------------------------------
FEATURE_COLUMNS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidity", "wind_speed",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_24h",
]

# --------------------------------------------------
# Fetch AQI + pollutant data (includes forecasts)
# --------------------------------------------------
def fetch_raw_aqi_data() -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": (
            "pm2_5,pm10,"
            "nitrogen_dioxide,ozone,"
            "carbon_monoxide,sulphur_dioxide,"
            "us_aqi"
        ),
        "timezone": TIMEZONE,
    }

    r = requests.get(AQI_API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
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
# Fetch weather data
# --------------------------------------------------
def fetch_weather_data() -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),
        "timezone": TIMEZONE,
    }

    r = requests.get(WEATHER_API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })

# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Lag features
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # Change rates
    df["aqi_change_1h"] = df["aqi"] - df["aqi"].shift(1)
    df["aqi_change_24h"] = df["aqi"] - df["aqi"].shift(24)

    # Target
    df["target_aqi"] = df["aqi"].shift(-1)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_aqi"]
    ).reset_index(drop=True)

    return df

# --------------------------------------------------
# Save online features
# --------------------------------------------------
def save_to_mongodb(df: pd.DataFrame) -> None:
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    collection.delete_many({
        "city": CITY,
        "dataset_type": "online",
    })

    records = df.to_dict("records")
    for r in records:
        r["dataset_type"] = "online"
        r["schema_version"] = 2

    if records:
        collection.insert_many(records)

    print(f"Inserted {len(records)} online feature rows")

# --------------------------------------------------
# Runner
# --------------------------------------------------
def run_feature_pipeline() -> pd.DataFrame:
    aqi_df = fetch_raw_aqi_data()
    weather_df = fetch_weather_data()

    raw_df = aqi_df.merge(weather_df, on="timestamp", how="left")
    feature_df = engineer_features(raw_df)

    save_to_mongodb(feature_df)
    return feature_df

if __name__ == "__main__":
    df = run_feature_pipeline()
    print(df.head())
