# src/feature_pipeline.py 
import requests
import pandas as pd
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
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
# AQI + pollutant data (CURRENT/PAST ONLY)
# --------------------------------------------------
def fetch_raw_aqi_data() -> pd.DataFrame:
    """
    Fetch AQI data for the PAST 48 hours to ensure we have enough data
    for lag feature calculation (need t-24 hours for lag_24).
    """
    # Get past 48 hours to have enough for lag features
    end_date = datetime.utcnow().date()
    start_date = (datetime.utcnow() - timedelta(days=2)).date()

    print(f"📊 Fetching AQI data from {start_date} to {end_date}")

    r = requests.get(
        AQI_API_URL,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": (
                "pm2_5,pm10,"
                "nitrogen_dioxide,ozone,"
                "carbon_monoxide,sulphur_dioxide,"
                "us_aqi"
            ),
            "timezone": "UTC",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
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
    
    # CRITICAL: Filter to only past/current data (no future)
    now = pd.Timestamp.utcnow()
    df = df[df['timestamp'] <= now].copy()
    
    print(f"✅ Fetched {len(df)} AQI records (up to {df['timestamp'].max()})")
    
    return df


# --------------------------------------------------
# Weather data (CURRENT ONLY)
# --------------------------------------------------
def fetch_weather_data() -> pd.DataFrame:
    """
    Fetch weather data. The forecast API returns current + future data,
    but we'll filter to only current/past in the main pipeline.
    """
    print(f"🌤 Fetching weather data...")
    
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
            "forecast_days": 3, 
            "past_days": 2,    
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })
    

    now = pd.Timestamp.utcnow()
    df = df[df['timestamp'] <= now].copy()
    
    print(f"✅ Fetched {len(df)} weather records (up to {df['timestamp'].max()})")
    
    return df


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw data.
    Creates lag features and temporal features.
    """
    df = df.sort_values("timestamp").copy()

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Lag features
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # Change features
    df["aqi_change_1h"] = df["aqi"] - df["aqi"].shift(1)
    df["aqi_change_24h"] = df["aqi"] - df["aqi"].shift(24)

    # Target (next hour AQI)
    df["target_aqi"] = df["aqi"].shift(-1)

    print(f"⚙️ Engineered features for {len(df)} rows")

    # Drop rows with missing critical features
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["target_aqi"]).reset_index(drop=True)
    
    print(f"✅ {len(df_clean)} rows after dropping NaNs")
    
    # CRITICAL: Final check - ensure no future data
    now = pd.Timestamp.utcnow()
    df_clean = df_clean[df_clean['timestamp'] <= now].copy()
    
    print(f"✅ Final: {len(df_clean)} rows (all <= {now})")

    return df_clean


# --------------------------------------------------
# MongoDB write (safe)
# --------------------------------------------------
def save_to_mongodb(df: pd.DataFrame) -> None:
    """
    Save features to MongoDB, replacing any existing records
    with the same timestamps.
    """
    if df.empty:
        print("⚠️ No data to save")
        return
    
    col = MongoClient(
        os.getenv("MONGODB_URI")
    )[os.getenv("MONGODB_DB")][os.getenv("MONGODB_COLLECTION")]

    timestamps = df["timestamp"].tolist()
    
    # Delete existing records for these timestamps
    deleted = col.delete_many({
        "timestamp": {"$in": timestamps},
        "dataset_type": "online",
        "city": CITY,
    })
    
    if deleted.deleted_count > 0:
        print(f"🗑️ Deleted {deleted.deleted_count} existing records")

    # Prepare records
    records = df.to_dict("records")
    for r in records:
        r["dataset_type"] = "online"
        r["schema_version"] = 2

    # Insert new records
    col.insert_many(records)

    print(f"✅ Inserted {len(records)} online feature rows")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


# --------------------------------------------------
# Runner
# --------------------------------------------------
def run_feature_pipeline():
    """
    Main pipeline: Fetch data, engineer features, save to MongoDB.
    """
    print("\n" + "="*80)
    print("FEATURE PIPELINE - RUNNING")
    print(f"Current time (UTC): {datetime.utcnow()}")
    print("="*80 + "\n")
    
    # Fetch raw data
    aqi = fetch_raw_aqi_data()
    weather = fetch_weather_data()

    # Merge datasets
    print("\n🔗 Merging AQI and weather data...")
    raw = aqi.merge(weather, on="timestamp", how="inner")  # Use inner join
    print(f"✅ Merged dataset: {len(raw)} rows")

    # Engineer features
    print("\n⚙️ Engineering features...")
    features = engineer_features(raw)

    # Save to MongoDB
    print("\n💾 Saving to MongoDB...")
    save_to_mongodb(features)
    
    print("\n" + "="*80)
    print("FEATURE PIPELINE - COMPLETE")
    print("="*80)
    
    return features


if __name__ == "__main__":
    df = run_feature_pipeline()
    print(f"\n📊 Sample of saved features:")
    print(df.head())
    print(f"\n📊 Latest timestamp: {df['timestamp'].max()}")
