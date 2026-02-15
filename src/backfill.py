# src/backfill.py 
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

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

HISTORY_DAYS = 90  # 90 days for training data


def fetch_historical_data(start_date, end_date):
    """
    Fetch historical AQI and pollutant data.
    CRITICAL: Only fetches PAST data, never future.
    """
    print(f"📊 Fetching AQI data from {start_date} to {end_date}")
    
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
        "timezone": "UTC",  # FIXED: Use UTC consistently
    }

    r = requests.get(AQI_API_URL, params=params, timeout=60)
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
    
    # CRITICAL: Ensure no future data
    now = pd.Timestamp.utcnow()
    original_count = len(df)
    df = df[df['timestamp'] <= now].copy()
    filtered_count = original_count - len(df)
    
    if filtered_count > 0:
        print(f"⚠️ Filtered out {filtered_count} future records from AQI data")
    
    print(f"✅ Fetched {len(df)} AQI records (up to {df['timestamp'].max()})")
    
    return df


def fetch_weather(start_date, end_date):
    """
    Fetch historical weather data using the ARCHIVE API.
    CRITICAL: Archive API only returns historical data, never future.
    """
    print(f"🌤 Fetching weather data from {start_date} to {end_date}")
    
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

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })
    

    now = pd.Timestamp.utcnow()
    original_count = len(df)
    df = df[df['timestamp'] <= now].copy()
    filtered_count = original_count - len(df)
    
    if filtered_count > 0:
        print(f"⚠️ Filtered out {filtered_count} future records from weather data")
    
    print(f"✅ Fetched {len(df)} weather records (up to {df['timestamp'].max()})")
    
    return df


def save_training_data(df):
    """
    Save historical data to MongoDB.
    Uses same collection as feature_pipeline but marks as 'historical'.
    """
    if df.empty:
        print("⚠️ No data to save (empty dataframe)")
        return
    

    now = pd.Timestamp.utcnow()
    future_records = df[df['timestamp'] > now]
    
    if len(future_records) > 0:
        print(f"ERROR: Found {len(future_records)} future records!")
        print(f"   First future timestamp: {future_records['timestamp'].min()}")
        print(f"   Last future timestamp: {future_records['timestamp'].max()}")
        print(f"   Current time: {now}")
        raise ValueError("Cannot save future data to training set!")
    
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    

    col = db[os.getenv("MONGODB_COLLECTION")]  # aqi_features_v1

    print(f"🗑️ Deleting old historical data for {CITY}...")
    deleted = col.delete_many({
        "city": CITY,
        "dataset_type": "historical"
    })
    
    if deleted.deleted_count > 0:
        print(f"   Deleted {deleted.deleted_count} old records")


    records = df.to_dict("records")
    for r in records:
        r["dataset_type"] = "historical"  
        r["schema_version"] = 2


    col.insert_many(records)

    print(f"✅ Inserted {len(records)} historical training rows to {os.getenv('MONGODB_COLLECTION')}")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def run_backfill():
    """
    Main backfill function.
    Fetches HISTORY_DAYS of historical data and engineers features.
    """
    print("\n" + "="*80)
    print("BACKFILL PIPELINE - RUNNING")
    print(f"Current time (UTC): {datetime.utcnow()}")
    print("="*80 + "\n")
    
    # Calculate date range
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    print(f"📅 Backfilling from {start_date} to {end_date} ({HISTORY_DAYS} days)")
    print(f"⚠️ Note: Only data up to current time will be kept\n")

    # Fetch AQI data
    aqi = fetch_historical_data(start_date.isoformat(), end_date.isoformat())
    
    if aqi.empty:
        print("❌ No AQI data fetched. Aborting.")
        return None
    
    # Fetch weather data
    weather = fetch_weather(start_date.isoformat(), end_date.isoformat())
    
    if weather.empty:
        print("❌ No weather data fetched. Aborting.")
        return None

    # Merge datasets
    print("\n🔗 Merging AQI and weather data...")
    raw = aqi.merge(weather, on="timestamp", how="inner")  # Use inner join
    print(f"✅ Merged dataset: {len(raw)} rows")
    
    if raw.empty:
        print("❌ No overlapping timestamps between AQI and weather. Aborting.")
        return None

    # Engineer features
    print("\n⚙️ Engineering features...")
    features = engineer_features(raw)
    
    if features.empty:
        print("⚠️ No features after engineering (all dropped due to NaN lag features)")
        print("   This is expected if you have less than 25 hours of data")
        return None


    now = pd.Timestamp.utcnow()
    features_clean = features[features['timestamp'] <= now].copy()
    
    if len(features_clean) < len(features):
        filtered = len(features) - len(features_clean)
        print(f"⚠️ Filtered {filtered} future records before saving")
    
    # Save to MongoDB
    print("\n💾 Saving to MongoDB...")
    save_training_data(features_clean)
    
    # Summary statistics
    print("\n" + "="*80)
    print("BACKFILL PIPELINE - COMPLETE")
    print("="*80)
    print(f"\n📊 Final Statistics:")
    print(f"   Total records: {len(features_clean):,}")
    print(f"   Date range: {features_clean['timestamp'].min()} to {features_clean['timestamp'].max()}")
    print(f"   Days covered: {(features_clean['timestamp'].max() - features_clean['timestamp'].min()).days}")
    print(f"   Features per record: {len(features_clean.columns)}")
    print(f"   Missing values: {features_clean.isnull().sum().sum()}")
    
    # Validate no future data
    print(f"\n✅ VALIDATION:")
    print(f"   Current time: {now}")
    print(f"   Latest record: {features_clean['timestamp'].max()}")
    print(f"   All records in past: {(features_clean['timestamp'] <= now).all()}")
    
    return features_clean


if __name__ == "__main__":
    df = run_backfill()
    
    if df is not None:
        print(f"\n🔍 Sample of first 5 rows:")
        print(df.head())
        
        print(f"\n🔍 Sample of last 5 rows:")
        print(df.tail())
        
        print(f"\n📊 Feature columns:")
        print(df.columns.tolist())
        
        print(f"\n📊 Data types:")
        print(df.dtypes)
    else:
        print("\n❌ Backfill failed - no data saved")
