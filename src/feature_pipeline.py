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

# --------------------------------------------------
# Feature configuration (MATCHES TRAINING DATA)
# --------------------------------------------------
LAGS = [1, 24]

FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "no2",
    "o3",
    "co",
    "so2",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_change_1h",
    "aqi_change_24h",
]

# --------------------------------------------------
# Fetch latest raw AQI data (online pipeline)
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

    response = requests.get(AQI_API_URL, params=params, timeout=30)
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
# Feature engineering (USED BY BACKFILL & ONLINE)
# --------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    # -------------------------
    # Time-based features
    # -------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # -------------------------
    # Lag features
    # -------------------------
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # -------------------------
    # Change-rate features
    # -------------------------
    df["aqi_change_1h"] = df["aqi"] - df["aqi"].shift(1)
    df["aqi_change_24h"] = df["aqi"] - df["aqi"].shift(24)

    # -------------------------
    # Target (next-hour AQI)
    # -------------------------
    df["target_aqi"] = df["aqi"].shift(-1)

    # -------------------------
    # Cleanup
    # -------------------------
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_aqi"]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# Save online features to MongoDB (optional)
# --------------------------------------------------
def save_to_mongodb(df: pd.DataFrame) -> None:
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    timestamps = df["timestamp"].tolist()

    collection.delete_many({
        "timestamp": {"$in": timestamps},
        "city": CITY,
    })

    records = df.to_dict(orient="records")

    for r in records:
        r["schema_version"] = 1
        r["feature_set"] = "v1"
        r["dataset_type"] = "online"

    if records:
        collection.insert_many(records)

    print(f"✅ Inserted {len(records)} online feature rows")


# --------------------------------------------------
# Online feature pipeline runner
# --------------------------------------------------
def run_feature_pipeline() -> pd.DataFrame:
    raw_df = fetch_raw_aqi_data()
    feature_df = engineer_features(raw_df)
    save_to_mongodb(feature_df)
    return feature_df


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    df = run_feature_pipeline()
    print("Sample engineered rows:")
    print(df.head())
    print(f"Generated {len(df)} feature rows")
