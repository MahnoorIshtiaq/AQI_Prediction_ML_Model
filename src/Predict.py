# src/Predict.py - PRODUCTION VERSION 3.0
# All improvements from IQ 188 evaluation implemented

import os
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
from collections import deque
from functools import lru_cache

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

FORECAST_HOURS = 72
MODEL_NAME = "AQI_NextHour_Model"

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidity", "wind_speed",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_24h",
]

# =============================================================================
# LOAD PRODUCTION MODEL (WITH CACHING)
# =============================================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load production model from MLflow with comprehensive error handling.
    Cached to avoid repeated network calls.
    
    Returns:
        mlflow.pyfunc.PyFuncModel: Loaded model
        
    Raises:
        ValueError: If credentials missing or model not found
        RuntimeError: If MLflow connection fails
    """
    print("🔄 Loading model from MLflow...")
    
    # Validate environment variables
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")
    
    if not all([owner, repo, token]):
        missing = []
        if not owner: missing.append("DAGSHUB_REPO_OWNER")
        if not repo: missing.append("DAGSHUB_REPO_NAME")
        if not token: missing.append("DAGSHUB_TOKEN")
        
        raise ValueError(
            f"❌ Missing DagsHub credentials: {', '.join(missing)}\n"
            f"Set these environment variables in .env file:\n"
            f"  DAGSHUB_REPO_OWNER=your_username\n"
            f"  DAGSHUB_REPO_NAME=your_repo\n"
            f"  DAGSHUB_TOKEN=your_token"
        )
    
    # Configure MLflow
    tracking_uri = f"https://dagshub.com/mahanoorishtiaq03/my-first-repo.mlflow/"
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"📡 MLflow URI: {tracking_uri}")
    
    # Load model with error handling
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        print(f"✅ Model loaded successfully: {MODEL_NAME}/Production")
        return model
        
    except mlflow.exceptions.MlflowException as e:
        error_msg = str(e)
        
        if "RESOURCE_DOES_NOT_EXIST" in error_msg:
            raise ValueError(
                f"❌ Model '{MODEL_NAME}' not found in Production stage.\n"
                f"Possible causes:\n"
                f"  1. Model hasn't been trained yet\n"
                f"  2. Model exists but not promoted to Production\n"
                f"  3. Model name mismatch\n\n"
                f"Solutions:\n"
                f"  1. Run training: python src/train.py\n"
                f"  2. Check DagsHub Model Registry: {tracking_uri.replace('.mlflow', '')}\n"
                f"  3. Verify MODEL_NAME in code matches registry"
            )
        
        elif "PERMISSION_DENIED" in error_msg or "401" in error_msg or "403" in error_msg:
            raise ValueError(
                f"❌ Authentication failed to DagsHub.\n"
                f"Possible causes:\n"
                f"  1. Invalid DAGSHUB_TOKEN\n"
                f"  2. Token expired\n"
                f"  3. Incorrect username/repo name\n\n"
                f"Solutions:\n"
                f"  1. Generate new token: https://dagshub.com/user/settings/tokens\n"
                f"  2. Verify credentials in .env\n"
                f"  3. Check tracking URI: {tracking_uri}"
            )
        
        else:
            raise RuntimeError(
                f"❌ MLflow error while loading model: {error_msg}\n"
                f"Tracking URI: {tracking_uri}\n"
                f"Model: {MODEL_NAME}/Production"
            )
    
    except Exception as e:
        raise RuntimeError(
            f"❌ Unexpected error loading model: {type(e).__name__}: {e}\n"
            f"This might indicate network issues or MLflow service problems."
        )


# =============================================================================
# LOAD RECENT HISTORY (WITH COMPREHENSIVE VALIDATION)
# =============================================================================

def load_recent_history():
    """
    Load last 24 hours of historical data with comprehensive validation.
    
    Returns:
        pd.DataFrame: Validated historical data with at least 24 hours
        
    Raises:
        ValueError: If data is insufficient, stale, or invalid
    """
    print("\n📊 Loading historical data...")
    
    # Connect to MongoDB
    try:
        client = MongoClient(
            os.getenv("MONGODB_URI"),
            serverSelectionTimeoutMS=5000
        )
        db = client[os.getenv("MONGODB_DB")]
        col = db[os.getenv("MONGODB_COLLECTION")]
    except Exception as e:
        raise RuntimeError(
            f"❌ MongoDB connection failed: {e}\n"
            f"Check MONGODB_URI in .env file"
        )
    
    # Fetch with margin (30 docs to handle potential gaps)
    docs = list(col.find(
        {"dataset_type": "online"},
        sort=[("timestamp", -1)],
        limit=30
    ))
    
    if not docs:
        raise ValueError(
            "❌ No historical data found in MongoDB.\n"
            "Possible causes:\n"
            "  1. Feature pipeline hasn't run yet\n"
            "  2. Collection is empty\n"
            "  3. Wrong collection name\n\n"
            "Solutions:\n"
            "  1. Run feature pipeline: python src/feature_pipeline.py\n"
            "  2. Check MongoDB collection exists\n"
            "  3. Verify MONGODB_COLLECTION in .env"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"   Fetched {len(df)} records from MongoDB")
    
    # =============================================================================
    # VALIDATION 1: DATA RECENCY
    # =============================================================================
    
    now = pd.Timestamp.utcnow()
    latest = df["timestamp"].max()
    age_hours = (now - latest).total_seconds() / 3600
    
    print(f"   Latest record: {latest} ({age_hours:.1f}h ago)")
    
    if age_hours > 3:
        raise ValueError(
            f"❌ Data is too stale!\n"
            f"Latest record: {latest}\n"
            f"Current time: {now}\n"
            f"Age: {age_hours:.1f} hours\n\n"
            f"Predictions require fresh data (< 3 hours old).\n"
            f"Run feature pipeline: python src/feature_pipeline.py"
        )
    
    # =============================================================================
    # VALIDATION 2: SUFFICIENT DATA
    # =============================================================================
    
    # Get last 24 hours
    cutoff = now - pd.Timedelta(hours=24)
    df_recent = df[df["timestamp"] >= cutoff].copy()
    
    print(f"   Records in last 24h: {len(df_recent)}")
    
    if len(df_recent) < 20:
        raise ValueError(
            f"❌ Insufficient recent data!\n"
            f"Found: {len(df_recent)} hours\n"
            f"Need: >= 20 hours (allows up to 4 hours missing)\n\n"
            f"Ensure feature pipeline runs hourly.\n"
            f"Check GitHub Actions or cron jobs."
        )
    
    # =============================================================================
    # VALIDATION 3: GAP DETECTION
    # =============================================================================
    
    # Check for large gaps between consecutive records
    time_diffs = df_recent["timestamp"].diff()
    max_gap_seconds = time_diffs.max().total_seconds()
    max_gap_hours = max_gap_seconds / 3600
    
    print(f"   Max gap between records: {max_gap_hours:.1f}h")
    
    if max_gap_hours > 2.5:
        gap_idx = time_diffs.idxmax()
        gap_start = df_recent.loc[gap_idx - 1, "timestamp"]
        gap_end = df_recent.loc[gap_idx, "timestamp"]
        
        raise ValueError(
            f"❌ Large data gap detected!\n"
            f"Gap: {max_gap_hours:.1f} hours\n"
            f"From: {gap_start}\n"
            f"To: {gap_end}\n\n"
            f"Predictions require continuous hourly data.\n"
            f"Check feature pipeline is running every hour."
        )
    
    # =============================================================================
    # VALIDATION 4: REQUIRED FIELDS
    # =============================================================================
    
    # CRITICAL FIX: Check for 'aqi' not 'target_aqi'
    required_fields = ["aqi"] + FEATURE_COLUMNS
    missing = [col for col in required_fields if col not in df_recent.columns]
    
    if missing:
        raise ValueError(
            f"❌ Missing required columns in MongoDB data!\n"
            f"Missing: {missing}\n"
            f"Available: {list(df_recent.columns)}\n\n"
            f"This indicates feature pipeline has a bug.\n"
            f"Check src/feature_pipeline.py"
        )
    
    # =============================================================================
    # VALIDATION 5: DATA QUALITY
    # =============================================================================
    
    # Check for negative AQI
    if (df_recent["aqi"] < 0).any():
        negative_count = (df_recent["aqi"] < 0).sum()
        raise ValueError(
            f"❌ Invalid negative AQI values detected!\n"
            f"Count: {negative_count}\n"
            f"This indicates corrupted data or API issues."
        )
    
    # Warn about extreme values
    if (df_recent["aqi"] > 500).any():
        extreme_count = (df_recent["aqi"] > 500).sum()
        max_aqi = df_recent["aqi"].max()
        print(f"   ⚠️ Warning: {extreme_count} extreme AQI values (>500, max: {max_aqi:.0f})")
    
    # Check for NaN in critical features
    null_counts = df_recent[required_fields].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"❌ Missing values detected in critical features!\n"
            f"Null counts: {null_cols}\n"
            f"All features must be present for predictions."
        )
    
    # =============================================================================
    # GET EXACTLY 24 HOURS
    # =============================================================================
    
    # Take the most recent 24 records
    df_24h = df_recent.tail(24).reset_index(drop=True)
    
    print(f"   ✅ Validation passed: {len(df_24h)} hours of clean data")
    print(f"   Date range: {df_24h['timestamp'].min()} to {df_24h['timestamp'].max()}")
    print(f"   AQI range: {df_24h['aqi'].min():.0f} - {df_24h['aqi'].max():.0f}")
    
    return df_24h


# =============================================================================
# DETERMINISTIC RECURSIVE FORECAST
# =============================================================================

def hourly_forecast():
    """
    Generate 72-hour deterministic forecast using recursive prediction.
    
    Returns:
        list: List of dicts with 'timestamp' and 'predicted_aqi'
    """
    
    print("\n" + "="*80)
    print("🔮 STARTING 72-HOUR FORECAST")
    print("="*80)
    
    # Load model and history
    model = load_model()
    history_df = load_recent_history()
    
    # Initialize from last available record
    last_row = history_df.iloc[-1].to_dict()
    current_time = last_row["timestamp"]
    
    print(f"\n📍 Starting from: {current_time}")
    print(f"📍 Initial AQI: {last_row['aqi']:.0f}")
    
    # CRITICAL FIX: Use 'aqi' not 'target_aqi'
    aqi_history = deque(history_df["aqi"].tolist(), maxlen=24)
    
    print(f"📊 Initialized 24-hour lag history")
    print(f"   Oldest: {aqi_history[0]:.0f} (24h ago)")
    print(f"   Newest: {aqi_history[-1]:.0f} (1h ago)")
    
    forecasts = []
    
    # Generate 72 hourly predictions
    for i in range(FORECAST_HOURS):
        
        # Move time forward by 1 hour
        current_time += timedelta(hours=1)
        
        # Update temporal features
        last_row["hour"] = current_time.hour
        last_row["day"] = current_time.day
        last_row["month"] = current_time.month
        last_row["day_of_week"] = current_time.dayofweek
        
        # Update lag features from history
        last_row["aqi_lag_1"] = aqi_history[-1]    # Most recent (1h ago)
        last_row["aqi_lag_24"] = aqi_history[0]    # Oldest (24h ago)
        
        # Update change features
        last_row["aqi_change_1h"] = aqi_history[-1] - aqi_history[-2]  # 1h change
        last_row["aqi_change_24h"] = aqi_history[-1] - aqi_history[0]  # 24h change
        
        # Keep exogenous features constant (deterministic approach)
        # NOTE: For better accuracy, integrate real weather forecast API here
        # last_row["temperature"] = fetch_weather_forecast(current_time)["temp"]
        # last_row["humidity"] = fetch_weather_forecast(current_time)["humidity"]
        # last_row["wind_speed"] = fetch_weather_forecast(current_time)["wind"]
        
        # Prepare input for model
        input_df = pd.DataFrame([{k: last_row[k] for k in FEATURE_COLUMNS}])
        
        # Predict
        pred = float(model.predict(input_df)[0])
        pred = max(0, pred)  # Ensure non-negative
        pred = min(500, pred)  # Cap at 500 (EPA AQI max)
        
        # Store prediction
        forecasts.append({
            "timestamp": current_time.isoformat(),
            "predicted_aqi": round(pred, 2)
        })
        
        # Update history with new prediction
        aqi_history.append(pred)
        
        # Progress indicator every 12 hours
        if (i + 1) % 12 == 0:
            print(f"   ⏳ Generated {i + 1}/{FORECAST_HOURS} predictions...")
    
    print(f"\n✅ Forecast complete: {len(forecasts)} hourly predictions")
    print("="*80)
    
    return forecasts


# =============================================================================
# PUBLIC API
# =============================================================================

def get_72h_forecast():
    """
    Public API function to get 72-hour AQI forecast.
    
    Returns:
        list: List of forecast dictionaries with:
            - timestamp (str): ISO format timestamp
            - predicted_aqi (float): Predicted AQI value
            
    Raises:
        ValueError: If data is insufficient or invalid
        RuntimeError: If model loading or prediction fails
    """
    try:
        return hourly_forecast()
    except Exception as e:
        print(f"\n❌ Forecast failed: {type(e).__name__}: {e}")
        raise


# =============================================================================
# TESTING & DIAGNOSTICS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 PREDICTION ENGINE - DIAGNOSTIC MODE")
    print("="*80)
    
    try:
        # Run forecast
        result = get_72h_forecast()
        
        # Display results
        print(f"\n📊 FORECAST SUMMARY")
        print("="*80)
        print(f"Total predictions: {len(result)}")
        
        # Extract AQI values
        aqis = [p['predicted_aqi'] for p in result]
        
        print(f"\n📈 Statistics:")
        print(f"   Mean AQI: {np.mean(aqis):.2f}")
        print(f"   Min AQI:  {np.min(aqis):.2f}")
        print(f"   Max AQI:  {np.max(aqis):.2f}")
        print(f"   Std Dev:  {np.std(aqis):.2f}")
        
        print(f"\n📋 First 5 predictions:")
        for pred in result[:5]:
            ts = pd.to_datetime(pred['timestamp'])
            print(f"   {ts.strftime('%Y-%m-%d %H:%M')}: AQI = {pred['predicted_aqi']:.1f}")
        
        print(f"\n📋 Last 5 predictions:")
        for pred in result[-5:]:
            ts = pd.to_datetime(pred['timestamp'])
            print(f"   {ts.strftime('%Y-%m-%d %H:%M')}: AQI = {pred['predicted_aqi']:.1f}")
        
        # Identify peaks and valleys
        max_idx = np.argmax(aqis)
        min_idx = np.argmin(aqis)
        
        print(f"\n🔺 Peak AQI:")
        peak = result[max_idx]
        print(f"   {pd.to_datetime(peak['timestamp']).strftime('%Y-%m-%d %H:%M')}: "
              f"AQI = {peak['predicted_aqi']:.1f}")
        
        print(f"\n🔻 Minimum AQI:")
        valley = result[min_idx]
        print(f"   {pd.to_datetime(valley['timestamp']).strftime('%Y-%m-%d %H:%M')}: "
              f"AQI = {valley['predicted_aqi']:.1f}")
        
        # Check for alerts
        print(f"\n🚨 Alerts:")
        alerts = [p for p in result if p['predicted_aqi'] > 150]
        if alerts:
            print(f"   ⚠️ {len(alerts)} hours with unhealthy AQI (>150)")
            print(f"   First alert: {pd.to_datetime(alerts[0]['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"   ✅ No unhealthy AQI levels in forecast")
        
        hazardous = [p for p in result if p['predicted_aqi'] > 300]
        if hazardous:
            print(f"   ☢️ {len(hazardous)} hours with hazardous AQI (>300)")
        
        print("\n" + "="*80)
        print("✅ DIAGNOSTIC COMPLETE - All systems operational")
        print("="*80)
        
    except ValueError as e:
        print(f"\n❌ DATA ERROR: {e}")
        print("\nThis is usually fixable by:")
        print("1. Running: python src/feature_pipeline.py")
        print("2. Running: python src/backfill.py (if fresh install)")
        print("3. Checking MongoDB connection")
        
    except RuntimeError as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        print("\nThis is usually fixable by:")
        print("1. Running: python src/train.py (to create model)")
        print("2. Checking DagsHub credentials in .env")
        print("3. Verifying internet connection")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
