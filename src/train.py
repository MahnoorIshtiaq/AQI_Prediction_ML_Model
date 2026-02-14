# src/train.py

import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Config - CORRECTED: Added weather features
# --------------------------------------------------
FEATURE_COLUMNS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidity", "wind_speed",  
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_24h",
]

TARGET_COLUMN = "target_aqi"

EXPERIMENT_NAME = "AQI_NextHour_Forecasting"

# --------------------------------------------------
# Load training data - CORRECTED
# --------------------------------------------------
def load_training_data() -> pd.DataFrame:
    """Load both historical and online data for training"""
    col = MongoClient(
        os.getenv("MONGODB_URI")
    )[os.getenv("MONGODB_DB")][os.getenv("MONGODB_COLLECTION")]

    df = pd.DataFrame(
        list(col.find({"dataset_type": {"$in": ["historical", "online"]}}))
    )

    if df.empty:
        raise ValueError("❌ No data found in MongoDB")

    df = df.drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop rows with missing features or target
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    print(f"📊 Loaded {len(df)} total rows")
    print(f"   - Historical: {len(df[df['dataset_type'] == 'historical'])}")
    print(f"   - Online: {len(df[df['dataset_type'] == 'online'])}")

    return df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

# --------------------------------------------------
# Time-based split
# --------------------------------------------------
def time_split(df: pd.DataFrame):
    """80/20 train/validation split preserving temporal order"""
    split = int(len(df) * 0.8)

    train, val = df.iloc[:split], df.iloc[split:]

    print(f"✂️ Train: {len(train)} rows | Validation: {len(val)} rows")

    return (
        train[FEATURE_COLUMNS],
        val[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
        val[TARGET_COLUMN],
    )

# --------------------------------------------------
# Models - CORRECTED: Improved hyperparameters
# --------------------------------------------------
MODELS = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,  
        min_samples_split=10,  
        min_samples_leaf=4,  
        subsample=0.8,  
        random_state=42,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,  
        min_samples_split=10,  
        min_samples_leaf=4,  
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    ),
    "Ridge": Ridge(
        alpha=1.0,
        random_state=42
    ),
}

# --------------------------------------------------
# Main  GitHub Actions compatible DagsHub/MLflow setup
# --------------------------------------------------
def main():
    """Main training pipeline"""
    
    # Initialize MLflow with DagsHub tracking URI

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_username = os.getenv("DAGSHUB_USERNAME", repo_owner)
    
    if not all([repo_owner, repo_name, dagshub_token]):
        raise ValueError("❌ Missing DagsHub credentials in environment variables")

    mlflow_tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set DagsHub credentials for authentication
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    print(f"🔗 MLflow Tracking URI: {mlflow_tracking_uri}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    df = load_training_data()
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Split data
    X_train, X_val, y_train, y_val = time_split(df)

    run_date = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")

    with mlflow.start_run(run_name=f"daily-train-{run_date}"):

        # Log global parameters
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("date_range_start", str(df['timestamp'].min()))
        mlflow.log_param("date_range_end", str(df['timestamp'].max()))

        best_model = None
        best_rmse = float("inf")
        best_name = None

        # Train each model
        for name, model in MODELS.items():

            print(f"\n🚀 Training {name}...")

            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            metrics = compute_metrics(y_val, preds)

            mlflow.log_metrics({
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            })
            
            mlflow.log_metrics({
                f"{name}_rmse": metrics["rmse"],
                f"{name}_mae": metrics["mae"],
                f"{name}_r2": metrics["r2"],
            })
            
            mlflow.set_tag("model_name", name)

            print(
                f"   ✅ RMSE={metrics['rmse']:.2f} | "
                f"MAE={metrics['mae']:.2f} | "
                f"R²={metrics['r2']:.3f}"
            )

            # Track best model
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_name = name

        print(f"\n🏆 Best model: {best_name} (RMSE: {best_rmse:.2f})")

        # Log best model info
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_rmse", best_rmse)

        # Register best model
        print("\n📦 Registering best model...")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="AQI_NextHour_Model"
        )

        # Promote to Production
        client = MlflowClient()
        latest_version = client.get_latest_versions(
            name="AQI_NextHour_Model",
            stages=["None"]
        )[0]
        
        client.transition_model_version_stage(
            name="AQI_NextHour_Model",
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Model version {latest_version.version} promoted to Production")

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
