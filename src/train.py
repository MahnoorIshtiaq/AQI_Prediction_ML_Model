# src/train.py

import os
import sys
import traceback
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
# Configuration
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
REGISTERED_MODEL_NAME = "AQI_NextHour_Model"


# --------------------------------------------------
# Utility: Validate environment
# --------------------------------------------------
def validate_env():
    required = [
        "DAGSHUB_REPO_OWNER",
        "DAGSHUB_REPO_NAME",
        "DAGSHUB_TOKEN",
        "MONGODB_URI",
        "MONGODB_DB",
        "MONGODB_COLLECTION",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"❌ Missing required environment variables: {missing}")


# --------------------------------------------------
# Configure MLflow → DagsHub (CI SAFE)
# --------------------------------------------------
def configure_mlflow():
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")

    tracking_uri = f"https://dagshub.com/mahanoorishtiaq03/my-first-repo.mlflow/"

    # Explicit MLflow auth (NO OAuth flow)
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("🔗 MLflow configured")
    print(f"   Tracking URI: {tracking_uri}")


# --------------------------------------------------
# Load data from MongoDB (Feature Store)
# --------------------------------------------------
def load_training_data() -> pd.DataFrame:
    client = MongoClient(os.getenv("MONGODB_URI"))
    col = client[os.getenv("MONGODB_DB")][os.getenv("MONGODB_COLLECTION")]

    df = pd.DataFrame(
        list(col.find({"dataset_type": {"$in": ["historical", "online"]}}))
    )

    if df.empty:
        raise ValueError("❌ No data found in MongoDB")

    df = df.drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"📊 Loaded {len(df)} rows")
    return df


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": mse ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# --------------------------------------------------
# Time-based split
# --------------------------------------------------
def time_split(df):
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    val = df.iloc[split:]

    print(f"✂️ Train: {len(train)} | Validation: {len(val)}")

    return (
        train[FEATURE_COLUMNS],
        val[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
        val[TARGET_COLUMN],
    )


# --------------------------------------------------
# Model definitions
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
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    "Ridge": Ridge(alpha=1.0),
}


# --------------------------------------------------
# Training Pipeline
# --------------------------------------------------
def main():

    print("=" * 80)
    print("🚀 AQI TRAINING PIPELINE STARTED")
    print("=" * 80)

    validate_env()
    configure_mlflow()

    df = load_training_data()
    X_train, X_val, y_train, y_val = time_split(df)

    run_name = f"daily-train-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

    best_model = None
    best_rmse = float("inf")
    best_name = None

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        for name, model in MODELS.items():
            print(f"\n🤖 Training {name}")

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metrics = compute_metrics(y_val, preds)

            print(
                f"   RMSE={metrics['rmse']:.2f} | "
                f"MAE={metrics['mae']:.2f} | "
                f"R²={metrics['r2']:.3f}"
            )

            mlflow.log_metrics(
                {f"{name}_{k}": v for k, v in metrics.items()}
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_name = name

        print(f"\n🏆 Best Model: {best_name} (RMSE={best_rmse:.2f})")

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_rmse", best_rmse)

        # Log & Register model
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("📦 Model logged")

        # Promote to Production
        client = MlflowClient()
        latest_versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME
        )

        if latest_versions:
            latest_version = latest_versions[0]
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=latest_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"🚀 Model v{latest_version.version} → Production")

    print("\n✅ TRAINING COMPLETE")


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ Training failed")
        traceback.print_exc()
        sys.exit(1)
