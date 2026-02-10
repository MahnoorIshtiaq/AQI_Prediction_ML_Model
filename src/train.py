import os
import joblib
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import dagshub
import mlflow

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Config
# --------------------------------------------------
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

TARGET_COLUMN = "target_aqi"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# MongoDB loader
# --------------------------------------------------
def load_training_data() -> pd.DataFrame:
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db["training_features"]

    df = pd.DataFrame(list(collection.find({
        "dataset_type": "training"
    })))

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }



# --------------------------------------------------
# Time-aware train/validation split
# --------------------------------------------------
def time_split(df: pd.DataFrame):
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    return X_train, X_val, y_train, y_val


# --------------------------------------------------
# Models to compare
# --------------------------------------------------
MODELS = {
    "gbr": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    ),
    "rf": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
    "ridge": Ridge(alpha=1.0),
}


# --------------------------------------------------
# Training routine
# --------------------------------------------------
def train_models(df: pd.DataFrame):
    # Safety check
    if len(df) < 100:
        raise ValueError(
            f"Not enough training data. Found only {len(df)} rows."
        )

    X_train, X_val, y_train, y_val = time_split(df)

    results = {}
    trained_models = {}

    for name, model in MODELS.items():
        print(f"🚀 Training model: {name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        metrics = compute_metrics(y_val, preds)
        results[name] = metrics
        trained_models[name] = model

        print(
            f"   RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, "
            f"R²={metrics['r2']:.3f}"
        )

    best_model_name = min(results, key=lambda m: results[m]["rmse"])
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, results


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    dagshub.init(
        repo_owner="mahanoorishtiaq03",
        repo_name="my-first-repo",
        mlflow=True,
    )

    mlflow.set_experiment("AQI_NextHour_Forecasting")

    df = load_training_data()
    print(f"📊 Loaded {len(df)} training rows")

    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    with mlflow.start_run(run_name=f"daily-train-{run_date}"):

        mlflow.log_param("feature_set", "v1")
        mlflow.log_param("training_frequency", "daily")
        mlflow.log_param("data_granularity", "hourly")
        mlflow.log_param("target", TARGET_COLUMN)

        best_name, best_model, metrics = train_models(df)

        model_path = os.path.join(MODEL_DIR, "aqi_model.pkl")
        joblib.dump(best_model, model_path)

        print(f"\n✅ Best model: {best_name}")
        print(f"📦 Model saved to: {model_path}")

        for model_name, m in metrics.items():
            mlflow.log_metrics({
                f"{model_name}_rmse": m["rmse"],
                f"{model_name}_mae": m["mae"],
                f"{model_name}_r2": m["r2"],
            })

        mlflow.log_param("best_model", best_name)
        mlflow.log_artifact(model_path)


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()
