# src/train.py
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
    "pm2_5", "pm10", "no2", "o3", "co", "so2",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_24h",
]

TARGET_COLUMN = "target_aqi"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# Load training data
# --------------------------------------------------
def load_training_data() -> pd.DataFrame:
    col = MongoClient(
        os.getenv("MONGODB_URI")
    )[os.getenv("MONGODB_DB")]["aqi_features_v1"]

    df = pd.DataFrame(
        list(col.find({"dataset_type": "online"}))
    )

    if df.empty:
        raise ValueError("No data found in MongoDB")

    df = df.drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df.sort_values("timestamp").reset_index(drop=True)

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
# Time-based split (NO leakage)
# --------------------------------------------------
def time_split(df: pd.DataFrame):
    split = int(len(df) * 0.8)

    train, val = df.iloc[:split], df.iloc[split:]

    return (
        train[FEATURE_COLUMNS],
        val[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
        val[TARGET_COLUMN],
    )

# --------------------------------------------------
# Models
# --------------------------------------------------
MODELS = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    ),
    "Ridge": Ridge(alpha=1.0),
}

# --------------------------------------------------
# Train + select best
# --------------------------------------------------
def train_models(df: pd.DataFrame):
    X_train, X_val, y_train, y_val = time_split(df)

    results = {}
    trained = {}

    for name, model in MODELS.items():
        print(f"Training {name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        metrics = compute_metrics(y_val, preds)
        results[name] = metrics
        trained[name] = model

        print(
            f"   RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f} | "
            f"R²={metrics['r2']:.3f}"
        )

    best_name = min(results, key=lambda m: results[m]["rmse"])
    return best_name, trained[best_name], results

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
    print(f"Training rows: {len(df)}")

    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    with mlflow.start_run(run_name=f"daily-train-{run_date}"):

        mlflow.log_params({
            "feature_set": "v2",
            "granularity": "hourly",
            "target": TARGET_COLUMN,
        })

        best_name, best_model, metrics = train_models(df)

        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        mlflow.log_param("best_model", best_name)

        for model_name, m in metrics.items():
            mlflow.log_metrics({
                f"{model_name}_rmse": m["rmse"],
                f"{model_name}_mae": m["mae"],
                f"{model_name}_r2": m["r2"],
            })

        print(f"\nBest model: {best_name}")
        print(f"Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
