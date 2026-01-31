import os
import joblib
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import dagshub
import mlflow

load_dotenv()

FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_change_rate",
    "rolling_24h_mean",
]

TARGET_COLUMN = "target_aqi"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_training_data():
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    df = pd.DataFrame(list(collection.find()))
    if df.empty:
        raise ValueError("MongoDB returned no rows")

    df = df.drop(columns=["_id"])
    print(f"Loaded {len(df)} rows from MongoDB")
    return df

def evaluate(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def main():
    dagshub.init(
        repo_owner="mahanoorishtiaq03",
        repo_name="my-first-repo",
        mlflow=True,
    )

    mlflow.set_experiment("AQI_Prediction")

    df = load_training_data()

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"Rows after NaN filtering: {len(df)}")

    if df.empty:
        raise ValueError("No valid training rows after NaN filtering")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }

    best_model = None
    best_rmse = float("inf")
    best_name = None

    with mlflow.start_run():
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = evaluate(y_test, preds)
            print(f"\n{name} metrics:", metrics)

            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_name = name

        model_path = os.path.join(MODEL_DIR, "best_aqi_model.pkl")
        joblib.dump(best_model, model_path)

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_artifact(model_path)

        print("\n🏆 Best model:", best_name)
        print("🔥 Best RMSE:", best_rmse)

if __name__ == "__main__":
    main()
