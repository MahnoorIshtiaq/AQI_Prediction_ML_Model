import os
import joblib
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import dagshub
import mlflow

load_dotenv()

# -----------------------
# Config
# -----------------------
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
    "rolling_24h_mean"
]

TARGET_COLUMN = "aqi"

MODEL_DIR = "models"
MODEL_NAME = "rf_aqi_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------
# Load training data
# -----------------------
def load_training_data():
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db["training_features"]

    df = pd.DataFrame(list(collection.find({"dataset_type": "training"})))
    df = df.drop(columns=["_id"])

    return df


# -----------------------
# Train model
# -----------------------
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    return model, metrics


# -----------------------
# Main
# -----------------------
def main():
    
    dagshub.init(repo_owner='mahanoorishtiaq03', repo_name='my-first-repo', mlflow=True)

    mlflow.set_experiment("AQI_Prediction")

    df = load_training_data()

    with mlflow.start_run():
        model, metrics = train_model(df)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)

        print("Training complete")
        print(metrics)


if __name__ == "__main__":
    main()
