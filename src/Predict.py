import os
import joblib
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "models/rf_aqi_model.pkl"
FORECAST_HOURS = 72


def load_latest_features():
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    df = pd.DataFrame(
        list(collection.find().sort("timestamp", -1).limit(24))
    ).sort_values("timestamp")

    df = df.drop(columns=["_id"])
    return df


def recursive_forecast(model, history_df):
    predictions = []

    current_df = history_df.copy()

    last_timestamp = current_df["timestamp"].iloc[-1]

    for step in range(FORECAST_HOURS):
        row = current_df.iloc[-1].copy()

        X = row[[
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
        ]].to_frame().T

        pred_aqi = model.predict(X)[0]

        next_time = last_timestamp + timedelta(hours=1)

        new_row = row.copy()
        new_row["timestamp"] = next_time
        new_row["aqi"] = pred_aqi

        # Update lag features
        new_row["aqi_lag_24"] = row["aqi_lag_23"] if "aqi_lag_23" in row else row["aqi_lag_24"]
        new_row["aqi_lag_12"] = row["aqi_lag_11"] if "aqi_lag_11" in row else row["aqi_lag_12"]
        new_row["aqi_lag_6"] = row["aqi_lag_5"] if "aqi_lag_5" in row else row["aqi_lag_6"]
        new_row["aqi_lag_3"] = row["aqi_lag_2"] if "aqi_lag_2" in row else row["aqi_lag_3"]
        new_row["aqi_lag_1"] = row["aqi"]

        new_row["aqi_change_rate"] = pred_aqi - row["aqi"]
        new_row["rolling_24h_mean"] = (
            current_df["aqi"].tail(23).sum() + pred_aqi
        ) / 24

        new_row["hour"] = next_time.hour
        new_row["day"] = next_time.day
        new_row["month"] = next_time.month
        new_row["day_of_week"] = next_time.dayofweek

        predictions.append({
            "timestamp": next_time,
            "predicted_aqi": pred_aqi
        })

        current_df = pd.concat([current_df, pd.DataFrame([new_row])])
        last_timestamp = next_time

    return pd.DataFrame(predictions)


def main():
    model = joblib.load(MODEL_PATH)
    history_df = load_latest_features()

    forecast_df = recursive_forecast(model, history_df)

    print(forecast_df.head())
    print("Forecast complete")

def get_72h_forecast():
    model = joblib.load(MODEL_PATH)
    history_df = load_latest_features()
    forecast_df = recursive_forecast(model, history_df)
    return forecast_df



#if __name__ == "__main__":
   # main()

