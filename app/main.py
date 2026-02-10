# app/main.py

from fastapi import FastAPI
from src.Predict import get_72h_forecast

app = FastAPI(
    title="Karachi AQI Predictor",
    description="Hourly AQI forecast for the next 3 days",
    version="1.0.0",
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/Predict")
def predict():
    """
    Returns hourly AQI forecast for next 72 hours
    """
    return get_72h_forecast()
