from fastapi import FastAPI
from src.Predict import get_72h_forecast

app = FastAPI(
    title="Karachi AQI Predictor",
    description="72-hour AQI forecast using ML",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/Predict")
def predict_aqi():
    df = get_72h_forecast()

    return {
        "city": "Karachi",
        "horizon_hours": 72,
        "forecast": df.to_dict(orient="records")
    }
