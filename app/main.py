# app/main.py - CORRECTED VERSION

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.Predict import get_72h_forecast
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Karachi AQI Predictor",
    description="72-hour AQI forecast using ML models trained on historical data",
    version="2.0.0"
)

# ADDED: CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "AQI Prediction API",
        "version": "2.0.0"
    }

@app.get("/Predict")
def predict_aqi():
    """
    Generate 72-hour AQI forecast for Karachi
    
    Returns:
        dict: Contains city, forecast hours, and hourly predictions
    """
    try:
        data = get_72h_forecast()
        
        return {
            "city": "Karachi",
            "latitude": 24.8607,
            "longitude": 67.0011,
            "hours": 72,
            "forecast": data,
            "model": "AQI_NextHour_Model",
            "model_stage": "Production"
        }
    
    except Exception as e:
        # ADDED: Proper error handling
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}"
        )

# ADDED: Metadata endpoint for dashboard
@app.get("/info")
def get_info():
    """Get API and model information"""
    return {
        "city": "Karachi",
        "forecast_horizon_hours": 72,
        "features_used": [
            "pm2_5", "pm10", "no2", "o3", "co", "so2",
            "temperature", "humidity", "wind_speed",
            "hour", "day", "month", "day_of_week",
            "aqi_lag_1", "aqi_lag_24",
            "aqi_change_1h", "aqi_change_24h"
        ],
        "model_name": "AQI_NextHour_Model",
        "update_frequency": "hourly",
        "training_frequency": "daily"
    }
