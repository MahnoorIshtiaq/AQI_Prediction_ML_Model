import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8000//Predict"
CITY = "Karachi"
LAT, LON = 24.8607, 67.0011

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB")
COLLECTION = os.getenv("MONGODB_COLLECTION")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
# ---st.sidebar.title("🌍 AQI Dashboard")
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Overview", "Forecast", "Historical Trends", "Pollutants", "About AQI"],
        icons=["house", "graph-up", "bar-chart", "flask", "info-circle"],
        menu_icon="cast",
    )
   


dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# -----------------------------
# Theme handling (Streamlit-style)
# -----------------------------
if dark_mode:
    st.markdown("""
        <style>
            body { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def aqi_label(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 150:
        return "Unhealthy (Sensitive) 🟠"
    elif aqi <= 200:
        return "Unhealthy 🔴"
    else:
        return "Hazardous ☠️"

@st.cache_data(ttl=300)
def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current_weather": True
    }
    return requests.get(url, params=params).json()["current_weather"]

@st.cache_data(ttl=300)
def get_forecast():
    return pd.DataFrame(requests.get(API_URL).json()["forecast"])

@st.cache_data(ttl=300)
def get_past_week():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    since = datetime.utcnow() - timedelta(days=7)
    df = pd.DataFrame(list(col.find({"timestamp": {"$gte": since}})))
    return df.drop(columns=["_id"]).sort_values("timestamp")

# -----------------------------
# Data load
# -----------------------------
weather = get_weather()
forecast_df = get_forecast()
history_df = get_past_week()

current_aqi = int(history_df.iloc[-1]["aqi"])
severity = aqi_label(current_aqi)

# =============================
# OVERVIEW PAGE
# =============================
if selected == "Overview":
    st.title("🌍 Karachi Air Quality Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current AQI", current_aqi, severity)
    col2.metric("Temperature (°C)", weather["temperature"])
    col3.metric("Wind Speed (km/h)", weather["windspeed"])

    if current_aqi > 150:
        st.error(f"⚠️ Hazardous air quality detected (AQI {current_aqi})")
    elif current_aqi > 100:
        st.warning("⚠️ Unhealthy AQI levels expected")
    else:
        st.success("✅ AQI levels are acceptable")

# =============================
# FORECAST PAGE
# =============================
elif selected == "Forecast":
    st.title("📈 72-Hour AQI Forecast")
    st.line_chart(forecast_df.set_index("timestamp")["predicted_aqi"])
    st.dataframe(forecast_df)

# =============================
# HISTORICAL TRENDS
# =============================
elif selected == "Historical Trends":
    st.title("📊 AQI — Past 7 Days")
    st.line_chart(history_df.set_index("timestamp")["aqi"])

# =============================
# POLLUTANTS
# =============================
elif selected == "Pollutants":
    st.title("🧪 Pollutant Levels")

    col1, col2 = st.columns(2)
    col1.line_chart(history_df.set_index("timestamp")["pm2_5"])
    col2.line_chart(history_df.set_index("timestamp")["pm10"])

# =============================
# ABOUT AQI
# =============================
else:
    st.title("ℹ️ What is AQI?")

    st.markdown("""
    **Air Quality Index (AQI)** is a measure of air pollution and its health impact.

    | AQI | Category | Health Impact |
    |----|----|----|
    | 0–50 | Good | No risk |
    | 51–100 | Moderate | Sensitive people affected |
    | 101–150 | Unhealthy (Sensitive) | Asthma risk |
    | 151–200 | Unhealthy | Everyone affected |
    | 201+ | Hazardous | Emergency conditions |

    **Major Pollutants**
    - PM2.5 (fine particles)
    - PM10 (coarse particles)

    Source: US EPA
    """)

