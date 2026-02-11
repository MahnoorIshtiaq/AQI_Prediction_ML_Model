# app/dashboard.py

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import mlflow
import altair as alt
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Config
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/Predict"
CITY = "Karachi"

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB")
COLLECTION = os.getenv("MONGODB_COLLECTION")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

mlflow.set_tracking_uri(MLFLOW_URI)

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Overview", "Forecast", "Historical Trends", "Model Comparison", "Alerts"],
        icons=["house", "graph-up", "bar-chart", "cpu", "exclamation-triangle"],
        menu_icon="cast",
    )

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------
# AQI Helpers
# --------------------------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def aqi_color(aqi):
    if aqi <= 50:
        return "#00E400"
    elif aqi <= 100:
        return "#FFFF00"
    elif aqi <= 150:
        return "#FF7E00"
    elif aqi <= 200:
        return "#FF0000"
    elif aqi <= 300:
        return "#8F3F97"
    else:
        return "#7E0023"

# --------------------------------------------------
# Data Loaders
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_forecast():
    df = pd.DataFrame(requests.get(API_URL).json()["forecast"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Category"] = df["aqi"].apply(aqi_category)
    return df

@st.cache_data(ttl=300)
def get_history():
    client = MongoClient(MONGO_URI)
    since = datetime.utcnow() - timedelta(days=7)

    df = pd.DataFrame(
        list(client[DB_NAME][COLLECTION].find({"timestamp": {"$gte": since}}))
    )

    if df.empty:
        return df

    df = df.drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")

@st.cache_data(ttl=300)
def get_latest_model_metrics():
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.model_name != ''",
    )

    if runs.empty:
        return pd.DataFrame()

    df = runs[[
        "tags.model_name",
        "metrics.rmse",
        "metrics.mae",
        "metrics.r2",
    ]].copy()

    df.columns = ["Model", "RMSE", "MAE", "R2"]
    return df.sort_values("RMSE").reset_index(drop=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
forecast_df = get_forecast()
history_df = get_history()
models_df = get_latest_model_metrics()

# Latest values
if not history_df.empty:
    latest = history_df.iloc[-1]
    current_aqi = int(latest["aqi"])
    temperature = latest.get("temperature", "N/A")
    wind = latest.get("wind_speed", "N/A")
    humidity = latest.get("humidity", "N/A")

    pm25 = latest.get("pm2_5", "N/A")
    pm10 = latest.get("pm10", "N/A")
    no2 = latest.get("no2", "N/A")
    o3 = latest.get("o3", "N/A")
    so2 = latest.get("so2", "N/A")
else:
    current_aqi = 0
    temperature = wind = humidity = "N/A"
    pm25 = pm10 = no2 = o3 = so2 =  "N/A"

# ==================================================
# OVERVIEW
# ==================================================
if selected == "Overview":

    st.title("🌍 Karachi AQI Overview")

    color = aqi_color(current_aqi)

    st.markdown(f"""
    <div style="
        background-color:{color};
        padding:40px;
        border-radius:20px;
        text-align:center;
        font-size:60px;
        font-weight:bold;
        color:black;">
        AQI {current_aqi}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🌡️ Weather")

    w1, w2, w3 = st.columns(3)
    w1.metric("Temperature", f"{temperature} °C")
    w2.metric("Wind Speed", f"{wind} m/s")
    w3.metric("Humidity", f"{humidity}%")

    st.markdown("## 💨 Pollutants")

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("PM2.5", f"{pm25} μg/m³")
    p2.metric("PM10", f"{pm10} μg/m³")
    p3.metric("NO₂", f"{no2} μg/m³")
    p4.metric("O₃", f"{o3} μg/m³")
    p5.metric("SO₂", f"{so2} μg/m³")

    st.markdown("---")
    st.subheader("📈 3-Day AQI Forecast")

    chart = alt.Chart(forecast_df).mark_line(point=True).encode(
        x="timestamp:T",
        y="aqi:Q",
        color=alt.Color(
            "Category:N",
            scale=alt.Scale(
                domain=[
                    "Good", "Moderate", "Unhealthy (Sensitive)",
                    "Unhealthy", "Very Unhealthy", "Hazardous"
                ],
                range=[
                    "#00E400", "#FFFF00", "#FF7E00",
                    "#FF0000", "#8F3F97", "#7E0023"
                ]
            )
        )
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

# ==================================================
# FORECAST
# ==================================================
elif selected == "Forecast":

    st.title("📈 Hourly AQI Forecast (Next 72 Hours)")

    chart = alt.Chart(forecast_df).mark_line(point=True).encode(
        x="timestamp:T",
        y="aqi:Q",
        color="Category:N"
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

    def highlight(val):
        return f"background-color:{aqi_color(val)}; color:black;"

    styled = forecast_df.style.applymap(
        highlight, subset=["aqi"]
    )

    st.dataframe(styled, use_container_width=True)

    st.markdown("""
    ### AQI Scale
    🟢 0-50: Good  
    🟡 51-100: Moderate  
    🟠 101-150: Unhealthy (Sensitive)  
    🔴 151-200: Unhealthy  
    🟣 201-300: Very Unhealthy  
    ⚫ 301+: Hazardous  
    """)

# ==================================================
# HISTORICAL
# ==================================================
elif selected == "Historical Trends":

    st.title("📊 AQI — Past 7 Days")

    if history_df.empty:
        st.warning("No historical data found.")
    else:
        hist_chart = alt.Chart(history_df).mark_line().encode(
            x="timestamp:T",
            y="aqi:Q"
        ).properties(height=400)

        st.altair_chart(hist_chart, use_container_width=True)

# ==================================================
# MODEL COMPARISON
# ==================================================
elif selected == "Model Comparison":

    st.title("🤖 Model Performance Comparison")

    if models_df.empty:
        st.warning("No trained models found in MLflow yet.")
    else:
        best = models_df.iloc[0]["Model"]
        st.success(f"🏆 Best Model: {best}")

        st.dataframe(models_df, use_container_width=True)

        st.bar_chart(models_df.set_index("Model")["RMSE"])

# ==================================================
# ALERTS
# ==================================================
else:

    st.title("🚨 AQI Alerts")

    if forecast_df.empty:
        st.warning("No forecast data available.")
    else:
        max_aqi = forecast_df["aqi"].max()

        if max_aqi > 300:
            st.error("⚫ Hazardous air quality expected!")
        elif max_aqi > 200:
            st.error("🟣 Very Unhealthy conditions expected.")
        elif max_aqi > 150:
            st.warning("🔴 Unhealthy air quality ahead.")
        elif max_aqi > 100:
            st.warning("🟠 Sensitive groups should be cautious.")
        else:
            st.success("🟢 Air quality stable for next 72 hours.")

        st.markdown("""
        ### AQI Scale
        🟢 0-50: Good  
        🟡 51-100: Moderate  
        🟠 101-150: Unhealthy (Sensitive)  
        🔴 151-200: Unhealthy  
        🟣 201-300: Very Unhealthy  
        ⚫ 301+: Hazardous  
        """)
