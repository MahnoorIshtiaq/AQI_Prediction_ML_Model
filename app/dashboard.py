# app/dashboard.py - CORRECTED VERSION

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

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)

# --------------------------------------------------
# Page Setup
# --------------------------------------------------

st.set_page_config(
    page_title=f"{CITY} AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# --------------------------------------------------
# AQI Helpers
# --------------------------------------------------

def aqi_category(aqi):
    """Categorize AQI value"""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy (Sensitive)"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def aqi_color(aqi):
    """Get color for AQI value"""
    if aqi <= 50: return "#00E400"
    elif aqi <= 100: return "#FFFF00"
    elif aqi <= 150: return "#FF7E00"
    elif aqi <= 200: return "#FF0000"
    elif aqi <= 300: return "#8F3F97"
    else: return "#7E0023"

# --------------------------------------------------
# Data Loaders
# --------------------------------------------------

@st.cache_data(ttl=60)
def get_forecast():
    """Fetch forecast from API"""
    try:
        response = requests.get(API_URL, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "forecast" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["forecast"])
        if df.empty:
            return df

        df.rename(columns={"predicted_aqi": "aqi"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Karachi")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["Category"] = df["aqi"].apply(aqi_category)

        return df

    except Exception as e:
        st.error(f"❌ Forecast API Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_history():
    """Fetch historical data from MongoDB"""
    if not MONGO_URI:
        return pd.DataFrame()

    try:
        client = MongoClient(MONGO_URI)
        since = datetime.utcnow() - timedelta(days=7)

        # CORRECTED: Load from correct collection
        records = list(
            client[DB_NAME][COLLECTION].find(
                {
                    "timestamp": {"$gte": since},
                    "dataset_type": "online"  # ADDED: Only online for recent history
                }
            )
        )

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df.drop(columns=["_id"], errors="ignore", inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df.sort_values("timestamp")

    except Exception as e:
        st.error(f"❌ MongoDB Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_latest_model_metrics():
    """
    CORRECTED: Get model comparison data from MLflow
    """
    if not EXPERIMENT_NAME or not MLFLOW_URI:
        return pd.DataFrame(), None

    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return pd.DataFrame(), None

        # Get latest runs
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=10  # Get recent runs
        )

        if runs.empty:
            return pd.DataFrame(), None

        # CORRECTED: Check for model_name tag
        if "tags.model_name" not in runs.columns:
            # Fallback: Try to extract from metrics
            model_metrics = []
            
            for _, run in runs.head(3).iterrows():  # Latest 3 runs
                # Extract model-specific metrics
                for model in ["GradientBoosting", "RandomForest", "Ridge"]:
                    rmse_col = f"metrics.{model}_rmse"
                    mae_col = f"metrics.{model}_mae"
                    r2_col = f"metrics.{model}_r2"
                    
                    if all(col in run.index for col in [rmse_col, mae_col, r2_col]):
                        model_metrics.append({
                            "Model": model,
                            "RMSE": run[rmse_col],
                            "MAE": run[mae_col],
                            "R2": run[r2_col]
                        })
            
            if model_metrics:
                df = pd.DataFrame(model_metrics)
                df = df.drop_duplicates(subset=["Model"], keep="first")
                df = df.sort_values("RMSE").reset_index(drop=True)
            else:
                df = pd.DataFrame()
        else:
            # Use tagged runs (preferred method)
            required = [
                "tags.model_name",
                "metrics.rmse",
                "metrics.mae",
                "metrics.r2",
            ]

            if not all(col in runs.columns for col in required):
                return pd.DataFrame(), None

            df = runs[required].copy()
            df.columns = ["Model", "RMSE", "MAE", "R2"]
            df = df.drop_duplicates(subset=["Model"], keep="first")
            df = df.sort_values("RMSE").reset_index(drop=True)

        # Get Production model version
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.get_latest_versions(
                name="AQI_NextHour_Model",
                stages=["Production"]
            )
            prod_version = versions[0].version if versions else None
        except Exception:
            prod_version = None

        return df, prod_version

    except Exception as e:
        st.error(f"❌ MLflow Error: {e}")
        return pd.DataFrame(), None


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

    # ADDED: System info
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.markdown(f"**City:** {CITY}")
    st.markdown(f"**Forecast:** 72 hours")
    st.markdown(f"**Updated:** Every hour")

# --------------------------------------------------
# Load Data
# --------------------------------------------------

forecast_df = get_forecast()
history_df = get_history()
models_df, production_version = get_latest_model_metrics()


# Latest snapshot
if not history_df.empty and "aqi" in history_df.columns:
    latest = history_df.iloc[-1]
    current_aqi = int(latest.get("aqi", 0))
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
    pm25 = pm10 = no2 = o3 = so2 = "N/A"

# ==================================================
# OVERVIEW
# ==================================================

if selected == "Overview":

    st.title(f"🌍 {CITY} AQI Overview")

    # AQI Block
    color = aqi_color(current_aqi)

    st.markdown(
        f"""
        <div style="background-color:{color};
                    padding:40px;
                    border-radius:20px;
                    text-align:center;
                    font-size:60px;
                    font-weight:bold;
                    color:black;">
            AQI {current_aqi}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"### Category: **{aqi_category(current_aqi)}**")

    st.markdown("---")

    # Weather
    st.subheader("🌡 Weather Conditions")
    w1, w2, w3 = st.columns(3)
    w1.metric("Temperature (°C)", temperature)
    w2.metric("Wind Speed (m/s)", wind)
    w3.metric("Humidity (%)", humidity)

    st.markdown("---")

    # Pollutants
    st.subheader("💨 Pollutant Levels (μg/m³)")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("PM2.5", pm25)
    p2.metric("PM10", pm10)
    p3.metric("NO₂", no2)
    p4.metric("O₃", o3)
    p5.metric("SO₂", so2)

    st.markdown("---")

    # Forecast
    if not forecast_df.empty:
        st.subheader("📈 3-Day AQI Forecast")

        chart = alt.Chart(forecast_df).mark_line(point=True).encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("aqi:Q", title="AQI"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=[
                        "Good","Moderate","Unhealthy (Sensitive)",
                        "Unhealthy","Very Unhealthy","Hazardous"
                    ],
                    range=[
                        "#00E400","#FFFF00","#FF7E00",
                        "#FF0000","#8F3F97","#7E0023"
                    ],
                ),
                legend=alt.Legend(title="Category")
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("aqi:Q", title="AQI", format=".1f"),
                alt.Tooltip("Category:N", title="Category")
            ]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("⚠️ No forecast data available. Please check API connection.")

# ==================================================
# FORECAST
# ==================================================

elif selected == "Forecast":

    st.title("📈 Hourly AQI Forecast (Next 72 Hours)")

    if forecast_df.empty:
        st.warning("⚠️ No forecast data available.")
    else:
        chart = alt.Chart(forecast_df).mark_line(point=True).encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("aqi:Q", title="AQI"),
            color=alt.Color("Category:N", legend=alt.Legend(title="Category")),
            tooltip=["timestamp:T", "aqi:Q", "Category:N"]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

        # CORRECTED: Better table formatting
        st.subheader("📋 Detailed Forecast Table")
        
        display_df = forecast_df[["timestamp", "aqi", "Category"]].copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df.columns = ["Time", "AQI", "Category"]
        
        def highlight_row(row):
            """Highlight row based on AQI"""
            color = aqi_color(row["AQI"])
            return [f"background-color:{color}; color:black;" for _ in row]

        styled = display_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled, use_container_width=True, height=400)

# ==================================================
# HISTORICAL
# ==================================================

elif selected == "Historical Trends":

    st.title("📊 AQI — Past 7 Days")

    if history_df.empty:
        st.warning("⚠️ No historical data found.")
    else:
        # ADDED: Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average AQI", f"{history_df['aqi'].mean():.1f}")
        col2.metric("Max AQI", f"{history_df['aqi'].max():.1f}")
        col3.metric("Min AQI", f"{history_df['aqi'].min():.1f}")
        col4.metric("Current Trend", 
                   "📈" if history_df['aqi'].iloc[-1] > history_df['aqi'].iloc[-24] else "📉")

        st.markdown("---")

        # Chart
        hist_chart = alt.Chart(history_df).mark_line().encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("aqi:Q", title="AQI"),
            tooltip=["timestamp:T", "aqi:Q"]
        ).properties(height=400)

        st.altair_chart(hist_chart, use_container_width=True)

# ==================================================
# MODEL COMPARISON
# ==================================================

elif selected == "Model Comparison":

    st.title("🤖 Model Performance Comparison")

    if models_df.empty:
        st.warning("⚠️ No trained models found in MLflow.")
        st.info("💡 Models will appear after running the training pipeline.")
    else:

        if production_version:
            st.success(f"🚀 Production Model Version: {production_version}")

        best = models_df.iloc[0]["Model"]
        best_rmse = models_df.iloc[0]["RMSE"]
        
        st.success(f"🏆 Best Model: **{best}** (RMSE: {best_rmse:.2f})")

        st.markdown("---")

        # ADDED: Formatted metrics table
        st.subheader("📊 Model Metrics Comparison")
        
        formatted_df = models_df.copy()
        formatted_df["RMSE"] = formatted_df["RMSE"].round(2)
        formatted_df["MAE"] = formatted_df["MAE"].round(2)
        formatted_df["R2"] = formatted_df["R2"].round(3)
        
        st.dataframe(formatted_df, use_container_width=True)

        st.markdown("---")

        # RMSE Comparison Chart
        st.subheader("📊 RMSE Comparison")

        bar_chart = alt.Chart(models_df).mark_bar().encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("RMSE:Q", title="RMSE"),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model:N", "RMSE:Q", "MAE:Q", "R2:Q"]
        ).properties(height=400)

        st.altair_chart(bar_chart, use_container_width=True)

        # ADDED: R² Comparison
        st.subheader("📊 R² Score Comparison")
        
        r2_chart = alt.Chart(models_df).mark_bar().encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("R2:Q", title="R² Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model:N", "R2:Q"]
        ).properties(height=400)

        st.altair_chart(r2_chart, use_container_width=True)

# ==================================================
# ALERTS
# ==================================================

else:

    st.title("🚨 AQI Alerts")

    if forecast_df.empty:
        st.warning("⚠️ No forecast data available.")
    else:
        max_aqi = forecast_df["aqi"].max()
        max_time = forecast_df.loc[forecast_df["aqi"].idxmax(), "timestamp"]

        # ADDED: More detailed alerts
        if max_aqi > 300:
            st.error("⚫ **HAZARDOUS** air quality expected!")
            st.error(f"Peak AQI: **{max_aqi:.0f}** at {max_time.strftime('%Y-%m-%d %H:%M')}")
            st.markdown("""
            **Health Implications:**
            - Health warnings of emergency conditions
            - Everyone should avoid all outdoor exertion
            
            **Recommendations:**
            - Stay indoors with air purifiers
            - Seal windows and doors
            - Wear N95 masks if you must go outside
            """)
        elif max_aqi > 200:
            st.error("🟣 **VERY UNHEALTHY** conditions expected.")
            st.warning(f"Peak AQI: **{max_aqi:.0f}** at {max_time.strftime('%Y-%m-%d %H:%M')}")
            st.markdown("""
            **Health Implications:**
            - Health alert: everyone may experience serious effects
            
            **Recommendations:**
            - Avoid outdoor activities
            - Keep indoor air clean
            - Children and elderly stay indoors
            """)
        elif max_aqi > 150:
            st.warning("🔴 **UNHEALTHY** air quality ahead.")
            st.warning(f"Peak AQI: **{max_aqi:.0f}** at {max_time.strftime('%Y-%m-%d %H:%M')}")
            st.markdown("""
            **Health Implications:**
            - Everyone may begin to experience health effects
            
            **Recommendations:**
            - Limit prolonged outdoor activities
            - Sensitive groups should stay indoors
            """)
        elif max_aqi > 100:
            st.warning("🟠 Sensitive groups should be cautious.")
            st.info(f"Peak AQI: **{max_aqi:.0f}** at {max_time.strftime('%Y-%m-%d %H:%M')}")
            st.markdown("""
            **Health Implications:**
            - Unhealthy for sensitive groups
            
            **Recommendations:**
            - People with respiratory issues limit outdoor exertion
            """)
        else:
            st.success("🟢 Air quality stable for next 72 hours.")
            st.success(f"Max AQI: **{max_aqi:.0f}**")
            st.markdown("""
            **Status:** Air quality is satisfactory
            
            **Recommendations:**
            - Normal outdoor activities are safe
            """)

        st.markdown("---")
        
        # ADDED: Hourly breakdown
        st.subheader("📅 Hourly Alert Timeline")
        
        alert_df = forecast_df[["timestamp", "aqi", "Category"]].copy()
        alert_df = alert_df[alert_df["aqi"] > 100]  # Only show concerning levels
        
        if not alert_df.empty:
            alert_df["timestamp"] = alert_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            alert_df.columns = ["Time", "AQI", "Category"]
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.info("✅ No concerning AQI levels in the next 72 hours!")
