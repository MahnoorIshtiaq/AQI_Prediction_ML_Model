# app/dashboard.py

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from src.Predict import get_72h_forecast

load_dotenv()

SHAP_DIR = Path(__file__).parent.parent / "figures" / "shap"

# =============================================================================
# CONFIGURATION
# =============================================================================

CITY = "Karachi"

MONGO_URI       = os.getenv("MONGODB_URI")       or st.secrets.get("MONGODB_URI")
DB_NAME         = os.getenv("MONGODB_DB")         or st.secrets.get("MONGODB_DB", "aqi_db")
COLLECTION      = os.getenv("MONGODB_COLLECTION") or st.secrets.get("MONGODB_COLLECTION", "historical_data")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI") or st.secrets.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME") or st.secrets.get("MLFLOW_EXPERIMENT_NAME", "AQI_Prediction")

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)

# AQI Category Definitions
AQI_CATEGORIES = [
    {"name": "Good",                   "min": 0,   "max": 50,  "color": "#00e400"},
    {"name": "Moderate",               "min": 51,  "max": 100, "color": "#ffff00"},
    {"name": "Unhealthy (Sensitive)",  "min": 101, "max": 150, "color": "#ff7e00"},
    {"name": "Unhealthy",              "min": 151, "max": 200, "color": "#ff0000"},
    {"name": "Very Unhealthy",         "min": 201, "max": 300, "color": "#8f3f97"},
    {"name": "Hazardous",              "min": 301, "max": 500, "color": "#7e0023"},
]

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"{CITY} AQI Dashboard - Production v2.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .alert-box { padding: 15px; border-radius: 10px; margin: 10px 0; }
    .alert-success { background-color: #d4edda; border-left: 4px solid #28a745; }
    .alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .alert-danger  { background-color: #f8d7da; border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def aqi_category(aqi):
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat["name"]
    return "Unknown"

def aqi_color(aqi):
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat["color"]
    return "#808080"

def get_health_message(aqi):
    if aqi <= 50:
        return "✅ Air quality is satisfactory. Enjoy outdoor activities!"
    elif aqi <= 100:
        return "😊 Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "⚠️ Sensitive groups may experience health effects. General public less likely to be affected."
    elif aqi <= 200:
        return "🚨 Everyone may begin to experience health effects. Sensitive groups may experience serious effects."
    elif aqi <= 300:
        return "⛔ Health alert! Everyone may experience serious health effects."
    else:
        return "☢️ Health emergency conditions! Avoid all outdoor activities!"

def shap_image(filename: str):
    """
    Safely display a SHAP figure using an absolute path.
    Shows a friendly warning instead of crashing when the file is missing.
    """
    path = SHAP_DIR / filename
    if path.exists():
        st.image(str(path))
    else:
        st.warning(
            f"⚠️ SHAP figure not found: `{path}`\n\n"
            "Run `notebooks/SHAP_Explainability.ipynb` locally, commit the generated "
            "`figures/shap/` folder to your repository, then redeploy."
        )


# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================

@st.cache_data(ttl=300, show_spinner="🔮 Generating 72-hour forecast...")
def get_forecast_data():
    """Call prediction function directly — no HTTP request needed."""
    try:
        forecast_data = get_72h_forecast()

        if not forecast_data:
            st.error("Forecast generation returned empty data.")
            return pd.DataFrame()

        df = pd.DataFrame(forecast_data)
        df.rename(columns={"predicted_aqi": "aqi"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Karachi")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df = df[df["aqi"] >= 0].copy()
        df["aqi"] = df["aqi"].clip(0, 500)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["category"] = df["aqi"].apply(aqi_category)
        df["color"]    = df["aqi"].apply(aqi_color)
        return df

    except Exception as e:
        st.error(f"❌ Forecast generation failed: {e}")
        with st.expander("🔍 View detailed error"):
            st.exception(e)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading historical data...")
def get_history(days_back=7):
    if not MONGO_URI:
        st.warning("⚠️ MongoDB URI not configured.")
        return pd.DataFrame()
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        since  = datetime.utcnow() - timedelta(days=days_back)
        records = list(
            client[DB_NAME][COLLECTION]
            .find({"timestamp": {"$gte": since}, "dataset_type": "online"}, {"_id": 0})
            .sort("timestamp", 1)
        )
        if not records:
            st.info(f"No historical data found for past {days_back} days.")
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception as e:
        st.error(f"MongoDB Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading model metrics...")
def get_latest_model_metrics():
    if not EXPERIMENT_NAME or not MLFLOW_URI:
        return pd.DataFrame(), None
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            st.warning(f"Experiment '{EXPERIMENT_NAME}' not found.")
            return pd.DataFrame(), None

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            return pd.DataFrame(), None

        latest_run = runs.iloc[0]
        model_data = []
        for model_name in ["GradientBoosting", "RandomForest", "Ridge"]:
            rmse = latest_run.get(f"metrics.{model_name}_rmse")
            mae  = latest_run.get(f"metrics.{model_name}_mae")
            r2   = latest_run.get(f"metrics.{model_name}_r2")
            if pd.notna(rmse) and pd.notna(mae) and pd.notna(r2):
                model_data.append({
                    "Model": model_name,
                    "RMSE": float(rmse),
                    "MAE":  float(mae),
                    "R²":   float(r2),
                    "Training Date": latest_run["start_time"],
                })

        if not model_data:
            return pd.DataFrame(), None

        df = pd.DataFrame(model_data).sort_values("RMSE").reset_index(drop=True)

        client_mlflow = mlflow.tracking.MlflowClient()
        try:
            versions = client_mlflow.get_latest_versions("AQI_NextHour_Model", stages=["Production"])
            prod_version = versions[0].version if versions else None
        except Exception:
            prod_version = None

        return df, prod_version

    except Exception as e:
        st.error(f"MLflow Error: {e}")
        return pd.DataFrame(), None


# =============================================================================
# CHART HELPERS
# =============================================================================

def create_aqi_gauge(aqi_value, theme="plotly_white"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={"text": "Current AQI", "font": {"size": 20}},
        number={"font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": aqi_color(aqi_value), "thickness": 0.75},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0,   50],  "color": "#e8f5e9"},
                {"range": [50,  100], "color": "#fff9c4"},
                {"range": [100, 150], "color": "#ffe0b2"},
                {"range": [150, 200], "color": "#ffcdd2"},
                {"range": [200, 300], "color": "#e1bee7"},
                {"range": [300, 500], "color": "#f8bbd0"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": aqi_value},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20), template=theme)
    return fig


def create_forecast_chart(df, theme="plotly_white", show_zones=True):
    fig = go.Figure()
    if show_zones:
        for cat in AQI_CATEGORIES:
            fig.add_hrect(
                y0=cat["min"], y1=cat["max"],
                fillcolor=cat["color"], opacity=0.15,
                layer="below", line_width=0,
                annotation_text=cat["name"],
                annotation_position="right",
                annotation=dict(font_size=10, font_color="gray"),
            )
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"], mode="lines",
        name="Predicted AQI", line=dict(color="royalblue", width=3),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>",
    ))
    max_idx, min_idx = df["aqi"].idxmax(), df["aqi"].idxmin()
    fig.add_trace(go.Scatter(
        x=[df.loc[max_idx, "timestamp"]], y=[df.loc[max_idx, "aqi"]],
        mode="markers+text", marker=dict(size=12, color="red", symbol="triangle-up"),
        text=[f"Peak: {df.loc[max_idx, 'aqi']:.0f}"], textposition="top center", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[df.loc[min_idx, "timestamp"]], y=[df.loc[min_idx, "aqi"]],
        mode="markers+text", marker=dict(size=12, color="green", symbol="triangle-down"),
        text=[f"Min: {df.loc[min_idx, 'aqi']:.0f}"], textposition="bottom center", showlegend=False,
    ))
    fig.update_layout(
        title="72-Hour AQI Forecast", xaxis_title="Time", yaxis_title="AQI",
        hovermode="x unified", template=theme, height=450,
        yaxis=dict(range=[0, max(500, df["aqi"].max() * 1.1)]),
    )
    return fig


def create_pollutant_chart(history_df, theme="plotly_white"):
    pollutants = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
    available  = [p for p in pollutants if p in history_df.columns]
    if not available:
        return None
    colors = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
    fig = go.Figure()
    for idx, p in enumerate(available):
        fig.add_trace(go.Scatter(
            x=history_df["timestamp"], y=history_df[p], mode="lines",
            name=p.upper().replace("_", "."), line=dict(color=colors[idx % len(colors)], width=2),
        ))
    fig.update_layout(
        title="Pollutant Concentrations (μg/m³)", xaxis_title="Time",
        yaxis_title="Concentration", hovermode="x unified", template=theme, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_historical_chart_with_rolling(df, theme="plotly_white", show_rolling=True):
    fig = go.Figure()
    for cat in AQI_CATEGORIES:
        fig.add_hrect(y0=cat["min"], y1=cat["max"], fillcolor=cat["color"],
                      opacity=0.1, layer="below", line_width=0)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"], mode="lines", name="Hourly AQI",
        line=dict(color="lightblue", width=1), opacity=0.5,
    ))
    if show_rolling and len(df) > 24:
        df2 = df.copy()
        df2["aqi_24h"] = df2["aqi"].rolling(24, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df2["timestamp"], y=df2["aqi_24h"], mode="lines",
            name="24-Hour Average", line=dict(color="darkblue", width=3),
        ))
    fig.update_layout(
        title="Historical AQI Trend", xaxis_title="Time", yaxis_title="AQI",
        hovermode="x unified", template=theme, height=450,
    )
    return fig


def create_distribution_chart(df, theme="plotly_white"):
    if "aqi" not in df.columns:
        return None
    df2 = df.copy()
    df2["category"] = df2["aqi"].apply(aqi_category)
    counts = df2["category"].value_counts().reset_index()
    counts.columns = ["Category", "Hours"]
    order = [c["name"] for c in AQI_CATEGORIES]
    counts["Category"] = pd.Categorical(counts["Category"], categories=order, ordered=True)
    counts = counts.sort_values("Category")
    color_map = {c["name"]: c["color"] for c in AQI_CATEGORIES}
    fig = px.bar(counts, x="Category", y="Hours", color="Category",
                 color_discrete_map=color_map,
                 title="Time Spent in Each AQI Category", template=theme)
    fig.update_layout(showlegend=False, height=400, xaxis_title=None)
    return fig


def create_model_comparison_chart(models_df, theme="plotly_white"):
    best_model = models_df.iloc[0]["Model"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("RMSE (Lower is Better)", "MAE (Lower is Better)", "R² (Higher is Better)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )
    for col_n, (col_name, color) in enumerate(
        [("RMSE", "indianred"), ("MAE", "lightsalmon"), ("R²", "lightseagreen")], start=1
    ):
        fig.add_trace(go.Bar(
            x=models_df["Model"], y=models_df[col_name], name=col_name,
            marker_color=color,
            text=models_df[col_name].round(3 if col_name == "R²" else 2),
            textposition="outside",
        ), row=1, col=col_n)
    fig.update_layout(
        title_text=f"Model Performance Comparison — Best: {best_model}",
        showlegend=False, height=450, template=theme,
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📊 Dashboard Controls")
    st.markdown("---")
    st.subheader("🎨 Appearance")
    theme = st.selectbox("Chart Theme",
                         ["plotly_white", "plotly", "plotly_dark", "seaborn", "ggplot2"])
    st.markdown("---")
    st.subheader("⚙️ Display Settings")
    show_aqi_zones  = st.checkbox("Show AQI zones on charts", value=True)
    show_rolling_avg = st.checkbox("Show 24h rolling average", value=True)
    st.markdown("---")
    days_back = st.slider("Historical data (days)", 1, 30, 7)
    st.markdown("---")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info(f"""
    **Location:** {CITY}, Pakistan
    **Forecast:** 72 hours
    **Update:** Every hour
    **Models:** GB, RF, Ridge
    **Version:** 2.0
    """)
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# =============================================================================
# LOAD DATA
# =============================================================================

forecast_df                    = get_forecast_data()
history_df                     = get_history(days_back=days_back)
models_df, production_version  = get_latest_model_metrics()

if not history_df.empty and "aqi" in history_df.columns:
    latest      = history_df.iloc[-1]
    current_aqi = int(latest.get("aqi", 0))
    temperature = latest.get("temperature", "N/A")
    wind        = latest.get("wind_speed", "N/A")
    humidity    = latest.get("humidity", "N/A")
    pm25        = latest.get("pm2_5", "N/A")
    pm10        = latest.get("pm10", "N/A")
    no2         = latest.get("no2", "N/A")
else:
    current_aqi = 0
    temperature = wind = humidity = pm25 = pm10 = no2 = "N/A"


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title(f"🌍 {CITY} Air Quality Dashboard")
st.markdown("*Real-time air quality monitoring and 72-hour forecasting powered by ML*")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Forecast",
    "⏳ Historical Analysis",
    "🤖 Model Performance",
    "📚 Data Explorer",
])


# ─── TAB 1: OVERVIEW & FORECAST ──────────────────────────────────────────────

with tab1:
    st.header("Current Conditions & 72-Hour Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎯 Current AQI")
        if current_aqi > 0:
            st.plotly_chart(create_aqi_gauge(current_aqi, theme), use_container_width=True)
            category = aqi_category(current_aqi)
            color    = aqi_color(current_aqi)
            txt_col  = "black" if category == "Moderate" else "white"
            st.markdown(
                f"<div style='background-color:{color};padding:10px;border-radius:8px;"
                f"text-align:center;margin-top:-20px;'>"
                f"<h3 style='color:{txt_col};margin:0;'>{category}</h3></div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No current data available.")

    with col2:
        st.subheader("🌡️ Weather Conditions")
        st.metric("Temperature", f"{temperature}°C" if temperature != "N/A" else "N/A")
        st.metric("Humidity",    f"{humidity}%"  if humidity    != "N/A" else "N/A")
        st.metric("Wind Speed",  f"{wind} m/s"   if wind        != "N/A" else "N/A")

    with col3:
        st.subheader("💨 Key Pollutants")
        st.metric("PM2.5", f"{pm25} μg/m³" if pm25 != "N/A" else "N/A")
        st.metric("PM10",  f"{pm10} μg/m³" if pm10 != "N/A" else "N/A")
        st.metric("NO₂",   f"{no2} μg/m³"  if no2  != "N/A" else "N/A")

    if current_aqi > 0:
        msg = get_health_message(current_aqi)
        cls = "alert-success" if current_aqi <= 100 else ("alert-warning" if current_aqi <= 200 else "alert-danger")
        st.markdown(f'<div class="alert-box {cls}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 72-Hour AQI Forecast")

    if not forecast_df.empty:
        st.plotly_chart(create_forecast_chart(forecast_df, theme, show_aqi_zones), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current",    str(current_aqi))
        c2.metric("24h Avg",    f"{forecast_df.head(24)['aqi'].mean():.0f}")
        c3.metric("Max (72h)",  f"{forecast_df['aqi'].max():.0f}")
        c4.metric("Min (72h)",  f"{forecast_df['aqi'].min():.0f}")

        max_aqi = forecast_df["aqi"].max()
        if max_aqi > 150:
            max_time = forecast_df.loc[forecast_df["aqi"].idxmax(), "timestamp"]
            st.warning(f"**Alert:** AQI expected to reach {max_aqi:.0f} "
                       f"({aqi_category(max_aqi)}) at {max_time.strftime('%Y-%m-%d %H:%M')}")

        with st.expander("📥 Download Forecast Data"):
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download as CSV", csv,
                               f"aqi_forecast_{CITY}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               "text/csv")
            st.dataframe(
                forecast_df[["timestamp", "aqi", "category"]]
                .style.background_gradient(subset=["aqi"], cmap="RdYlGn_r"),
                use_container_width=True, height=300,
            )
    else:
        st.error("No forecast data available. Check your data pipeline and model registry.")

    st.markdown("---")
    st.subheader("💨 Pollutant Trends")
    if not history_df.empty:
        fig_p = create_pollutant_chart(history_df, theme)
        st.plotly_chart(fig_p, use_container_width=True) if fig_p else st.info("No pollutant data available.")
    else:
        st.info("No historical data loaded.")


# ─── TAB 2: HISTORICAL ANALYSIS ──────────────────────────────────────────────

with tab2:
    st.header(f"Historical Data Analysis (Past {days_back} Days)")

    if not history_df.empty:
        st.subheader("📊 Summary Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Average AQI", f"{history_df['aqi'].mean():.1f}")
        c2.metric("Max AQI",     f"{history_df['aqi'].max():.0f}")
        c3.metric("Min AQI",     f"{history_df['aqi'].min():.0f}")
        c4.metric("Std Dev",     f"{history_df['aqi'].std():.1f}")
        good = (history_df["aqi"] <= 50).sum()
        c5.metric("Good Hours",  f"{good} ({good/len(history_df)*100:.0f}%)")

        st.markdown("---")
        st.subheader("📈 AQI Trend Over Time")
        st.plotly_chart(
            create_historical_chart_with_rolling(history_df, theme, show_rolling_avg),
            use_container_width=True,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 AQI Distribution")
            fig_d = create_distribution_chart(history_df, theme)
            if fig_d:
                st.plotly_chart(fig_d, use_container_width=True)

        with col2:
            st.subheader("📊 Hourly Pattern")
            if "timestamp" in history_df.columns:
                h_df = history_df.copy()
                h_df["hour"] = h_df["timestamp"].dt.hour
                hourly_avg = h_df.groupby("hour")["aqi"].mean().reset_index()
                fig_h = px.line(hourly_avg, x="hour", y="aqi",
                                title="Average AQI by Hour of Day",
                                markers=True, template=theme)
                fig_h.update_layout(xaxis_title="Hour of Day", yaxis_title="Average AQI", height=400)
                st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("---")
        st.subheader("🔗 Correlation Analysis")
        corr_cols = ["aqi", "pm2_5", "pm10", "temperature", "humidity", "wind_speed"]
        available_corr = [c for c in corr_cols if c in history_df.columns]
        if len(available_corr) >= 3:
            corr_data = history_df[available_corr].corr()
            fig_c = px.imshow(corr_data, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdBu_r",
                              title="Feature Correlation Matrix", template=theme)
            fig_c.update_layout(height=500)
            st.plotly_chart(fig_c, use_container_width=True)
            aqi_corr = corr_data["aqi"].drop("aqi").sort_values(ascending=False)
            st.write("**Top Correlations with AQI:**")
            for feat, val in aqi_corr.items():
                st.write(f"- {feat}: {val:.3f}")
    else:
        st.warning(f"No historical data for the past {days_back} days.")
        st.info("Ensure the feature pipeline has run at least once.")


# ─── TAB 3: MODEL PERFORMANCE ────────────────────────────────────────────────

with tab3:
    st.header("🤖 Model Performance Analysis")

    if not models_df.empty:
        training_date = models_df.iloc[0]["Training Date"]
        st.info(f"📅 Models trained on: {training_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        best_model = models_df.iloc[0]["Model"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Model", best_model)
        c2.metric("RMSE",          f"{models_df.iloc[0]['RMSE']:.2f}")
        c3.metric("R² Score",      f"{models_df.iloc[0]['R²']:.3f}")

        if production_version:
            st.success(f"✅ Model version {production_version} currently in Production")

        st.markdown("---")
        st.subheader("📊 Model Comparison")
        st.plotly_chart(create_model_comparison_chart(models_df, theme), use_container_width=True)

        if len(models_df) > 1:
            improvement = ((models_df.iloc[1]["RMSE"] - models_df.iloc[0]["RMSE"])
                           / models_df.iloc[1]["RMSE"] * 100)
            st.success(f"🎯 Best model is **{improvement:.1f}%** better than second-best model")

        st.markdown("---")
        st.subheader("📋 Detailed Metrics Table")
        styled = (
            models_df[["Model", "RMSE", "MAE", "R²"]]
            .style
            .highlight_min(subset=["RMSE", "MAE"], color="lightgreen")
            .highlight_max(subset=["R²"],           color="lightgreen")
            .format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}"})
        )
        st.dataframe(styled, use_container_width=True)

        # ── SHAP SECTION ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Model Explainability (SHAP)")
        st.caption(
            "SHAP figures are generated by running `notebooks/SHAP_Explainability.ipynb`. "
            "The output images must be committed to `figures/shap/` in your repository."
        )

        tab_shap1, tab_shap2, tab_shap3 = st.tabs([
            "📊 Feature Importance",
            "🎯 Example Explanations",
            "🧪 Feature Interactions",
        ])

        with tab_shap1:
            st.markdown("### Global Feature Importance")
            st.markdown(
                "The bar chart shows the mean absolute SHAP value per feature — "
                "how much each feature moves the AQI prediction on average. "
                "The beeswarm plot shows the direction: red = high feature value, "
                "blue = low feature value."
            )
            shap_image("01_feature_importance_bar.png")
            shap_image("02_summary_beeswarm.png")

        with tab_shap2:
            st.markdown("### Why These Specific Predictions?")
            st.markdown(
                "Force plots show which features pushed a single prediction **up** (red) "
                "or **down** (blue) from the model's base (average) value."
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**High AQI Example**")
                shap_image("04_force_plot_high_aqi.png")
                st.markdown("**High AQI — Cumulative Waterfall**")
                shap_image("06_waterfall_high_aqi.png")
            with col2:
                st.markdown("**Low AQI Example**")
                shap_image("05_force_plot_low_aqi.png")
                st.markdown("**Low AQI — Cumulative Waterfall**")
                shap_image("07_waterfall_low_aqi.png")

        with tab_shap3:
            st.markdown("### Feature Interactions")
            st.markdown(
                "The dependence plot shows how the top two features interact: "
                "the colour of each dot represents the second feature's value, "
                "revealing how it modulates the primary feature's effect on AQI."
            )
            shap_image("08_feature_interaction.png")
            st.markdown("### Prediction Decision Paths (50 samples)")
            shap_image("09_decision_plot.png")

        st.markdown("---")
        st.subheader("📖 Model Interpretation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**What do these metrics mean?**")
            st.markdown("""
- **RMSE:** Average prediction error in AQI units. Lower is better.
- **MAE:** Average absolute difference. Lower is better.
- **R²:** Proportion of variance explained (0–1). Higher is better.
            """)
        with col2:
            st.markdown("**Model Characteristics:**")
            st.markdown("""
- **GradientBoosting:** Ensemble boosting, typically best accuracy.
- **RandomForest:** Bagging ensemble, robust to noise.
- **Ridge:** Linear baseline, fast and interpretable.
            """)

    else:
        st.warning("⚠️ No model performance data available.")
        st.info("""
**To see model metrics:**
1. Run training: `python src/train.py`
2. Verify MLflow is accessible
3. Ensure metrics are logged with the correct prefix (e.g. `GradientBoosting_rmse`)
        """)
        if MLFLOW_URI:
            st.code(f"MLflow URI: {MLFLOW_URI}")


# ─── TAB 4: DATA EXPLORER ────────────────────────────────────────────────────

with tab4:
    st.header("📚 Raw Data Explorer")

    st.subheader(f"Historical Data (Past {days_back} Days)")
    if not history_df.empty:
        st.write(f"**Total Records:** {len(history_df):,}")
        st.write(f"**Date Range:** {history_df['timestamp'].min()} → {history_df['timestamp'].max()}")
        st.write(f"**Columns:** {', '.join(history_df.columns)}")
        st.dataframe(history_df, use_container_width=True, height=400)
        csv_h = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Historical Data CSV", csv_h,
                           f"historical_{CITY}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    else:
        st.info("No historical data to display.")

    st.markdown("---")
    st.subheader("Forecast Data (Next 72 Hours)")
    if not forecast_df.empty:
        st.write(f"**Total Predictions:** {len(forecast_df)}")
        st.write(f"**Forecast Range:** {forecast_df['timestamp'].min()} → {forecast_df['timestamp'].max()}")
        st.dataframe(forecast_df, use_container_width=True, height=400)
    else:
        st.info("No forecast data to display.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:gray;padding:20px;'>
    <p><b>Karachi AQI Dashboard v2.0</b> | Production Ready</p>
    <p>Data: MongoDB Atlas &nbsp;|&nbsp; Models: DagsHub/MLflow &nbsp;|&nbsp; Stack: Streamlit + Plotly</p>
    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PKT</p>
</div>
""", unsafe_allow_html=True)
