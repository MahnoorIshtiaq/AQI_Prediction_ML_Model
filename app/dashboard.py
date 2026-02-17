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
# CONFIGURATION - STREAMLIT-ONLY DEPLOYMENT
# =============================================================================

CITY = "Karachi"

MONGO_URI = os.getenv("MONGODB_URI") or st.secrets.get("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB") or st.secrets.get("MONGODB_DB", "aqi_db")
COLLECTION = os.getenv("MONGODB_COLLECTION") or st.secrets.get("MONGODB_COLLECTION", "historical_data")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI") or st.secrets.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME") or st.secrets.get("MLFLOW_EXPERIMENT_NAME", "AQI_Prediction")

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)

# AQI Category Definitions
AQI_CATEGORIES = [
    {"name": "Good", "min": 0, "max": 50, "color": "#00e400"},
    {"name": "Moderate", "min": 51, "max": 100, "color": "#ffff00"},
    {"name": "Unhealthy (Sensitive)", "min": 101, "max": 150, "color": "#ff7e00"},
    {"name": "Unhealthy", "min": 151, "max": 200, "color": "#ff0000"},
    {"name": "Very Unhealthy", "min": 201, "max": 300, "color": "#8f3f97"},
    {"name": "Hazardous", "min": 301, "max": 500, "color": "#7e0023"}
]

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"{CITY} AQI Dashboard - Production v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #0D2CF5;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #0D2CF5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-success {
        background-color: #37fa66;
        border-left: 4px solid #28a745;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def aqi_category(aqi):
    """Get AQI category name"""
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat["name"]
    return "Unknown"

def aqi_color(aqi):
    """Get AQI category color"""
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat["color"]
    return "#808080"

def get_health_message(aqi):
    """Get health implications for AQI level"""
    if aqi <= 50:
        return "✅ Air quality is satisfactory. Enjoy outdoor activities!"
    elif aqi <= 100:
        return "😊 Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "⚠️ Members of sensitive groups may experience health effects. General public less likely to be affected."
    elif aqi <= 200:
        return "🚨 Everyone may begin to experience health effects. Sensitive groups may experience serious effects."
    elif aqi <= 300:
        return "⛔ Health alert! Everyone may experience serious health effects."
    else:
        return "☢️ Health warnings of emergency conditions! Avoid all outdoor activities!"

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
# DATA LOADING FUNCTIONS (CACHED)
# =============================================================================

@st.cache_data(ttl=300, show_spinner="🔮 Generating 72-hour forecast...")
def get_forecast_data():
    """
    Fetch 72-hour forecast by calling prediction function directly
    """
    try:
        # Call prediction function directly (no HTTP request)
        forecast_data = get_72h_forecast()
        
        if not forecast_data:
            st.error("Forecast generation returned empty data")
            return pd.DataFrame()
        
        df = pd.DataFrame(forecast_data)
        
        if df.empty:
            st.warning("Forecast data is empty")
            return df
        
        # Rename and process
        df.rename(columns={"predicted_aqi": "aqi"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Karachi")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        
        # Validate AQI values
        df = df[df['aqi'] >= 0].copy()
        df['aqi'] = df['aqi'].clip(0, 500)
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["category"] = df["aqi"].apply(aqi_category)
        df["color"] = df["aqi"].apply(aqi_color)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Forecast generation failed: {str(e)}")
        with st.expander("🔍 View detailed error"):
            st.exception(e)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading historical data...")
def get_history(days_back=7):
    """Fetch historical data from MongoDB"""
    if not MONGO_URI:
        st.warning("⚠️ MongoDB URI not configured. Add to Streamlit secrets.")
        return pd.DataFrame()

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        since = datetime.utcnow() - timedelta(days=days_back)

        records = list(
            client[DB_NAME][COLLECTION].find(
                {
                    "timestamp": {"$gte": since},
                    "dataset_type": "online"
                },
                {"_id": 0}
            ).sort("timestamp", 1)
        )

        if not records:
            st.info(f"No historical data found for past {days_back} days")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # Validate data
        required_cols = ['aqi', 'pm2_5', 'pm10', 'temperature', 'humidity', 'wind_speed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns in historical data: {missing_cols}")
        
        return df

    except Exception as e:
        st.error(f"MongoDB Error: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading model metrics...")
def get_latest_model_metrics():
    """Get model comparison from LATEST training run"""
    if not EXPERIMENT_NAME or not MLFLOW_URI:
        return pd.DataFrame(), None

    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            st.warning(f"Experiment '{EXPERIMENT_NAME}' not found")
            return pd.DataFrame(), None

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if runs.empty:
            st.warning("No training runs found")
            return pd.DataFrame(), None

        latest_run = runs.iloc[0]
        
        models = ["GradientBoosting", "RandomForest", "Ridge"]
        model_data = []
        
        for model_name in models:
            rmse = latest_run.get(f"metrics.{model_name}_rmse")
            mae = latest_run.get(f"metrics.{model_name}_mae")
            r2 = latest_run.get(f"metrics.{model_name}_r2")
            
            if pd.notna(rmse) and pd.notna(mae) and pd.notna(r2):
                model_data.append({
                    "Model": model_name,
                    "RMSE": float(rmse),
                    "MAE": float(mae),
                    "R²": float(r2),
                    "Training Date": latest_run["start_time"]
                })
        
        if not model_data:
            st.warning("No valid model metrics found in latest run")
            return pd.DataFrame(), None
        
        df = pd.DataFrame(model_data)
        df = df.sort_values("RMSE").reset_index(drop=True)
        
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.get_latest_versions(
                name="AQI_NextHour_Model",
                stages=["Production"]
            )
            prod_version = versions[0].version if versions else None
        except:
            prod_version = None
        
        return df, prod_version

    except Exception as e:
        st.error(f"MLflow Error: {str(e)}")
        with st.expander("🔍 View detailed error"):
            st.exception(e)
        return pd.DataFrame(), None


# =============================================================================
# CHART CREATION FUNCTIONS
# =============================================================================

def create_aqi_gauge(aqi_value, theme="plotly_white"):
    """Create AQI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={'text': "Current AQI", 'font': {'size': 20}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1},
            'bar': {'color': aqi_color(aqi_value), 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 100], 'color': '#fff9c4'},
                {'range': [100, 150], 'color': '#ffe0b2'},
                {'range': [150, 200], 'color': '#ffcdd2'},
                {'range': [200, 300], 'color': '#e1bee7'},
                {'range': [300, 500], 'color': '#f8bbd0'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        template=theme,
        font={'family': "Arial"}
    )
    
    return fig


def create_forecast_chart(df, theme="plotly_white", show_zones=True):
    """Create 72-hour forecast chart with AQI zones"""
    fig = go.Figure()
    
    # Add AQI zone backgrounds
    if show_zones:
        for cat in AQI_CATEGORIES:
            fig.add_hrect(
                y0=cat["min"], y1=cat["max"],
                fillcolor=cat["color"], opacity=0.15,
                layer="below", line_width=0,
                annotation_text=cat["name"],
                annotation_position="right",
                annotation=dict(font_size=10, font_color="gray")
            )
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi'],
        mode='lines',
        name='Predicted AQI',
        line=dict(color='royalblue', width=3),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>' +
                      'AQI: %{y:.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add markers for peak and valley
    max_idx = df['aqi'].idxmax()
    min_idx = df['aqi'].idxmin()
    
    fig.add_trace(go.Scatter(
        x=[df.loc[max_idx, 'timestamp']],
        y=[df.loc[max_idx, 'aqi']],
        mode='markers+text',
        name='Peak',
        marker=dict(size=12, color='red', symbol='triangle-up'),
        text=[f"Peak: {df.loc[max_idx, 'aqi']:.0f}"],
        textposition="top center",
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[df.loc[min_idx, 'timestamp']],
        y=[df.loc[min_idx, 'aqi']],
        mode='markers+text',
        name='Valley',
        marker=dict(size=12, color='green', symbol='triangle-down'),
        text=[f"Min: {df.loc[min_idx, 'aqi']:.0f}"],
        textposition="bottom center",
        showlegend=False
    ))
    
    fig.update_layout(
        title="72-Hour AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI",
        hovermode='x unified',
        template=theme,
        height=450,
        yaxis=dict(range=[0, max(500, df['aqi'].max() * 1.1)])
    )
    
    return fig


def create_pollutant_chart(history_df, theme="plotly_white"):
    """Create pollutant trends chart"""
    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co']
    available_pollutants = [p for p in pollutants if p in history_df.columns]
    
    if not available_pollutants:
        return None
    
    fig = go.Figure()
    
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    
    for idx, pollutant in enumerate(available_pollutants):
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df[pollutant],
            mode='lines',
            name=pollutant.upper().replace('_', '.'),
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title="Pollutant Concentrations (μg/m³)",
        xaxis_title="Time",
        yaxis_title="Concentration",
        hovermode='x unified',
        template=theme,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_historical_chart_with_rolling(df, theme="plotly_white", show_rolling=True):
    """Create historical chart with optional rolling average"""
    fig = go.Figure()
    
    # Add AQI zones
    for cat in AQI_CATEGORIES:
        fig.add_hrect(
            y0=cat["min"], y1=cat["max"],
            fillcolor=cat["color"], opacity=0.1,
            layer="below", line_width=0
        )
    
    # Raw hourly data (semi-transparent)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi'],
        mode='lines',
        name='Hourly AQI',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))
    
    # Rolling average
    if show_rolling and len(df) > 24:
        df_rolling = df.copy()
        df_rolling['aqi_24h'] = df_rolling['aqi'].rolling(window=24, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=df_rolling['timestamp'],
            y=df_rolling['aqi_24h'],
            mode='lines',
            name='24-Hour Average',
            line=dict(color='darkblue', width=3)
        ))
    
    fig.update_layout(
        title="Historical AQI Trend",
        xaxis_title="Time",
        yaxis_title="AQI",
        hovermode='x unified',
        template=theme,
        height=450
    )
    
    return fig


def create_distribution_chart(df, theme="plotly_white"):
    """Create AQI category distribution chart"""
    if 'aqi' not in df.columns:
        return None
    
    # Categorize AQI
    df_dist = df.copy()
    df_dist['category'] = df_dist['aqi'].apply(aqi_category)
    
    # Count categories
    category_counts = df_dist['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Hours']
    
    # Sort by AQI range
    category_order = [cat["name"] for cat in AQI_CATEGORIES]
    category_counts['Category'] = pd.Categorical(
        category_counts['Category'], 
        categories=category_order, 
        ordered=True
    )
    category_counts = category_counts.sort_values('Category')
    
    # Create color map
    color_map = {cat["name"]: cat["color"] for cat in AQI_CATEGORIES}
    
    fig = px.bar(
        category_counts,
        x='Category',
        y='Hours',
        title='Time Spent in Each AQI Category',
        color='Category',
        color_discrete_map=color_map,
        template=theme
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title=None,
        yaxis_title="Hours"
    )
    
    return fig


def create_model_comparison_chart(models_df, theme="plotly_white"):
    """Create comprehensive model comparison chart"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('RMSE (Lower is Better)', 
                        'MAE (Lower is Better)', 
                        'R² (Higher is Better)'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(
            x=models_df['Model'],
            y=models_df['RMSE'],
            name='RMSE',
            marker_color='indianred',
            text=models_df['RMSE'].round(2),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(
            x=models_df['Model'],
            y=models_df['MAE'],
            name='MAE',
            marker_color='lightsalmon',
            text=models_df['MAE'].round(2),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # R²
    fig.add_trace(
        go.Bar(
            x=models_df['Model'],
            y=models_df['R²'],
            name='R²',
            marker_color='lightseagreen',
            text=models_df['R²'].round(3),
            textposition='outside'
        ),
        row=1, col=3
    )
    
    # Highlight best model
    best_model = models_df.iloc[0]['Model']
    
    fig.update_layout(
        title_text=f"Model Performance Comparison - Best: {best_model}",
        showlegend=False,
        height=450,
        template=theme
    )
    
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=1, col=3)
    
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📊 Dashboard Controls")
    
    # Theme selector
    st.markdown("---")
    st.subheader("🎨 Appearance")
    theme = st.selectbox(
        "Chart Theme",
        ["plotly_white", "plotly", "plotly_dark", "seaborn", "ggplot2"],
        index=0
    )
    
    # Settings
    st.markdown("---")
    st.subheader("⚙️ Display Settings")
    
    show_aqi_zones = st.checkbox("Show AQI zones on charts", value=True)
    show_rolling_avg = st.checkbox("Show 24h rolling average", value=True)
    
    # Historical data days
    st.markdown("---")
    days_back = st.slider("Historical data (days)", 1, 30, 7)
    
    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Info
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info(f"""
    **Location:** {CITY}, Pakistan  
    **Forecast:** 72 hours  
    **Update:** Every every hour  
    **Models:** GB, RF, Ridge  
    **Version:** 2.0 (Streamlit-Only)
    """)
    
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# =============================================================================
# LOAD DATA
# =============================================================================

# CRITICAL CHANGE: Call get_forecast_data() instead of API request

forecast_df = get_forecast_data()
history_df = get_history(days_back=days_back)
models_df, production_version = get_latest_model_metrics()

# Extract current conditions
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
    co = latest.get("co", "N/A")
else:
    current_aqi = 0
    temperature = wind = humidity = "N/A"
    pm25 = pm10 = no2 = o3 = so2 = co = "N/A"

# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================

st.title(f"🌍 {CITY} Air Quality Dashboard")
st.markdown("*Real-time air quality monitoring and 72-hour forecasting powered by ML*")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Forecast",
    "⏳ Historical Analysis", 
    "🤖 Model Performance",
    "📚 Data Explorer"
])


# =============================================================================
# TAB 1: OVERVIEW & FORECAST
# =============================================================================

with tab1:
    st.header("Current Conditions & 72-Hour Forecast")
    
    # Row 1: Current AQI Gauge + Stats
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("🎯 Current AQI")
        if current_aqi > 0:
            fig_gauge = create_aqi_gauge(current_aqi, theme)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Category badge
            category = aqi_category(current_aqi)
            color = aqi_color(current_aqi)
            st.markdown(f"""
            <div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; margin-top:-20px;'>
                <h3 style='color:{"black" if category=="Moderate" else "white"}; margin:0;'>{category}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No current data available")
    
    with col2:
        st.subheader("🌡️ Weather Conditions")
        st.metric("Temperature", f"{temperature}°C" if temperature != "N/A" else "N/A")
        st.metric("Humidity", f"{humidity}%" if humidity != "N/A" else "N/A")
        st.metric("Wind Speed", f"{wind} m/s" if wind != "N/A" else "N/A")
        
    with col3:
        st.subheader("💨 Key Pollutants")
        st.metric("PM2.5", f"{pm25} μg/m³" if pm25 != "N/A" else "N/A")
        st.metric("PM10", f"{pm10} μg/m³" if pm10 != "N/A" else "N/A")
        st.metric("NO₂", f"{no2} μg/m³" if no2 != "N/A" else "N/A")
    
    # Health message
    if current_aqi > 0:
        health_msg = get_health_message(current_aqi)
        
        if current_aqi <= 100:
            st.markdown(f'<div class="alert-box alert-success">{health_msg}</div>', unsafe_allow_html=True)
        elif current_aqi <= 200:
            st.markdown(f'<div class="alert-box alert-warning">{health_msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-box alert-danger">{health_msg}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Forecast Chart
    st.subheader("📈 72-Hour AQI Forecast")
    
    if not forecast_df.empty:
        fig_forecast = create_forecast_chart(forecast_df, theme, show_aqi_zones)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current", f"{current_aqi}")
        col2.metric("24h Avg", f"{forecast_df.head(24)['aqi'].mean():.0f}")
        col3.metric("Max (72h)", f"{forecast_df['aqi'].max():.0f}")
        col4.metric("Min (72h)", f"{forecast_df['aqi'].min():.0f}")
        
        # Alerts
        max_aqi = forecast_df['aqi'].max()
        if max_aqi > 150:
            max_time = forecast_df.loc[forecast_df['aqi'].idxmax(), 'timestamp']
            st.warning(f"**Alert:** AQI expected to reach {max_aqi:.0f} ({aqi_category(max_aqi)}) at {max_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Download forecast
        with st.expander("📥 Download Forecast Data"):
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name=f"aqi_forecast_{CITY}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            st.dataframe(
                forecast_df[['timestamp', 'aqi', 'category']].style.background_gradient(
                    subset=['aqi'], cmap='RdYlGn_r'
                ),
                use_container_width=True,
                height=300
            )
    else:
        st.error("No forecast data available. Please check API connection.")
        if not api_status:
            st.info("Start the API server first:\n```bash\nuvicorn app.main:app --reload\n```")
    
    # Pollutant forecast (if available)
    st.markdown("---")
    st.subheader("Pollutant Trends")
    
    if not history_df.empty:
        fig_pollutants = create_pollutant_chart(history_df, theme)
        if fig_pollutants:
            st.plotly_chart(fig_pollutants, use_container_width=True)
        else:
            st.info("No pollutant data available")

# =============================================================================
# TAB 2: HISTORICAL ANALYSIS
# =============================================================================

with tab2:
    st.header(f"Historical Data Analysis (Past {days_back} Days)")
    
    if not history_df.empty:
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Average AQI", f"{history_df['aqi'].mean():.1f}")
        col2.metric("Max AQI", f"{history_df['aqi'].max():.0f}")
        col3.metric("Min AQI", f"{history_df['aqi'].min():.0f}")
        col4.metric("Std Dev", f"{history_df['aqi'].std():.1f}")
        
        good_hours = (history_df['aqi'] <= 50).sum()
        col5.metric("Good Hours", f"{good_hours} ({good_hours/len(history_df)*100:.0f}%)")
        
        st.markdown("---")
        
        # Historical trend chart
        st.subheader("📈 AQI Trend Over Time")
        fig_hist = create_historical_chart_with_rolling(history_df, theme, show_rolling_avg)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 AQI Distribution")
            fig_dist = create_distribution_chart(history_df, theme)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("📊 Hourly Pattern")
            if 'timestamp' in history_df.columns:
                history_df_hourly = history_df.copy()
                history_df_hourly['hour'] = history_df_hourly['timestamp'].dt.hour
                hourly_avg = history_df_hourly.groupby('hour')['aqi'].mean().reset_index()
                
                fig_hourly = px.line(
                    hourly_avg,
                    x='hour',
                    y='aqi',
                    title='Average AQI by Hour of Day',
                    markers=True,
                    template=theme
                )
                fig_hourly.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Average AQI",
                    height=400
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        if all(col in history_df.columns for col in ['aqi', 'pm2_5', 'temperature', 'humidity', 'wind_speed']):
            corr_data = history_df[['aqi', 'pm2_5', 'pm10', 'temperature', 'humidity', 'wind_speed']].corr()
            
            fig_corr = px.imshow(
                corr_data,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Matrix',
                template=theme
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top correlations with AQI
            aqi_corr = corr_data['aqi'].drop('aqi').sort_values(ascending=False)
            st.write("**Top Correlations with AQI:**")
            for feature, corr_val in aqi_corr.items():
                st.write(f"- {feature}: {corr_val:.3f}")
        
    else:
        st.warning(f"No historical data available for the past {days_back} days")
        st.info("Historical data is stored from the feature pipeline. Ensure it has run at least once.")

# =============================================================================
# TAB 3: MODEL PERFORMANCE
# =============================================================================

with tab3:
    st.header("🤖 Model Performance Analysis")
    
    if not models_df.empty:
        # Training metadata
        training_date = models_df.iloc[0]['Training Date']
        st.info(f"📅 Models trained on: {training_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Best model highlight
        best_model = models_df.iloc[0]['Model']
        best_rmse = models_df.iloc[0]['RMSE']
        best_r2 = models_df.iloc[0]['R²']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🏆 Best Model", best_model)
        col2.metric("RMSE", f"{best_rmse:.2f}")
        col3.metric("R² Score", f"{best_r2:.3f}")
        
        # Production model info
        if production_version:
            st.success(f"✅ Model version {production_version} currently in Production")
        
        st.markdown("---")
        
        # Comparison chart
        st.subheader("📊 Model Comparison")
        fig_models = create_model_comparison_chart(models_df, theme)
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Performance improvement
        if len(models_df) > 1:
            second_rmse = models_df.iloc[1]['RMSE']
            improvement = ((second_rmse - best_rmse) / second_rmse) * 100
            st.success(f"🎯 Best model is **{improvement:.1f}%** better than second-best model")
        
        st.markdown("---")
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics Table")
        
        styled_df = models_df[['Model', 'RMSE', 'MAE', 'R²']].style.highlight_min(
            subset=['RMSE', 'MAE'], 
            color='lightgreen'
        ).highlight_max(
            subset=['R²'], 
            color='lightgreen'
        ).format({
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}',
            'R²': '{:.3f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
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
        
        # Model interpretation
        st.markdown("---")
        st.subheader("📖 Model Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**What do these metrics mean?**")
            st.markdown("""
            - **RMSE (Root Mean Square Error):** Average prediction error in AQI units. Lower is better.
            - **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual AQI. Lower is better.
            - **R² Score:** How well the model explains variance in AQI (0-1 scale). Higher is better.
            """)
        
        with col2:
            st.markdown("**Model Characteristics:**")
            st.markdown("""
            - **GradientBoosting:** Ensemble method, typically best accuracy
            - **RandomForest:** Robust, handles non-linearity well
            - **Ridge:** Linear model, fast and interpretable
            """)
        
    else:
        st.warning("⚠️ No model performance data available")
        st.info("""
        💡 **To see model metrics:**
        1. Ensure training has completed: `python src/train.py`
        2. Check that MLflow is accessible
        3. Verify models were logged with name prefixes (e.g., 'GradientBoosting_rmse')
        """)
        
        if MLFLOW_URI:
            st.code(f"MLflow URI: {MLFLOW_URI}", language="text")

# =============================================================================
# TAB 4: DATA EXPLORER
# =============================================================================

with tab4:
    st.header("📚 Raw Data Explorer")
    
    # Historical data
    st.subheader(f"Historical Data (Past {days_back} Days)")
    
    if not history_df.empty:
        st.write(f"**Total Records:** {len(history_df):,}")
        st.write(f"**Date Range:** {history_df['timestamp'].min()} to {history_df['timestamp'].max()}")
        st.write(f"**Columns:** {', '.join(history_df.columns)}")
        
        # Show dataframe
        st.dataframe(history_df, use_container_width=True, height=400)
        
        # Download
        csv_hist = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Historical Data CSV",
            data=csv_hist,
            file_name=f"historical_data_{CITY}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No historical data to display")
    
    st.markdown("---")
    
    # Forecast data
    st.subheader("Forecast Data (Next 72 Hours)")
    
    if not forecast_df.empty:
        st.write(f"**Total Predictions:** {len(forecast_df)}")
        st.write(f"**Forecast Range:** {forecast_df['timestamp'].min()} to {forecast_df['timestamp'].max()}")
        
        st.dataframe(forecast_df, use_container_width=True, height=400)
    else:
        st.info("No forecast data to display")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>Karachi AQI Dashboard v2.0</b> | Production Ready</p>
    <p>Data Sources: MongoDB Atlas | Models: DagsHub/MLflow | Framework: Streamlit + Plotly</p>
    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PKT</p>
</div>
""", unsafe_allow_html=True)
