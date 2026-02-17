# PEARLS AQI Predictor

**A fully serverless, end-to-end ML pipeline forecasting Air Quality Index (AQI) for Karachi, Pakistan over the next 72 hours.**

> Built with OpenMeteo · MongoDB Atlas · DagsHub · GitHub Actions · FastAPI · Streamlit

> AQI Prediction Dashboard Link: https://aqipredictionmlmodel.streamlit.app/

---

## 📌 Problem Statement

Air pollution is a critical public health issue in Karachi, one of the world's most polluted megacities. There is no accessible, real-time, locally-tailored AQI forecasting tool that provides actionable 72-hour predictions with model explainability.

This project solves that by building an automated, serverless ML pipeline that:
- Ingests live weather and pollutant data hourly
- Trains and updates ML models daily
- Serves 72-hour AQI forecasts through an interactive dashboard

---

## 🏗️ Architecture Overview

```
OpenMeteo API
     │
     ▼
Feature Pipeline (feature_pipeline.py)
     │  Hourly via GitHub Actions
     ▼
MongoDB Atlas (Feature Store)
     │
     ├──────────────────────┐
     ▼                      ▼
Training Pipeline       Predict.py (Inference)
(train.py)                  │
     │                      ▼
DagsHub Model Registry   FastAPI Backend
     │                      │
     └──────────────────────┘
                │
                ▼
       Streamlit Dashboard
```

---

## ⚙️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Data API | OpenMeteo | Free, reliable, no key needed |
| Feature Store | MongoDB Atlas | Flexible schema, free tier |
| Model Registry | DagsHub + MLflow | Git-native ML tracking |
| CI/CD | GitHub Actions | Native to repo, free |
| Backend | FastAPI | Async, production-grade |
| Frontend | Streamlit | Rapid ML dashboards |
| Explainability | SHAP | Best-in-class model interpretability |

---
## 📁 Project Structure

```
karachi-aqi-predictor/
│
├── src/
│   ├── feature_pipeline.py     # Hourly data ingestion & feature engineering
│   ├── backfill.py             # Historical data backfill (90 days)
│   ├── train.py                # Model training & MLflow logging
│   └── Predict.py              # 72-hour recursive AQI forecasting
│
├── app/
│   ├── main.py                 # FastAPI backend
│   └── dashboard.py            # Streamlit frontend
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── SHAP_Explainability.ipynb  # SHAP feature importance analysis
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Runs hourly
│       └── train.yml  # Runs daily
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔄 Pipelines

### Feature Pipeline (Hourly)
Fetches live data from OpenMeteo API and computes:
- **Pollutants:** PM2.5, PM10, NO₂, O₃, CO, SO₂
- **Weather:** Temperature, humidity, wind speed
- **Time features:** Hour, day, month, day_of_week
- **Lag features:** aqi_lag_1, aqi_lag_24
- **Derived features:** aqi_change_1h, aqi_change_24h

### Training Pipeline (Daily)
- Fetches 90-day historical data from MongoDB Atlas
- Trains 3 models: GradientBoosting, RandomForest, Ridge
- Evaluates using RMSE, MAE, R²
- Promotes best model to DagsHub Production registry

---

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| GradientBoosting | Best | Good | Best |
| RandomForest | Baseline | Baseline | Good |
| Ridge | Good| Best | Good |

---

## 🎯 Features

- ✅ Real-time AQI gauge with color-coded categories
- ✅ 72-hour interactive AQI forecast chart
- ✅ Historical AQI trends with rolling averages
- ✅ Pollutant breakdown charts (PM2.5, PM10, NO₂, O₃)
- ✅ Hazardous AQI alerts
- ✅ CSV data download
- ✅ Model performance comparison

---

## 🔬 AQI Categories

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0 – 50 | Good | 🟢 Green |
| 51 – 100 | Moderate | 🟡 Yellow |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 Orange |
| 151 – 200 | Unhealthy | 🔴 Red |
| 201 – 300 | Very Unhealthy | 🟣 Purple |
| 301 – 500 | Hazardous | ☢️ Maroon |

---

## 🤖 CI/CD Automation

```yaml
# Feature pipeline runs every hour
# .github/workflows/feature_pipeline.yml
schedule:
  - cron: '0 * * * *'

# Training pipeline runs daily at 2 AM UTC
# .github/workflows/train.yml
schedule:
  - cron: '0 2 * * *'
```

---

## 📈 SHAP Explainability


Key insights from SHAP analysis:
- `aqi_lag_1` is the strongest predictor (AQI persistence)
- `pm2_5` is the dominant pollutant driver
- `wind_speed` negatively impacts AQI (disperses pollution)
- Time features (hour, day_of_week) capture traffic patterns

---

## ⚠️ Known Limitations

- Forecasts are for Karachi only (Lat: 24.8607, Lon: 67.0011)
- Sudden pollution spikes (industrial accidents, fires) may reduce accuracy
- Dependent on OpenMeteo API availability

---

## 🔮 Future Improvements

- [ ] Integrate real 72-hour weather forecast API (OpenMeteo forecast endpoint)
- [ ] Add uncertainty bands to forecasts (quantile regression)
- [ ] Multi-city AQI prediction
- [ ] LSTM / Transformer models for long-range forecasting
- [ ] Email/SMS alerts for hazardous AQI
- [ ] Mobile application

---

## 👤 Author

**Mahnoor Ishtiaq - Data Science Intern @10PEARLS**
- Project: PEARLS AQI Predictor
- Stack: Python · MongoDB · MLflow · DagsHub · Streamlit · FastAPI

---


