# 🌦️ Smart Weather + ML Predictor

> A real-time weather app that doesn't just show today's weather — it **predicts tomorrow's** using a trained Random Forest ML model. Auto-detects your location, fetches live data, and visualises a 3-day weather trend.

🚀 **Live Demo:** [weather-app-7g95pv7eawidv4z7wxzj2d.streamlit.app](https://weather-app-7g95pv7eawidv4z7wxzj2d.streamlit.app/)

---

## 🧩 Problem Statement

Weather APIs give you the current conditions — but they don't tell you *why* it might rain tomorrow or how conditions compare to historical patterns. This app combines **live API data** with a **machine learning model trained on 10 years of Indian weather data (2013–2024)** to give smarter, localised predictions alongside real-time readings.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Browser                         │
│                  (Streamlit Frontend)                     │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │       app.py            │
              │   (Streamlit App Core)  │
              └──┬──────────┬───────────┘
                 │          │
    ┌────────────▼──┐   ┌───▼──────────────────┐
    │  ipapi.co     │   │   Open-Meteo API      │
    │ (Auto-detect  │   │  (Live weather data:  │
    │  city/country)│   │  temp, humidity,      │
    └───────────────┘   │  wind, pressure)      │
                        └───────────┬───────────┘
                                    │
                         ┌──────────▼──────────┐
                         │     model.py         │
                         │  (Feature engineering│
                         │   + ML inference)    │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │         /models/               │
                    │  rf_rain.pkl    rf_temp.pkl    │
                    │  rf_humidity.pkl  scaler.pkl   │
                    │  (Trained on Kaggle 2013-2024) │
                    └───────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📍 Auto Location | Detects your city, state, country via IP |
| 🌡️ Live Weather | Real-time temp, humidity, wind, pressure |
| 🌧️ Rain Prediction | Rain tomorrow? Yes/No + confidence % |
| 🌡️ Temp Prediction | Predicted temperature for tomorrow |
| 💧 Humidity Prediction | Predicted humidity for tomorrow |
| 📈 3-Day Trend | Line chart of upcoming weather |
| 📊 Model Diagnostics | Feature importance visualisation |

---

## 🧠 ML Models

All models are trained on the [Kaggle Indian Weather Dataset (2013–2024)](https://www.kaggle.com/):

| Model File | Predicts |
|---|---|
| `rf_rain.pkl` | Rain tomorrow (classification) |
| `rf_temp.pkl` | Temperature tomorrow (regression) |
| `rf_humidity.pkl` | Humidity tomorrow (regression) |
| `rf_weather.pkl` | Weather condition (optional) |
| `scaler.pkl` | Feature normalisation |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML:** scikit-learn (Random Forest), pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Weather API:** Open-Meteo (free, no key needed)
- **Location:** ipapi.co

---

## 📁 Project Structure

```
weather-app/
├── app.py                          # Main Streamlit app
├── model.py                        # Model training script
├── requirements.txt
├── features.pkl                    # Saved feature list
├── kaggel_weather_2013_to_2024.csv # Training dataset
├── README.md
└── models/
    ├── rf_rain.pkl
    ├── rf_temp.pkl
    ├── rf_humidity.pkl
    ├── rf_weather.pkl
    ├── enc_weather.pkl
    └── scaler.pkl
```

---

## 🚀 Setup & Run Locally

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/bharadwabansi/weather-app.git
cd weather-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain models
python model.py

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---


**Main Dashboard**
```
┌──────────────────────────────────────────────┐
│  📍 Rajkot, Gujarat, India                   │
│  🌡  32°C  💧 68%  💨 14 km/h               │
│                                               │
│  Tomorrow's Prediction:                       │
│  🌧 Rain: YES (78% confidence)               │
│  🌡 Temp: 30.4°C   💧 Humidity: 72%         │
│                                               │
│  [3-Day Trend Chart]                          │
└──────────────────────────────────────────────┘
```



