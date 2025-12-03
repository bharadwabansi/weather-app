'''# app.py

import streamlit as st
import requests
import pandas as pd
import pickle
import numpy as np

# ---------------------------
# Load ML Models
# ---------------------------
scaler = pickle.load(open("models/scaler.pkl", "rb"))
rf_rain = pickle.load(open("models/rf_rain.pkl", "rb"))
rf_temp = pickle.load(open("models/rf_temp.pkl", "rb"))
rf_humidity = pickle.load(open("models/rf_humidity.pkl", "rb"))
rf_weather = pickle.load(open("models/rf_weather.pkl", "rb"))
enc_weather = pickle.load(open("models/enc_weather.pkl", "rb"))


# ---------------------------
# Get User Location Automatically
# ---------------------------
def get_location():
    try:
        loc = requests.get("https://ipapi.co/json").json()
        return loc["latitude"], loc["longitude"], loc["city"]
    except:
        return 20.5937, 78.9629, "India"


# ---------------------------
# Fetch Real-Time Weather From Open-Meteo
# ---------------------------
def fetch_realtime_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true&"
        f"hourly=temperature_2m,relativehumidity_2m,windspeed_10m,"
        f"precipitation,precipitation_probability,weathercode,pressure_msl"
    )
    data = requests.get(url).json()

    c = data["current_weather"]
    hour = data["hourly"]

    # MATCH training features
    df = pd.DataFrame({
        "temp": [c["temperature"]],
        "humidity": [hour["relativehumidity_2m"][0]],
        "windspeed": [c["windspeed"]],
        "sealevelpressure": [hour["pressure_msl"][0]],
        "precip": [hour["precipitation"][0]],
        "precipprob": [hour["precipitation_probability"][0]],
        "raintoday": [1 if hour["precipitation"][0] > 0 else 0]
    })

    return df


# ---------------------------
# ML Prediction Function
# ---------------------------
def make_predictions(df):
    df_scaled = scaler.transform(df)

    rain = rf_rain.predict(df_scaled)[0]
    temp = rf_temp.predict(df_scaled)[0]
    hum = rf_humidity.predict(df_scaled)[0]
    weather = rf_weather.predict(df_scaled)[0]
    weather_label = enc_weather.inverse_transform([weather])[0]

    return rain, temp, hum, weather_label


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🌦 Real-Time Weather + ML Prediction App")
st.write("Using Random Forest + Indian Weather Dataset + Live API")

lat, lon, city = get_location()
st.info(f"📍 Detected Location: **{city}** ({lat}, {lon})")

df = fetch_realtime_weather(lat, lon)
st.subheader("Current Weather Data (Real-Time)")
st.write(df)

if st.button("Predict Tomorrow's Weather"):
    rain, temp, hum, weather = make_predictions(df)

    st.success("Prediction Completed!")

    st.metric("🌧 Will it Rain Tomorrow?", "Yes" if rain == 1 else "No")
    st.metric("🌡 Predicted Temperature Tomorrow", f"{temp:.2f} °C")
    st.metric("💧 Predicted Humidity Tomorrow", f"{hum:.2f} %")
    st.metric("🌤 Weather Condition Tomorrow", weather)
'''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Smart Weather + ML (Vertical UI)",
    layout="centered",
)

# ---------------- Styling ----------------
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    color: #111;
}
.big-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}
.section-title {
    font-size: 28px;
    font-weight: 600;
    margin-top: 25px;
}
.card {
    padding: 12px;
    background: #f1f4f9;
    border-radius: 12px;
    border: 1px solid #d0d7e2;
    margin-bottom: 20px;
}
.metric-box {
    padding: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Models ----------------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Missing model file: {path}")
        st.stop()
    return pickle.load(open(path, "rb"))

MODELS_DIR = "models"
scaler = load_model(f"{MODELS_DIR}/scaler.pkl")
rf_rain = load_model(f"{MODELS_DIR}/rf_rain.pkl")
rf_temp = load_model(f"{MODELS_DIR}/rf_temp.pkl")
rf_humidity = load_model(f"{MODELS_DIR}/rf_humidity.pkl")

# Optional
rf_weather = None
enc_weather = None
if os.path.exists(f"{MODELS_DIR}/rf_weather.pkl"):
    rf_weather = load_model(f"{MODELS_DIR}/rf_weather.pkl")
    enc_weather = load_model(f"{MODELS_DIR}/enc_weather.pkl")

# Feature names (training order)
FEATURES = list(scaler.feature_names_in_)

# ---------------- Helpers ----------------
def detect_location():
    try:
        r = requests.get("https://ipapi.co/json", timeout=5)
        j = r.json()

        city = j.get("city")
        region = j.get("region")
        country = j.get("country_name")

        # Build clean label
        location_parts = [city, region, country]
        location = ", ".join([p for p in location_parts if p and p.strip()])

        # fallback
        if not location:
            location = "India"

        lat = float(j.get("latitude", 20.5937))
        lon = float(j.get("longitude", 78.9629))

        return lat, lon, location

    except:
        return 20.5937, 78.9629, "India"


def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current_weather=true"
        "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,pressure_msl,"
        "precipitation,precipitation_probability,weathercode&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    data = r.json()

    cw = data["current_weather"]
    hr = data["hourly"]

    def hv(k):
        return hr.get(k, [0])[0]

    return {
        "temperature": cw.get("temperature", hv("temperature_2m")),
        "humidity": hv("relativehumidity_2m"),
        "windspeed": cw.get("windspeed", hv("windspeed_10m")),
        "pressure": hv("pressure_msl"),
        "precip": hv("precipitation"),
        "precipprob": hv("precipitation_probability"),
        "weathercode": cw.get("weathercode")
    }

RAIN_CODES = {51,53,55,61,63,65,80,81,82,95,96,99}

def to_feature_row(api, features):
    row = []
    for f in features:
        f_low = f.lower()
        if f_low == "raintoday":
            val = 1 if api["precip"] > 0 or api["weathercode"] in RAIN_CODES else 0
        else:
            val = api.get(f_low, 0)
        row.append(val)
    return pd.DataFrame([row], columns=features)

# ---------------- UI ----------------
st.markdown("<div class='big-title'>🌦️ Smart Weather + ML</div>", unsafe_allow_html=True)

lat, lon, city = detect_location()
st.markdown(f"### 📍 Location: **{city}**")

# 1️⃣ CURRENT WEATHER (Large Card)
st.markdown("<div class='section-title'>Current Weather</div>", unsafe_allow_html=True)

api = fetch_weather(lat, lon)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Temperature (°C)", api["temperature"])
    st.metric("Humidity (%)", api["humidity"])
    st.metric("Wind Speed", api["windspeed"])
    st.metric("Pressure (hPa)", api["pressure"])
    st.metric("Precipitation (mm)", api["precip"])
    st.metric("Precipitation Probability (%)", api["precipprob"])
    st.metric("Weather Code", api["weathercode"])
    st.markdown("</div>", unsafe_allow_html=True)

# 2️⃣ RUN BUTTON
run = st.button("🔮 Predict Tomorrow's Weather", use_container_width=True)

# 3️⃣ PREDICTIONS
if run:
    X = to_feature_row(api, FEATURES)
    X_scaled = scaler.transform(X)

    rain_pred = rf_rain.predict(X_scaled)[0]
    rain_prob = rf_rain.predict_proba(X_scaled)[0][1]

    temp_pred = rf_temp.predict(X_scaled)[0]
    hum_pred = rf_humidity.predict(X_scaled)[0]

    condition = None
    if rf_weather:
        enc_val = rf_weather.predict(X_scaled)[0]
        condition = enc_weather.inverse_transform([enc_val])[0]

    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("🌧 Rain Tomorrow", "Yes" if rain_pred==1 else "No", f"{rain_prob*100:.1f}%")
        st.metric("🌡️ Temperature", f"{temp_pred:.1f} °C")
        st.metric("💧 Humidity", f"{hum_pred:.1f} %")
        if condition:
            st.metric("🌥️ Condition", condition)
        st.markdown("</div>", unsafe_allow_html=True)
        # --------------------------
    # 4️⃣ NEXT 3 DAYS TREND PLOT
    # --------------------------
    st.markdown("<div class='section-title'>📈 Next 3 Days Trend</div>", unsafe_allow_html=True)

    # Fetch forecast for next 3 days
    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        f"&timezone=auto"
    )
    forecast_res = requests.get(forecast_url).json()

    days = forecast_res["daily"]["time"][:3]
    temp_max = forecast_res["daily"]["temperature_2m_max"][:3]
    temp_min = forecast_res["daily"]["temperature_2m_min"][:3]
    rain_prob = forecast_res["daily"]["precipitation_probability_max"][:3]

    # Create dataframe for plotting
    trend_df = pd.DataFrame({
        "Day": days,
        "Max Temp": temp_max,
        "Min Temp": temp_min,
        "Rain Probability (%)": rain_prob
    })

    st.write(trend_df)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, temp_max, marker='o', label="Max Temp (°C)")
    ax.plot(days, temp_min, marker='o', label="Min Temp (°C)")
    ax.plot(days, rain_prob, marker='o', label="Rain Probability (%)")

    ax.set_xlabel("Day")
    ax.set_ylabel("Value")
    ax.set_title("Next 3 Days Weather Trend")
    ax.legend()

    st.pyplot(fig)

    # 5 DIAGNOSTICS
    diag = st.expander("📊 Model Diagnostics")
    with diag:
        fi = rf_rain.feature_importances_
        fi_df = pd.DataFrame({"feature": FEATURES, "importance": fi}).sort_values("importance")
        fig, ax = plt.subplots()
        ax.barh(fi_df["feature"], fi_df["importance"])
        st.pyplot(fig)
        st.write("Model: RandomForestClassifier for rain prediction")
