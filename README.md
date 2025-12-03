🌦️ Smart Weather + ML

Real-time weather prediction app using Streamlit, Random Forest, and Open-Meteo API.

🚀 Features

Auto-detects current city, state, country

Fetches real-time weather (temperature, humidity, wind, pressure, rain prob.)

ML predictions:

🌧 Rain Tomorrow (Yes/No + Confidence)

🌡 Temperature Tomorrow

💧 Humidity Tomorrow

🌥 Weather Condition (optional)

📈 Next 3 Days Trend Plot

📊 Model Diagnostics (feature importance)

🧠 Machine Learning

Models trained on 2013–2024 Indian weather dataset:

rf_rain.pkl

rf_temp.pkl

rf_humidity.pkl

rf_weather.pkl (optional)

scaler.pkl

🛠 Tech Stack

Python, Streamlit

pandas, numpy

scikit-learn

matplotlib, seaborn

Open-Meteo API

ipapi (location detection)

📦 Project Structure
/models
    scaler.pkl
    rf_rain.pkl
    rf_temp.pkl
    rf_humidity.pkl
    rf_weather.pkl
    enc_weather.pkl
app.py
model.py
requirements.txt
README.md

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

🌐 Deployment

Hosted easily on Streamlit Cloud:
Connect GitHub → Select repo → Deploy.
