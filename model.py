# train_models.py

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

sns.set()

CSV_PATH = "kaggel_weather_2013_to_2024.csv"


def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = ["temp", "humidity", "windspeed", "sealevelpressure",
                "precip", "precipprob", "conditions"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Create RainToday
    df["raintoday"] = (df["precip"] > 0).astype(int)

    # Shift for tomorrow predictions
    df["raintomorrow"] = df["raintoday"].shift(-1)
    df["temptomorrow"] = df["temp"].shift(-1)
    df["humiditytomorrow"] = df["humidity"].shift(-1)
    df["weathertomorrow"] = df["conditions"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def prepare_features(df):
    feature_cols = [
        "temp", "humidity", "windspeed", "sealevelpressure",
        "precip", "precipprob", "raintoday"
    ]

    X = df[feature_cols]
    X = X.fillna(X.median())

    y_rain = df["raintomorrow"].astype(int)
    y_temp = df["temptomorrow"].astype(float)
    y_humidity = df["humiditytomorrow"].astype(float)
    y_weather = df["weathertomorrow"].astype(str)

    return X, y_rain, y_temp, y_humidity, y_weather


def train_and_save_models(X, y_rain, y_temp, y_humidity, y_weather):
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ----------------------------
    # 1️⃣ RainTomorrow (Classification)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_rain, test_size=0.2, shuffle=True
    )
    rf_rain = RandomForestClassifier(n_estimators=200)
    rf_rain.fit(X_train, y_train)
    with open("models/rf_rain.pkl", "wb") as f:
        pickle.dump(rf_rain, f)

    # ----------------------------
    # 2️⃣ TempTomorrow (Regression)
    # ----------------------------
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_scaled, y_temp, test_size=0.2
    )
    rf_temp = RandomForestRegressor(n_estimators=200)
    rf_temp.fit(X_train2, y_train2)
    with open("models/rf_temp.pkl", "wb") as f:
        pickle.dump(rf_temp, f)

    # ----------------------------
    # 3️⃣ HumidityTomorrow (Regression)
    # ----------------------------
    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X_scaled, y_humidity, test_size=0.2
    )
    rf_humidity = RandomForestRegressor(n_estimators=200)
    rf_humidity.fit(X_train3, y_train3)
    with open("models/rf_humidity.pkl", "wb") as f:
        pickle.dump(rf_humidity, f)

    # ----------------------------
    # 4️⃣ Weather Condition Tomorrow (Classification)
    # ----------------------------
    le = LabelEncoder()
    y_weather_encoded = le.fit_transform(y_weather)

    rf_weather = RandomForestClassifier(n_estimators=150)
    rf_weather.fit(X_scaled, y_weather_encoded)

    with open("models/rf_weather.pkl", "wb") as f:
        pickle.dump(rf_weather, f)

    with open("models/enc_weather.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\n📌 All models saved successfully inside models/")


def main():
    df = load_dataset(CSV_PATH)
    X, y_rain, y_temp, y_humidity, y_weather = prepare_features(df)
    train_and_save_models(X, y_rain, y_temp, y_humidity, y_weather)


if __name__ == "__main__":
    main()

FEATURES = ["temperature", "humidity", "windspeed", "pressure"]

import pickle
with open("features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)


'''import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ------------------------------
# 1. LOAD DATA
# ------------------------------
df = pd.read_csv("kaggel_weather_2013_to_2024.csv")

# Keep only useful columns that exist
needed = ["temp", "humidity", "precip", "windspeed", "sealevelpressure"]
existing = [c for c in needed if c in df.columns]
df = df[existing].copy()

# Create targets for tomorrow
df["RainToday"] = (df["precip"] > 0).astype(int)
df["RainTomorrow"] = df["RainToday"].shift(-1)
df["TempTomorrow"] = df["temp"].shift(-1)
df["HumidityTomorrow"] = df["humidity"].shift(-1)

df = df.dropna().reset_index(drop=True)

# ------------------------------
# 2. FEATURE / TARGETS
# ------------------------------
X = df[["temp", "humidity", "precip", "windspeed", "sealevelpressure", "RainToday"]]
y_rain = df["RainTomorrow"]
y_temp = df["TempTomorrow"]
y_hum = df["HumidityTomorrow"]

# ------------------------------
# 3. SCALE FEATURES
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 4. TRAIN MODELS
# ------------------------------
rf_rain = RandomForestClassifier()
rf_rain.fit(X_scaled, y_rain)

rf_temp = RandomForestRegressor()
rf_temp.fit(X_scaled, y_temp)

rf_hum = RandomForestRegressor()
rf_hum.fit(X_scaled, y_hum)

# ------------------------------
# 5. SAVE EVERYTHING IN ONE FILE
# ------------------------------
model_pack = {
    "scaler": scaler,
    "rf_rain": rf_rain,
    "rf_temp": rf_temp,
    "rf_hum":  rf_hum
}

with open("weather_model.pkl", "wb") as f:
    pickle.dump(model_pack, f)

print("\nMODEL TRAINED SUCCESSFULLY!")
print("Saved as weather_model.pkl")
'''