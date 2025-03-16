import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Must be the first Streamlit command
st.set_page_config(
    page_title="Soil Moisture Prediction",
    page_icon="🌱",
    layout="wide"
)

# ✅ Load trained model & trained feature names
model = joblib.load("stacking_regressor_soil_moisture.pkl")
trained_columns = joblib.load("trained_features.pkl")

# 🌱 **App Title & Description**
st.title("🌿 Soil Moisture Prediction App")
st.write("Welcome! Enter the weather parameters below to predict soil moisture levels.")

# 🌡️ **User Input Fields**
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1, value=20.0)
humidity = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, max_value=500.0, step=0.1, value=0.0)

# 🚀 **Add more parameters if needed**
wind_speed = st.number_input("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=50.0, step=0.1, value=5.0)
solar_radiation = st.number_input("☀️ Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, step=1.0, value=200.0)

# **Prediction Function**
def predict_soil_moisture(temp, hum, precip, wind, solar):
    input_data = np.array([[temp, hum, precip, wind, solar]])
    input_df = pd.DataFrame(input_data, columns=["temperature_2m (°C)", "relative_humidity_2m (%)", 
                                                  "precipitation (mm)", "wind_speed (m/s)", "solar_radiation (W/m²)"])
    
    # **Ensure columns match the trained model's features**
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    # ✅ **Predict soil moisture**
    prediction = model.predict(input_df)[0]
    return prediction

# **Button for Prediction**
if st.button("🔍 Predict"):
    result = predict_soil_moisture(temperature, humidity, precipitation, wind_speed, solar_radiation)
    st.success(f"🌿 **Predicted Soil Moisture: {result:.2f}**")

# **Footer**
st.write("---")
st.write("📌 Developed by Shushanth Premanand | Powered by ML & Streamlit 🚀")
