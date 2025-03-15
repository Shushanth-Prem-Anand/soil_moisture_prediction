import streamlit as st  # ✅ Import Streamlit FIRST
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Move `set_page_config()` to the very top (Only Call Once)
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

# ✅ **Fix: Prediction Function (Convert DataFrame to NumPy)**
def predict_soil_moisture(temp, hum, precip):
    input_data = np.array([[temp, hum, precip]])
    input_df = pd.DataFrame(input_data, columns=["temperature_2m (°C)", "relative_humidity_2m (%)", "precipitation (mm)"])
    
    # **Ensure columns match trained model's features**
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    # ✅ **Convert DataFrame to NumPy for XGBoost**
    input_array = input_df.to_numpy().astype(float)

    # ✅ **Predict soil moisture**
    prediction = model.predict(input_array)[0]
    return prediction

# **Button for Prediction**
if st.button("🔍 Predict"):
    result = predict_soil_moisture(temperature, humidity, precipitation)
    st.success(f"🌿 **Predicted Soil Moisture: {result:.2f}**")

# **Footer**
st.write("---")
st.write("📌 Developed by Shushanth Premanand | Powered by ML & Streamlit 🚀")
