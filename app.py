
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("stacking_regressor_soil_moisture.pkl")

# Load trained features
trained_columns = joblib.load("trained_features.pkl")

st.title("Soil Moisture Prediction")

# User Input
temperature = st.number_input("Temperature (°C)", value=20.0)
precipitation = st.number_input("Precipitation (mm)", value=0.0)
humidity = st.number_input("Relative Humidity (%)", value=50.0)

if st.button("Predict"):
 # Convert input into DataFrame
 input_data = pd.DataFrame([[temperature, precipitation, humidity]], columns=trained_columns)
 
 # Standardize input
 scaler = StandardScaler()
 input_scaled = scaler.fit_transform(input_data)

 # Predict soil moisture
 prediction = model.predict(input_scaled)[0]

 st.success(f"Predicted Soil Moisture: {prediction:.2f}")
    