# ✅ Must be the very first command
import streamlit as st
st.set_page_config(
    page_title="Soil Moisture Prediction",
    page_icon="🌱",
    layout="wide"
)

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Load trained model & trained features
model = joblib.load("stacking_regressor_soil_moisture.pkl")
trained_columns = joblib.load("trained_features.pkl")

# 🌱 **App Title & Description**
st.title("🌿 Soil Moisture Prediction App")
st.write("Welcome! Enter the weather parameters below to predict soil moisture levels.")

# 🌡️ **User Input Fields**
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1, value=20.0)
humidity = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, max_value=500.0, step=0.1, value=0.0)

# **Prediction Function**
def predict_soil_moisture(temp, hum, precip):
    input_data = np.array([[temp, hum, precip]])
    input_df = pd.DataFrame(input_data, columns=["temperature_2m (°C)", "relative_humidity_2m (%)", "precipitation (mm)"])
    
    # **Ensure columns match trained model's features**
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    # ✅ **Predict soil moisture**
    prediction = model.predict(input_df)[0]
    return prediction

# **Button for Prediction**
if st.button("🔍 Predict"):
    result = predict_soil_moisture(temperature, humidity, precipitation)
    st.success(f"🌿 **Predicted Soil Moisture: {result:.2f}**")

# **Footer**
st.write("---")
st.write("📌 Developed by Shushanth Premanand | Powered by ML & Streamlit 🚀")
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

# **Prediction Function**
def predict_soil_moisture(temp, hum, precip):
    input_data = np.array([[temp, hum, precip]])
    input_df = pd.DataFrame(input_data, columns=["temperature_2m (°C)", "relative_humidity_2m (%)", "precipitation (mm)"])
    
    # **Ensure columns match the trained model's features**
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    # ✅ **Predict soil moisture**
    prediction = model.predict(input_df)[0]
    return prediction

# **Button for Prediction**
if st.button("🔍 Predict"):
    result = predict_soil_moisture(temperature, humidity, precipitation)
    st.success(f"🌿 **Predicted Soil Moisture: {result:.2f}**")

# **Footer**
st.write("---")
st.write("📌 Developed by Shushanth Premanand | Powered by ML & Streamlit 🚀")

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
    
import streamlit as st

st.set_page_config(
    page_title="Soil Moisture Prediction",
    page_icon="🌱",
    layout="wide"
)

st.write("Welcome to the Soil Moisture Prediction App! 🚀")
import streamlit as st  # Keep this at the top!
st.set_page_config(page_title="Soil Moisture Prediction", layout="wide")

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("stacking_regressor_soil_moisture.pkl")
trained_columns = joblib.load("trained_features.pkl")

# User Input Fields
st.title("🌱 Soil Moisture Prediction App")
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Predict Function
def predict_soil_moisture():
    input_data = np.array([[temperature, humidity, precipitation]])
    input_df = pd.DataFrame(input_data, columns=["temperature_2m (°C)", "relative_humidity_2m (%)", "precipitation (mm)"])
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)
    
    # Standardize
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    return prediction

# Button for Prediction
if st.button("Predict"):
    result = predict_soil_moisture()
    st.success(f"🌿 Predicted Soil Moisture: {result:.2f}")

