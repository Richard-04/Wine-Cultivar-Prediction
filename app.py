import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load model and scaler
# -----------------------
MODEL_PATH = "best_svm.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="🍷 Wine Cultivar Predictor", layout="centered")
st.title("🍷 Wine Cultivar Prediction App")
st.markdown("🧪Input chemical properties of the wine to predict its cultivar/origin.")

# -----------------------
# Selected 6 features with realistic ranges
# -----------------------
FEATURE_NAMES = [
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'magnesium',
    'total_phenols'
]

FEATURE_RANGES = {
    'alcohol': (11.0, 15.0, 13.0),
    'malic_acid': (0.7, 5.8, 2.0),
    'ash': (1.3, 3.2, 2.4),
    'alcalinity_of_ash': (10.0, 30.0, 19.0),
    'magnesium': (70, 160, 100),
    'total_phenols': (0.9, 5.0, 2.5)
}

TARGET_NAMES = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']

# -----------------------
# Collect user input
# -----------------------
st.subheader("Wine Features Input")
user_data = {}
cols = st.columns(2)

for i, feature in enumerate(FEATURE_NAMES):
    min_val, max_val, default_val = FEATURE_RANGES[feature]
    label = feature.replace("_", " ").title()
    col = cols[i % 2]
    user_data[feature] = col.slider(label, float(min_val), float(max_val), float(default_val))

# Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# Apply scaling
input_scaled = scaler.transform(input_df)

# -----------------------
# Predict button
# -----------------------
if st.button("Predict Cultivar 🍇🍷"):
    prediction = model.predict(input_scaled)[0]
    cultivar = TARGET_NAMES[prediction]
    st.success(f"Predicted Wine Cultivar: {cultivar}")

# -----------------------
# Display EDA visuals (if you want)
# -----------------------
st.subheader("Exploratory Data Visuals")
st.image("static/eda_scatter.png", caption="Alcohol vs Flavanoids")
st.image("static/class_distribution.png", caption="Class Distribution")
