import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------
# Load model, scaler, and EDA images
# -----------------------
MODEL_PATH = "best_svm.pkl"
SCALER_PATH = "scaler.pkl"
EDA_SCATTER = "static/eda_scatter.png"
CLASS_DIST = "static/class_distribution.png"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="🍷 Wine Cultivar Predictor", layout="centered")
st.title("🍷 Wine Cultivar Prediction App")
st.markdown("🧪Input the chemical properties of the wine to predict its cultivar/class.")

# -----------------------
# Selected 6 features for prediction
# -----------------------
FEATURE_NAMES = [
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'magnesium',
    'total_phenols'
]

TARGET_NAMES = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']

# -----------------------
# Display EDA visuals
# -----------------------
st.subheader("EDA Visuals")
col1, col2 = st.columns(2)
col1.image(EDA_SCATTER, caption="Alcohol vs Flavanoids", use_column_width=True)
col2.image(CLASS_DIST, caption="Class Distribution", use_column_width=True)

# -----------------------
# Collect user input
# -----------------------
st.subheader("Input Wine Features")
user_data = {}
cols = st.columns(2)

for i, feature in enumerate(FEATURE_NAMES):
    label = feature.replace("_", " ").title()
    col = cols[i % 2]
    user_data[feature] = col.slider(label, 0.0, 20.0, 5.0)

input_df = pd.DataFrame([user_data])

# -----------------------
# Predict button
# -----------------------
if st.button("Predict Cultivar 🍇🍷"):
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    cultivar = TARGET_NAMES[prediction]

    st.success(f"Predicted Wine Cultivar: {cultivar}")
