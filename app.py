import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Load model, scaler, and features
# -----------------------
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("scaler.pkl")  # ensure this is fitted on selected_features
selected_features = joblib.load("model/selected_features.pkl")

st.set_page_config(page_title="🍷 Wine Cultivar Predictor", layout="centered")
st.title("🍷 Wine Cultivar Prediction App")
st.markdown("🧪 Input chemical properties of the wine to predict its cultivar/origin.")

# -----------------------
# Collect user input
# -----------------------
st.subheader("Wine Features")
user_data = {}
cols = st.columns(2)

for i, feature in enumerate(selected_features):
    label = feature.replace("_", " ").title()
    col = cols[i % 2]
    user_data[feature] = col.slider(label, 0.0, 20.0, 5.0)

input_df = pd.DataFrame([user_data])
input_df = input_df[selected_features]  # enforce order

# -----------------------
# Predict button
# -----------------------
if st.button("Predict Cultivar 🍇🍷"):
    try:
        # Only transform if scaler exists
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = np.max(model.predict_proba(input_scaled))
    except Exception:
        # fallback: if scaler not needed
        prediction = model.predict(input_df)[0]
        probability = np.max(model.predict_proba(input_df))

    st.success(f"Predicted Cultivar: Class {prediction} ({probability*100:.2f}% confidence)")
