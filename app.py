import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Load model, scaler, and features
# -----------------------
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("scaler.pkl")  # make sure this is the one fitted during training
selected_features = joblib.load("model/selected_features.pkl")  # list of 6 features used

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="🍷 Wine Cultivar Predictor", layout="centered")
st.title("🍷 Wine Cultivar Prediction App")
st.markdown("🧪Input chemical properties of the wine to predict its cultivar/origin.")

# -----------------------
# Collect user input
# -----------------------
st.subheader("Wine Features")
user_data = {}
cols = st.columns(2)  # two columns for sliders

# dynamically create inputs for selected features
for i, feature in enumerate(selected_features):
    label = feature.replace("_", " ").title()
    col = cols[i % 2]  # alternate columns

    # Adjust slider ranges (example: you can tweak these according to your dataset ranges)
    user_data[feature] = col.slider(label, 0.0, 20.0, 5.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# -----------------------
# Predict button
# -----------------------
if st.button("Predict Cultivar 🍇🍷"):
    # Ensure correct feature order
    input_df = input_df[selected_features]

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Predict class
    prediction = model.predict(input_scaled)[0]
    probability = np.max(model.predict_proba(input_scaled))  # highest probability

    st.success(f"Predicted Cultivar: Class {prediction} ({probability*100:.2f}% confidence)")
