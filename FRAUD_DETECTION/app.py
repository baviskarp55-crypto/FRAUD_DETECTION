# FRAUD DETECTION SYSTEM - STREAMLIT APP

import streamlit as st
import pandas as pd
import joblib

# 1. LOAD TRAINED MODEL & SCALER

model = joblib.load("fraud_detection_model.pkl")   # Trained ML model
scaler = joblib.load("scaler.pkl")                 # StandardScaler
feature_names = joblib.load("feature_names.pkl")   # Feature order

# 2. PAGE SETTINGS
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Fraud Detection System")
st.write("Check whether a transaction is **Fraud** or **Legitimate**")

# 3. USER INPUT SECTION
st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, step=100.0)
hour = st.slider("Transaction Hour (0 - 23)", 0, 23, 12)
frequency = st.number_input("Transaction Frequency", min_value=0)
account_age = st.number_input("Account Age (Months)", min_value=0)
international = st.selectbox("Is International Transaction?", ["No", "Yes"])

# Convert Yes/No → 0/1
international = 1 if international == "Yes" else 0

# 4. PREDICTION BUTTON
if st.button("🔍 Predict Fraud"):

    # Create DataFrame in correct feature order
    input_data = pd.DataFrame([[ 
        amount,
        hour,
        frequency,
        account_age,
        international
    ]], columns=feature_names)

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # 5. DISPLAY RESULT
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("FRAUD TRANSACTION DETECTED")
    else:
        st.success("NOT FRAUD TRANSACTION")

    st.metric("Fraud Probability", f"{probability:.2f}")
