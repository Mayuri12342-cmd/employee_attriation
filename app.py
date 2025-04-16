import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# Set background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
            font-family: 'Segoe UI', sans-serif;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.4);  /* Adjust opacity here */
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
add_bg_from_local("background.jpg")

# Load model and feature names
model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title and subtitle
st.markdown("""
    <h1 style='text-align: center; color: #003366;'>🔍 Employee Attrition Predictor</h1>
    <p style='text-align: center; font-size:18px;'>Use key details to estimate if an employee might leave the company.</p>
""", unsafe_allow_html=True)

# Info expander
with st.expander("ℹ️ What does this do?"):
    st.markdown("This tool uses a machine learning model trained on employee data to predict if an individual is likely to leave the company.")

st.markdown("---")

# Selected features
selected_features = [
    'Age',
    'MonthlyIncome',
    'JobSatisfaction',
    'DistanceFromHome',
    'TotalWorkingYears',
    'YearsAtCompany',
    'OverTime_Yes'
]

input_data = {}

# Input layout
col1, col2 = st.columns(2)

with col1:
    input_data['Age'] = st.slider("🧑 Age", 18, 60, 30)
    input_data['JobSatisfaction'] = st.selectbox("😊 Job Satisfaction", [1, 2, 3, 4])
    input_data['TotalWorkingYears'] = st.slider("📈 Total Working Years", 0, 40, 10)
    input_data['OverTime_Yes'] = 1 if st.checkbox("🕒 Works OverTime?") else 0

with col2:
    input_data['MonthlyIncome'] = st.number_input("💰 Monthly Income", 1000, 20000, 5000)
    input_data['DistanceFromHome'] = st.slider("📍 Distance From Home (km)", 1, 50, 10)
    input_data['YearsAtCompany'] = st.slider("🏢 Years at Company", 0, 40, 5)

# input for prediction
full_input = {f: 0 for f in feature_names}
full_input.update(input_data)
input_df = pd.DataFrame([full_input])

# Prediction
st.markdown("---")
if st.button("🔮 Predict Attrition"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ This employee is likely to leave.")
    else:
        st.success("✅ This employee is likely to stay.")
