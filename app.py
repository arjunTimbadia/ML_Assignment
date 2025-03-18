import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Try to load the trained model and scaler
try:
    model = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}. Please run assignment.py first to train the model.")

# Streamlit app layout
st.title("Disease Prediction App")

# Collect user input
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=22.0
)

glucose = st.number_input(
    "Glucose Level",
    min_value=50.0,
    max_value=200.0,
    value=100.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol Level",
    min_value=100,
    max_value=300,
    value=200
)

# Categorical inputs
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
alcohol = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
family_history = st.selectbox("Family History of Disease?", ["Yes", "No"])

# Map categorical inputs to numerical format
smoking = 1 if smoking == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0

# Prepare input features for the model
input_features = np.array([[
    age,
    bmi,
    glucose,
    blood_pressure,
    cholesterol,
    smoking,
    alcohol,
    family_history
]])

# When the button is pressed, predict the result
if st.button("Predict"):
    try:
        # Scale the input features
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)
        
        if prediction[0] == 1:
            st.write("Prediction: Disease Present")
        else:
            st.write("Prediction: No Disease")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
