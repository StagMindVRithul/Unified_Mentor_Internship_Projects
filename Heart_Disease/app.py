import streamlit as st
import numpy as np
import pickle
import joblib
from scipy.stats import boxcox

# Load model and transformation lambda
model = joblib.load('/Users/rithul.v/Desktop/Projects/Heart_Disease/model.pkl')

with open("/Users/rithul.v/Desktop/Projects/Heart_Disease/boxcox_lambda_resting_bp.pkl", "rb") as f:
    boxcox_lambda = pickle.load(f)

# Streamlit UI
st.title("Heart Disease Predictor")
st.subheader("Input Patient Details")

# Dropdowns and sliders
st_slope = st.selectbox("ST slope", options=['up', 'flat', 'down'])
max_heart_rate = st.slider("Max Heart Rate", min_value=70, max_value=205, value=120)
exercise_angina = st.selectbox("Exercise Angina", options=['no', 'yes'])
chest_pain_type = st.selectbox("Chest Pain Type", options=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
age = st.slider("Age", min_value=25, max_value=80, value=40)
oldpeak = st.slider("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, step=0.1)
cholesterol = st.slider("Cholesterol", min_value=80, max_value=600, value=200)
resting_bp = st.slider("Resting BP (systolic)", min_value=90, max_value=200, value=120)

# Mapping categorical inputs to numerical values (match your training)
st_slope_map = {'up': 1, 'flat': 2, 'down': 3}
exercise_angina_map = {'no': 0, 'yes': 1}
chest_pain_type_map = {
    'typical angina': 1,
    'atypical angina': 2,
    'non-anginal pain': 3,
    'asymptomatic': 4
}

# Transformations
oldpeak_log = np.log1p(oldpeak)
cholesterol_log = np.log1p(cholesterol)
resting_bp_boxcox = boxcox(resting_bp + 1, boxcox_lambda)

# Feature vector
features = np.array([[
    st_slope_map[st_slope],
    max_heart_rate,
    exercise_angina_map[exercise_angina],
    chest_pain_type_map[chest_pain_type],
    age,
    oldpeak_log,
    cholesterol_log,
    resting_bp_boxcox
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)[0]
    label = "Normal (0)" if prediction == 0 else "Heart Disease (1)"
    st.success(f"Prediction: {label}")
