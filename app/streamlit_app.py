"""Streamlit demo app for stroke risk prediction."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st

from stroke_risk.predict import predict_one

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Stroke Risk Prediction")
st.caption("MLOps demo - healthcare domain. Enter patient features to get a risk score.")

with st.form("patient"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 1, 100, 50)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart disease", [0, 1])
        ever_married = st.selectbox("Ever married", ["Yes", "No"])
    with col2:
        work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence type", ["Urban", "Rural"])
        glucose = st.number_input("Avg glucose level", 50.0, 300.0, 110.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 28.0)
        smoking = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking,
    }
    try:
        res = predict_one(record)
        st.metric("Stroke probability", f"{res['stroke_probability']:.2%}")
        if res["stroke_risk"] == "high":
            st.error(f"⚠️ HIGH RISK (threshold {res['threshold']})")
        else:
            st.success(f"✅ LOW RISK (threshold {res['threshold']})")
        st.json(res)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Train the model first: `python scripts/train.py`")