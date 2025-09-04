import streamlit as st
import joblib
import json
import pandas as pd

# -----------------------------
# 1. Load model and feature stats
# -----------------------------
MODEL_PATH = "models/model.pkl"
FEATURE_STATS_PATH = "metadata/feature_stats.json"

model = joblib.load(MODEL_PATH)

with open(FEATURE_STATS_PATH, "r") as f:
    stats = json.load(f)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Student Exam Pass/Fail Predictor 🎓")

st.write("กรอกข้อมูลนักศึกษาเพื่อทำนายผล Pass/Fail")

# numeric inputs
numeric_inputs = {}
for col, s in stats["numeric"].items():
    min_val = s["min"]
    max_val = s["max"]
    mean_val = s["mean"]
    numeric_inputs[col] = st.number_input(
        label=col.replace("_", " ").title(),
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(mean_val)
    )

# categorical inputs
categorical_inputs = {}
for col, s in stats["categorical"].items():
    choices = s["unique_values"]
    categorical_inputs[col] = st.selectbox(
        label=col.replace("_", " ").title(),
        options=choices
    )

# Predict button
if st.button("Predict Pass/Fail"):
    # combine inputs
    input_data = {**numeric_inputs, **categorical_inputs}
    X_new = pd.DataFrame([input_data])

    # predict numeric exam score
    score_pred = model.predict(X_new)[0]

    # convert to Pass/Fail
    threshold = 50  # กำหนดค่าผ่าน
    result = "Pass ✅" if score_pred >= threshold else "Fail ❌"

    st.success(f"Prediction: {result}")

    # predict
    pred = model.predict(X_new)[0]

    st.success(f"\nPredicted Exam Score: {pred:.2f}")