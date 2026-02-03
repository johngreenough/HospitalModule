import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load "best" model
@st.cache_resource
def load_model():
    model = joblib.load('best_triage_model.pkl')
    return model

model = load_model()

st.title("Hospital Triage Decision Support System")

# User inputs
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.number_input("Age", 0, 120, 50)
injury = st.selectbox("Injury", ["No", "Yes"])
pain = st.selectbox("Pain Present?", ["No", "Yes"])
mental = st.selectbox("Mental State", ["Alert", "Verbal Response", "Pain Response", "Unresponsive"])

sbp = st.number_input("Systolic BP", 60, 250, 120)
dbp = st.number_input("Diastolic BP", 30, 150, 80)
hr = st.number_input("Heart Rate", 30, 200, 80)
rr = st.number_input("Respiratory Rate", 5, 60, 18)
temp = st.number_input("Temperature (°C)", 30.0, 45.0, 36.5)
saturation = st.number_input("Oxygen Saturation (%)", 70.0, 100.0, 98.0)
nrs_pain = st.number_input("Pain Score (0–10)", 0, 10, 3)
ktas_rn = st.number_input("KTAS Score by Nurse", 1, 5, 3)

# Ordinal encoding for Mental State to match cleaned data
mental_order = ["Alert", "Verbal Response", "Pain Response", "Unresponsive"]
mental_ord = mental_order.index(mental) + 1  # 1-based index (as per dataset)

# Match input format
input_data = {
    'Age': age,
    'NRS_pain': nrs_pain,
    'SBP': sbp,
    'DBP': dbp,
    'HR': hr,
    'RR': rr,
    'BT': temp,
    'Saturation': saturation,
    'KTAS_RN': ktas_rn,
    'Sex_Male': int(sex == "Male"),
    'Injury_Yes': int(injury == "Yes"),
    'Pain_Yes': int(pain == "Yes"),
    'Mental_ord': mental_ord
}

# Predict
if st.button("Predict Triage Level"):
    input_df = pd.DataFrame([input_data])
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)

        decoded_pred = pred + 1  # Shift back to KTAS 1–5

        st.success(f"Predicted Triage Level: **KTAS {decoded_pred}**")

        proba_df = pd.DataFrame(proba, columns=[f"KTAS {i+1}" for i in range(proba.shape[1])])
        highlight = ["background-color: yellow" if i == pred else "" for i in range(proba.shape[1])]
        st.dataframe(proba_df.style.apply(lambda _: highlight, axis=1))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(input_df)