import streamlit as st
import pandas as pd
from joblib import load

st.title("Diabetes Prediction")
st.subheader("Enter your details")

st.write("Click the Predict button to get the model's prediction.")

preg = st.number_input(
    "Pregnancies", min_value=0, max_value=20, value=2, step=1
)

gluc = st.number_input(
    "Glucose", min_value=0, max_value=200, value=120, step=1
)

bp = st.number_input(
    "BloodPressure", min_value=0, max_value=150, value=70, step=1
)

skin = st.number_input(
    "SkinThickness", min_value=0, max_value=100, value=20, step=1
)

insulin = st.number_input(
    "Insulin", min_value=0, max_value=900, value=80, step=1
)

bmi = st.number_input(
    "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1
)

dpf = st.number_input(
    "DiabetesPedigreeFunction",
    min_value=0.0,
    max_value=3.0,
    value=0.47,
    step=0.01
)

age = st.number_input(
    "Age", min_value=1, max_value=120, value=30, step=1
)

new_data = pd.DataFrame({
    "Pregnancies": [preg],
    "Glucose": [gluc],
    "BloodPressure": [bp],
    "SkinThickness": [skin],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

model = load(r"model_dir\diabetes_model.joblib")

if st.button("Predict"):
    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.error("The model predicts a positive result.")
    else:
        st.success("The model predicts a negative result.")

    st.write("Prediction:", prediction[0])