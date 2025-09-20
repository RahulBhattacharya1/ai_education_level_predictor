import streamlit as st
import pandas as pd
import joblib

st.title("Education Level Predictor (Compact Model)")

# Choose which model file you uploaded
MODEL_PATH = "/models/edu_pipeline.joblib"  # or "edu_pipeline_sgd.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipe = load_model()

st.subheader("Input Features")
sex = st.selectbox("Sex", ["M", "F", "T"])
age = st.text_input("Age group (e.g., Y20-24)", "Y20-24")
country = st.text_input("Country code (e.g., AT)", "AT")
year = st.number_input("Year", value=2020, step=1)
value = st.number_input("Population (THS)", value=100.0, step=1.0)

if st.button("Predict"):
    X_new = pd.DataFrame(
        [[sex, age, country, year, value]],
        columns=["sex","age","geography","date","value"]
    )
    pred = pipe.predict(X_new)[0]
    st.success(f"Predicted Education Level (ISCED 2011 band): {pred}")
