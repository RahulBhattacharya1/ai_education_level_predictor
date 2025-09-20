import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("edu_model.pkl")

st.title("Education Level Predictor")

# Inputs
sex = st.selectbox("Sex", ["M", "F"])
age = st.text_input("Age group", "Y20-24")
country = st.text_input("Country code", "AT")
year = st.number_input("Year", 2020)
value = st.number_input("Population (thousands)", 100)

# Convert input to dataframe
input_df = pd.DataFrame([[sex, age, country, year, value]],
                        columns=["sex", "age", "geography", "date", "value"])

# One-hot encode to match training
input_df = pd.get_dummies(input_df).reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
prediction = model.predict(input_df)[0]
st.write(f"Predicted Education Level: {prediction}")
