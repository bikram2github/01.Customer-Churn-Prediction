import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler_churn.pkl")
    return model, scaler    

model, scaler = load_model()

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they will leave or stay.")


@st.cache_data
def user_input_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    credit_score = credit_score
    if geography == "France":
        geography= [0, 0]
    elif geography == "Spain":
        geography= [1, 0]
    else:
        geography= [0, 1]
    if gender == "Male":
        gender = 1
    else:
        gender = 0
    age = age
    tenure = tenure
    balance = balance
    num_of_products = num_of_products
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    estimated_salary = estimated_salary
    return np.array([
        credit_score,
        geography[0],
        geography[1],
        gender,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary
    ]).reshape(1, -1)


with st.form("churn_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card", options=["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", options=["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = user_input_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.subheader("Churn Probability ⬇️")
        proba = model.predict_proba(input_scaled)[0][1] 
        st.progress(int(proba * 100))
        st.write(f"Predicted probability of leaving: {proba * 100:.2f}%")


        if proba > 0.4:
            st.error("🚨 The customer is likely to leave.")
        elif proba > 0.25: 
            st.warning("⚠️ The customer has a moderate chance of leaving.")
        else:
            st.success("✅ The customer is likely to stay.")
            st.balloons()