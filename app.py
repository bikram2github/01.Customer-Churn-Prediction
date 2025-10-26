import streamlit as st
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
def user_input_features(credit_score, geography, gender, age, tenure,
                        balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    # Geography one-hot (Germany, Spain)
    if geography == "France":
        geo_germany, geo_spain = 0, 0
    elif geography == "Germany":
        geo_germany, geo_spain = 1, 0
    else:  # Spain
        geo_germany, geo_spain = 0, 1

    gender_male = 1 if gender == "Male" else 0
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0

    # ✅ Correct feature order
    features = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
        geo_germany,
        geo_spain,
        gender_male
    ]])

    return features

with st.form("churn_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3, step=1)
    balance = st.number_input("Balance", min_value=0.0, value=1000.0, format="%.2f")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1, step=1)
    has_cr_card = st.selectbox("Has Credit Card", options=["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", options=["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = user_input_features(
            credit_score, geography, gender, age, tenure,
            balance, num_of_products, has_cr_card, is_active_member, estimated_salary
        )

        # Scale & predict
        input_scaled = scaler.transform(input_data)
        proba = model.predict_proba(input_scaled)[0]

        st.subheader("Churn Probability ⬇️")
        col1, col2 = st.columns(2)
        col1.metric("Stay Probability", f"{proba[0]*100:.2f}%")
        col2.metric("Leave Probability", f"{proba[1]*100:.2f}%")

        st.subheader("Prediction ⬇️")
        if proba[1] > 0.5:
            st.error("🚨 The customer is likely to leave.")
        elif proba[1] > 0.25:
            st.warning("⚠️ The customer has a moderate chance of leaving.")
        else:
            st.success("✅ The customer is likely to stay.")
            st.balloons()
