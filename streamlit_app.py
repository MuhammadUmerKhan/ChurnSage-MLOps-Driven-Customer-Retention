import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os
import joblib

# ✅ Load MLflow production model
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model_name = "customer_churn_model"
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models_joblib_files", "scaler.pkl"))
scaler = joblib.load(scaler_path)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-title { font-size: 2.5em; font-weight: bold; color: #2C3E50; text-align: center; margin-bottom: 20px; }
        .section-title { font-size: 1.8em; color: #3498DB; font-weight: bold; margin-top: 30px; text-align: left; }
        .content { font-size: 1em; color: #7F8C8D; text-align: justify; line-height: 1.6; }
        .footer { font-size: 14px; color: #95A5A6; margin-top: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Welcome to the Customer Churn Prediction Tool 📊</div>', unsafe_allow_html=True)

# Feature Input Section
st.header("Predict Customer Churn 🔍")
col1, col2, col3 = st.columns(3)
with col1:
    monthly_charges = st.number_input("Monthly Charges", value=0.0, format="%.2f")
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen == "Yes" else 0
with col2:
    tenure = st.number_input("Tenure (months)", value=0, format="%d")
    partner = st.selectbox("Partner", ["No", "Yes"])
    partner = 1 if partner == "Yes" else 0
with col3:
    total_charges = st.number_input("Total Charges", value=0.0, format="%.2f")
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    dependents = 1 if dependents == "Yes" else 0

# Additional categorical inputs
col4, col5, col6 = st.columns(3)
with col4:
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_security = 1 if online_security == "Yes" else 0
with col5:
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    tech_support = 1 if tech_support == "Yes" else 0
with col6:
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    paperless_billing = 1 if paperless_billing == "Yes" else 0

col7, col8 = st.columns(2)
with col7:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
with col8:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    payment_method = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }[payment_method]

# Predict button
if st.button("Predict Churn 🚀"):
    try:
        input_data = [[
            senior_citizen, partner, dependents, tenure, online_security,
            tech_support, contract, paperless_billing, payment_method,
            monthly_charges, total_charges
        ]]
        input_df = pd.DataFrame(input_data, columns=[
            'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity',
            'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ])

        # Scale numeric features
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        prediction = loaded_model.predict(input_df)

        # Display result
        if prediction[0] == 1:
            st.error(f"😢 Customer may leave!")
        else:
            st.success(f"😊 Customer likely to stay!")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by MLflow and Streamlit. 🚀
    </div>
""", unsafe_allow_html=True)