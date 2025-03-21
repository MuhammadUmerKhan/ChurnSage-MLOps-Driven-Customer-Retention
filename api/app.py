import streamlit as st
import requests
import pandas as pd
import database

# ✅ FastAPI Base URL (Ensure FastAPI is running)
API_URL = "http://127.0.0.1:8000"

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# ✅ Custom Styles
st.markdown("""
    <style>
        .main-title { font-size: 2.5em; font-weight: bold; color: #2C3E50; text-align: center; }
        .section-title { font-size: 1.8em; color: #3498DB; font-weight: bold; margin-top: 30px; }
        .footer { font-size: 14px; color: #95A5A6; margin-top: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ✅ App Title
st.markdown('<div class="main-title">📊 Customer Churn Prediction & LLM Analysis</div>', unsafe_allow_html=True)

# ✅ Tab Layout
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict Churn", "💬 LLM Review Analysis", "📊 View Stored Data", "📂 Upload CSV for Bulk Prediction"])

def send_api_request(endpoint, payload):
    """Handles API requests and exceptions."""
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Failed: {str(e)}")
        return None

# 🚀 TAB 1: Customer Churn Prediction
with tab1:
    st.markdown('<div class="section-title">🔍 Predict Customer Churn</div>', unsafe_allow_html=True)
    
    # User Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    with col2:
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"])
    
    # Numeric Fields
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, format="%.2f")
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0, format="%.2f")

    # Convert inputs to API format
    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod": {
            "Electronic check": 0, 
            "Mailed check": 1, 
            "Bank transfer (auto)": 2, 
            "Credit card (auto)": 3
        }[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # 🚀 Predict Button
    if st.button("Predict Churn"):
        with st.spinner("Analyzing Customer Data... 🔍"):
            response = send_api_request("predict", input_data)

            if response == "Yes":
                st.error(f"😢 Customer may leave!")
            else:
                st.success(f"😊 Customer likely to stay!")


# 💬 TAB 2: LLM Review Analysis
with tab2:
    st.markdown('<div class="section-title">💬 Analyze Customer Review with LLM</div>', unsafe_allow_html=True)
    user_feedback = st.text_area("Enter customer feedback review:")
    
    if st.button("Analyze Review with LLM"):
        if not user_feedback:
            st.warning("⚠️ Please enter some feedback!")
        else:
            with st.spinner("Analyzing review with LLM... 🤖"):
                response = send_api_request("predict_review", {"user_feedback": user_feedback})
                if response:
                    st.write(f"🔮 Prediction: {response['llm_prediction']}")
                    st.write(f"🔮 Reasoning {response['llm_reasoning']}")

# 🚀 TAB 3: View Stored Predictions
with tab3:
    st.markdown('<div class="section-title">📊 View Stored Predictions</div>', unsafe_allow_html=True)

    # ✅ Display stored customer churn predictions
    st.subheader("📌 Stored Customer Churn Predictions")
    customer_data = database.get_all_customer_data()
    customer_data = pd.DataFrame(customer_data, columns=[
            "ID", "Senior Citizen", "Partner", "Dependents", "tenure", "Online Security", "Tech Support","Contract",
            "Paperless Billing", "Payment Method", "Monthly Charges", "Total Charges", "Timestamp", "Prediction"
        ])
    st.dataframe(customer_data, hide_index=True, height=200)

    # ✅ Display stored LLM feedback analysis
    st.subheader("📌 Stored LLM Feedback")
    llm_feedback_data = database.get_all_llm_feedback()
    st.dataframe(pd.DataFrame(llm_feedback_data, 
                              columns=["ID", "User Feedback", "Timestamp", "LLM Prediction", "LLM Reasoning"]), 
                                hide_index=True)


# 🚀 TAB 4: Bulk Prediction from CSV
with tab4:
    st.markdown('<div class="section-title">📂 Bulk CSV Prediction</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a cleaned CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # ✅ Read the uploaded CSV
            df_upload = pd.read_csv(uploaded_file)
            st.write("✔️ File successfully uploaded! Preview:")
            st.dataframe(df_upload, height=200)
            
            df_pred = df_upload.copy()
            # ✅ Ensure correct columns exist
            required_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity',
                                'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                'MonthlyCharges', 'TotalCharges']
            if not all(col in df_pred.columns for col in required_columns):
                st.error("⚠️ Uploaded CSV is missing required columns!")
            else:
                # ✅ Scale numeric features
                df_pred[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
                    df_pred[['tenure', 'MonthlyCharges', 'TotalCharges']]
                )
                
                # ✅ Get predictions
                predictions = loaded_model.predict(df_pred)
                df_upload['Prediction'] = np.where(predictions == 1, "Yes", "No")
                
                # ✅ Display and allow downloading
                st.success("✔️ Predictions completed!")
                st.dataframe(df_upload, height=200)
                
                # ✅ Downloadable CSV
                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


# ✅ Footer
st.markdown('<div class="footer">Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a> 🚀</div>', unsafe_allow_html=True)