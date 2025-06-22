import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc, joblib, dotenv, langchain_groq, pickle, re
from langchain.schema import HumanMessage
from api import database
from scripts import config

dotenv.load_dotenv()

# ✅ Load MLflow production model
mlflow.set_tracking_uri(f"sqlite:///{config.mlflow_db_path}")

# model_name = "customer_churn_model"
# loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

with open("./mlruns/2/356b34f4a73c478fa27eeba06f16b349/artifacts/models/model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
    print("✅ Model successfully loaded!")
    
scaler = joblib.load(config.scaler_path)

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
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict Churn", "💬 LLM Review Analysis", "📊 View Stored Data", "📂 Upload CSV for Bulk Prediction"])

with tab1:
    st.markdown('<div class="section-title">🔍 Predict Customer Churn</div>', unsafe_allow_html=True)
    
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

    # ✅ Map categorical values
    mapped_data = {
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
            "Bank transfer (automatic)": 2, 
            "Credit card (automatic)": 3
        }[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Predict button
    if st.button("Predict Churn 🚀"):
        try:
            input_df = pd.DataFrame([mapped_data])

            # ✅ Scale numeric features
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
                input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
            )

            # ✅ Get prediction
            prediction = loaded_model.predict(input_df)

            # ✅ Map prediction to label
            churn_prediction = "Customer likely to leave" if prediction[0] == 1 else "Customer likely to Stay"
            
            # ✅ Store in database
            try:
                database.save_customer_data(mapped_data, "Yes" if churn_prediction == "Customer likely to leave" else "No")
            except Exception as e:
                st.error(f"�� Error storing prediction in DB: {str(e)}")

            # ✅ Display result
            if prediction[0] == 1:
                st.error(f"{churn_prediction} 😢")
            else:
                st.success(f"{churn_prediction} 😊")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

with tab2:
    st.markdown('<div class="section-title">💬 Analyze Customer Review with LLM</div>', unsafe_allow_html=True)

    user_feedback = st.text_area("Enter customer feedback for churn prediction:")

    if st.button("Predict with LLM 🚀"):
        if not user_feedback:
            st.warning("⚠️ Please enter some feedback!")
        else:
            try:
                # ✅ Load LLM
                llm = langchain_groq.ChatGroq(groq_api_key=config.GROK_API_KEY, model_name="qwen-qwq-32b")

                # ✅ Construct prompt
                prompt = f"""
                You are an expert telecom retention analyst. Given the following customer review, predict whether the customer is likely to churn:

                🔹 **Customer Review:** "{user_feedback}"

                🎯 **Your Task:**
                - Analyze sentiment and concerns in the review.
                - Predict if the customer is likely to leave or stay.
                - Provide a short but engaging explanation for your decision.

                📌 **Format your response as follows:**
                - **Prediction:** ("Customer likely to leave" or "Customer will stay")
                - **Reasoning:** A brief but engaging analysis.

                🚀 **Make it sound professional yet interesting!**
                """

                # ✅ Get LLM response
                response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
                # Remove any <think> tags
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                # ✅ Parse LLM response
                if "Customer likely to leave" in response:
                    llm_prediction = "Yes"
                else:
                    llm_prediction = "No"
                try:
                    # ✅ Save to database
                    database.save_llm_feedback(user_feedback, llm_prediction, response.split("**Reasoning:**")[-1])
                except Exception as e:
                    st.error(f"Error storing LLM prediction in DB: {str(e)}")

                # ✅ Display result
                st.write(f"🔮 {response}")

            except Exception as e:
                st.error(f"❌ LLM Error: {str(e)}")
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
    uploaded_file = st.file_uploader("Upload a cleaned Labled CSV file", type=["csv"])
    
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
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by MLflow, LangChain, and Streamlit. 🚀
    </div>
""", unsafe_allow_html=True)
