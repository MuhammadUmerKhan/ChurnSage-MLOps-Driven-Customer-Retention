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
        /* Global Dark Theme Styles */
        body {
            background: url('https://i.imgur.com/3z8X5xB.png') no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
            color: #d1d5db;
        }
        .main-container {
            background-color: rgba(36, 40, 59, 0.9);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            margin: 20px;
        }
        .main-title {
            font-size: 3em;
            font-weight: 700;
            color: #60a5fa;
            text-align: center;
            margin-bottom: 30px;
            letter-spacing: -0.5px;
            animation: fadeIn 1s ease-in-out;
        }
        .section-title {
            font-size: 2em;
            font-weight: 600;
            color: #f9a8d4;
            margin-top: 40px;
            margin-bottom: 20px;
            text-align: left;
            border-left: 5px solid #f9a8d4;
            padding-left: 15px;
            animation: slideIn 0.5s ease-in-out;
        }
        .content {
            font-size: 1.1em;
            color: #94a3b8;
            line-height: 1.8;
            text-align: justify;
        }
        .stButton>button {
            background: linear-gradient(45deg, #7c3aed, #3b82f6);
            color: #e5e7eb;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            font-size: 1em;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #6d28d9, #2563eb);
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }
        .stSelectbox, .stNumberInput, .stTextArea {
            background-color: rgba(42, 46, 63, 0.95);
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #4b5563;
            color: #d1d5db;
            transition: all 0.3s ease;
        }
        .stSelectbox:hover, .stNumberInput:hover, .stTextArea:hover {
            border-color: #60a5fa;
            box-shadow: 0 0 8px rgba(96, 165, 250, 0.5);
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.2em;
            font-weight: 500;
            color: #94a3b8;
            padding: 15px 25px;
            border-radius: 10px 10px 0 0;
            transition: all 0.3s ease;
            background-color: rgba(31, 34, 48, 0.95);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(45deg, #7c3aed, #3b82f6);
            color: #e5e7eb;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #374151;
            color: #e5e7eb;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            background-color: rgba(42, 46, 63, 0.95);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .stDataFrame table {
            color: #d1d5db;
        }
        .footer {
            font-size: 0.95em;
            color: #9ca3af;
            margin-top: 50px;
            text-align: center;
            padding: 25px;
            background-color: rgba(31, 34, 48, 0.95);
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .footer a {
            color: #60a5fa;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        .footer a:hover {
            color: #f9a8d4;
            text-decoration: underline;
        }
        /* Custom input labels */
        .stSelectbox label, .stNumberInput label, .stTextArea label {
            font-weight: 500;
            color: #60a5fa;
            margin-bottom: 10px;
        }
        /* Error and Success Messages */
        .stAlert {
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            background-color: rgba(47, 49, 66, 0.95);
            color: #f3f4f6;
        }
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        /* Home Page Card Styling */
        .home-card {
            background: linear-gradient(135deg, rgba(42, 46, 63, 0.95), rgba(55, 65, 81, 0.95));
            border-radius: 12px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            animation: fadeIn 1.2s ease-in-out;
        }
        .home-card h2 {
            color: #f9a8d4;
            font-size: 2em;
            margin-bottom: 15px;
        }
        .home-card p {
            color: #d1d5db;
            font-size: 1.1em;
            line-height: 1.8;
        }
        .home-card ul {
            color: #94a3b8;
            font-size: 1.1em;
            line-height: 1.8;
            padding-left: 20px;
        }
        .home-card ul li::marker {
            color: #60a5fa;
        }
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: rgba(31, 34, 48, 0.95);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .css-1d391kg .css-1v3fvcr {
            color: #d1d5db;
        }
    </style>
""", unsafe_allow_html=True)

# Create tabs including the Home tab
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔍 Predict Churn", "💬 LLM Review Analysis", "📊 View Stored Data", "📂 Upload CSV for Bulk Prediction"])

# Home Landing Page
with tab0:
    st.markdown('<div class="main-title">📊 Welcome to the Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="home-card">
            <h2>About This Project</h2>
            <p>
                The Customer Churn Prediction Dashboard is a powerful tool designed to help telecom companies predict customer churn with precision and ease. Leveraging advanced machine learning models and large language models (LLMs), this application provides actionable insights to retain customers and optimize business strategies.
            </p>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>🔍 <strong>Single Customer Prediction:</strong> Input customer details to predict churn likelihood instantly.</li>
                <li>💬 <strong>LLM-Powered Review Analysis:</strong> Analyze customer feedback to gauge sentiment and churn risk.</li>
                <li>📊 <strong>Data Visualization:</strong> View stored predictions and feedback in an intuitive format.</li>
                <li>📂 <strong>Bulk Predictions:</strong> Upload CSV files for batch churn predictions.</li>
            </ul>
            <p>
                Built with MLflow for model management, LangChain for LLM integration, and Streamlit for a seamless user experience, this dashboard is your go-to solution for customer retention analytics.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Feature Input Section
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
                st.error(f"⚠️ Error storing prediction in DB: {str(e)}")

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
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank">Muhammad Umer Khan</a>. Powered by MLflow, LangChain, and Streamlit. 🚀
    </div>
""", unsafe_allow_html=True)