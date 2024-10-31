import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib

# Apply custom CSS for theme
st.markdown(
    """
    <style>
        /* Set background gradient */
        .main {
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            font-family: 'Roboto', sans-serif;
        }

        /* Customize title and headers */
        .stTitle {
            color: #005662;
            font-weight: bold;
            text-align: center;
            font-size: 2.5em;
            padding-bottom: 10px;
        }
        
        .stHeader {
            color: #005662;
            font-weight: bold;
            font-size: 1.5em;
            margin-top: 20px;
        }

        /* Style input boxes */
        .stTextInput > div > input {
            border: 2px solid #005662;
            border-radius: 10px;
            padding: 8px;
        }

        /* Customize buttons with hover animation */
        .stButton > button {
            background-color: #5CDB95;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            transition: background-color 0.3s ease;
            border: none;
            padding: 10px;
        }
        
        .stButton > button:hover {
            background-color: #379683;
        }

        /* Tabs styling */
        .stTabs .stTab {
            background-color: #edf5e1;
            border: 2px solid #005662;
            border-radius: 10px;
            padding: 5px;
            margin-top: 10px;
        }
        
        .stTab:hover {
            background-color: #8ee4af;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Customer Churn Prediction with NLP Insights")

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Define preprocessing function
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

# Load models and scalers
churn_feature_model = joblib.load("./model/Churn_Feature_Classifier.joblib")
churn_feature_scaler = joblib.load("./model/minmax_scaler_for_churn_prediction.joblib")

churn_sentiment_model = joblib.load("./model/Churn_Sentiment_Classifier.joblib")
churn_sentiment_scaler = joblib.load("./model/scaler_for_sentiment_analysis.joblib")

# Tabs for different sections
tab1, tab2 = st.tabs(["Predict Customer Churn", "Feedback Section"])

# First Tab: Customer Input for Prediction
with tab1:
    st.header("Predict Customer Churn")

    # Organize inputs into columns
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_charges = st.text_input("Monthly Charges", "0", key="mc")
        senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"], key="sc")
        senior_citizen = 1 if senior_citizen == "Yes" else 0

    with col2:
        tenure = st.text_input("Tenure (months)", "0", key="tenure")
        partner = st.selectbox("Partner", options=["No", "Yes"], key="partner")
        partner = 1 if partner == "Yes" else 0

    with col3:
        total_charges = st.text_input("Total Charges", "0", key="tc")
        dependents = st.selectbox("Dependents", options=["No", "Yes"], key="dep")
        dependents = 1 if dependents == "Yes" else 0

    # Additional input fields
    col4, col5, col6 = st.columns(3)
    with col4:
        online_security = st.selectbox("Online Security", options=["No", "Yes"], key="os")
    with col5:
        tech_support = st.selectbox("Tech Support", options=["No", "Yes"], key="ts")
    with col6:
        paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"], key="pb")

    col7, col8 = st.columns(2)
    with col7:
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], key="contract")
        contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
    with col8:
        payment_method = st.selectbox("Payment Method", options=[
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="pm")

    # Predict button
    if st.button("Predict"):
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
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = churn_feature_scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )
        prediction = churn_feature_model.predict(input_df.values)
        st.write("Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")

# Second Tab: Feedback Section
with tab2:
    st.header("Provide Feedback")
    feedback = st.text_area("Enter your feedback here", key="feedback")
    feedback_tenure = st.text_input("Tenure", "0", key="fb_tenure")
    feedback_monthly_charges = st.text_input("Monthly Charges", "0", key="fb_mc")

    if st.button("Submit Feedback"):
        cleaned_feedback = preprocess(feedback)
        feedback_vector = nlp(cleaned_feedback).vector
        feedback_data = np.array([[*feedback_vector, int(feedback_tenure), float(feedback_monthly_charges)]])
        sentiment_input_scaled = churn_sentiment_scaler.transform(feedback_data)
        prediction = churn_sentiment_model.predict(sentiment_input_scaled)
        st.write("Feedback Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")
