import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib
from spacy.cli import download

# Load or download SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.title("Customer Churn Prediction")

# Define preprocessing function
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

# Load models and scalers
churn_feature_model = joblib.load("./models/Churn_Feature_Classifier.joblib")
churn_feature_scaler = joblib.load("./models/minmax_scaler_for_churn_prediction.joblib")
churn_sentiment_model = joblib.load("./models/Churn_Sentiment_Classifier.joblib")
churn_sentiment_scaler = joblib.load("./models/scaler_for_sentiment_analysis.joblib")

# Define function to reverse categorical columns
def reverser(data):
    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 'No' if x == 0 else 'Yes')
    contract_mapping = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    payment_mapping = {0: "Electronic check", 1: "Mailed check", 2: "Bank transfer (automatic)", 3: "Credit card (automatic)"}
    data['Contract'] = data['Contract'].map(contract_mapping)
    data['PaymentMethod'] = data['PaymentMethod'].map(payment_mapping)
    return data

# Load sample data
sample_dataframe = pd.read_csv("./Datasets/Ready_data_for_model.csv")
sample_dataframe = reverser(sample_dataframe)
sample_dataframe_feedback = pd.read_csv("./Datasets/Ready_data_for_model_feedback.csv")
sample_dataframe_feedback = reverser(sample_dataframe_feedback)

# Initialize sample data in session state
if "df_sample_tab1" not in st.session_state:
    st.session_state.df_sample_tab1 = pd.concat([sample_dataframe[sample_dataframe['Churn'] == 'Yes'].sample(3),
                                                 sample_dataframe[sample_dataframe['Churn'] == "No"].sample(3)])

if "df_sample_tab2" not in st.session_state:
    st.session_state.df_sample_tab2 = pd.concat([sample_dataframe_feedback[sample_dataframe_feedback['Churn'] == 'Yes'].sample(3),
                                                 sample_dataframe_feedback[sample_dataframe_feedback['Churn'] == "No"].sample(3)])

# Tab layout
tab1, tab2 = st.tabs(["Predict Customer Churn", "Provide Feedback"])

# Tab 1: Prediction
with tab1:
    st.header("Predict Customer Churn")
    col1, col2, col3 = st.columns(3)
    monthly_charges = float(st.text_input("Monthly Charges", "0", key="mc"))
    tenure = float(st.text_input("Tenure (months)", "0", key="tenure"))
    total_charges = float(st.text_input("Total Charges", "0", key="tc"))

    senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"]) == "Yes"
    partner = st.selectbox("Partner", options=["No", "Yes"]) == "Yes"
    dependents = st.selectbox("Dependents", options=["No", "Yes"]) == "Yes"
    online_security = st.selectbox("Online Security", options=["No", "Yes"]) == "Yes"
    tech_support = st.selectbox("Tech Support", options=["No", "Yes"]) == "Yes"
    paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"]) == "Yes"
    contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Predict button
    if st.button("Predict"):
        input_data = [[senior_citizen, partner, dependents, tenure, online_security, tech_support, contract, paperless_billing, payment_method, monthly_charges, total_charges]]
        input_df = pd.DataFrame(input_data, columns=['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity', 'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = churn_feature_scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        prediction = churn_feature_model.predict(input_df.values)
        st.write("Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")

    st.write(st.session_state.df_sample_tab1)

# Tab 2: Feedback
with tab2:
    st.header("Provide Feedback")
    feedback = st.text_area("Enter your feedback here", key="feedback")
    feedback_tenure = float(st.text_input("Tenure", "0", key="fb_tenure"))
    feedback_monthly_charges = float(st.text_input("Monthly Charges", "0", key="fb_mc"))

    if st.button("Submit Feedback"):
        cleaned_feedback = preprocess(feedback)
        feedback_vector = nlp(cleaned_feedback).vector
        feedback_data = np.array([[*feedback_vector, feedback_tenure, feedback_monthly_charges]])
        sentiment_input_scaled = churn_sentiment_scaler.transform(feedback_data)
        prediction = churn_sentiment_model.predict(sentiment_input_scaled)
        st.write("Feedback Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")

    st.write(st.session_state.df_sample_tab2)