import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib
from spacy.cli import download

# Try loading the 'en_core_web_sm' model, or download if it's not available
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
churn_feature_model = joblib.load("./models/Churn_feature_classifier.joblib")
churn_feature_scaler = joblib.load("./models/minmax_scaler_for_churn_prediction.joblib")
churn_sentiment_model = joblib.load("./models/Churn_sentiment_classifier.joblib")
churn_sentiment_scaler = joblib.load("./models/scaler_for_sentiment_analysis.joblib")

# Define the reverser function to transform categorical columns
def reverser(data):
    columns_len_2 = ['SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'Churn']
    
    # Transform binary columns
    for col in columns_len_2:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 'No' if x == 0 else 'Yes')
    
    # Map Contract and PaymentMethod columns
    if 'Contract' in data.columns:
        data['Contract'] = data['Contract'].map({0: "Month-to-month", 1: "One year", 2: "Two year"})
    
    if 'PaymentMethod' in data.columns:
        data['PaymentMethod'] = data['PaymentMethod'].map({
            0: "Electronic check",
            1: "Mailed check",
            2: "Bank transfer (automatic)",
            3: "Credit card (automatic)"
        })
    
    return data

# Load the sample data
sample_dataframe = pd.read_csv("./Datasets/Ready_data_for_model.csv")
sample_dataframe = reverser(sample_dataframe)

sample_dataframe_feedback = pd.read_csv("./Datasets/Ready_data_for_model_feedback.csv")
sample_dataframe_feedback = reverser(sample_dataframe_feedback)

churn_class_1 = sample_dataframe[sample_dataframe['Churn'] == 'Yes']
churn_class_0 = sample_dataframe[sample_dataframe['Churn'] == "No"]

churn_feedback_class_1 = sample_dataframe_feedback[sample_dataframe_feedback['Churn'] == 'Yes']
churn_feedback_class_0 = sample_dataframe_feedback[sample_dataframe_feedback['Churn'] == "No"]

# Initialize session state for sample data in both tabs
if "df_sample_tab1" not in st.session_state:
    churn_class_1_sample = churn_class_1.sample(3)
    churn_class_0_sample = churn_class_0.sample(3)
    st.session_state.df_sample_tab1 = pd.concat([churn_class_1_sample, churn_class_0_sample])

if "df_sample_tab2" not in st.session_state:
    churn_feedback_class_1_sample = churn_feedback_class_1.sample(3)
    churn_feedback_class_0_sample = churn_feedback_class_0.sample(3)
    st.session_state.df_sample_tab2 = pd.concat([churn_feedback_class_1_sample, churn_feedback_class_0_sample])

# Tab layout
tab1, tab2 = st.tabs(["Predict Customer Churn", "Provide Feedback"])

# First Tab: Churn Prediction
with tab1:
    st.header("Predict Customer Churn")

    # Customer input fields in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_charges = st.number_input("Monthly Charges", value=0.0, key="mc", format="%.2f")
        senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"], key="sc")
        senior_citizen = 1 if senior_citizen == "Yes" else 0

    with col2:
        tenure = st.number_input("Tenure (months)", value=0, key="tenure", format="%d")
        partner = st.selectbox("Partner", options=["No", "Yes"], key="partner")
        partner = 1 if partner == "Yes" else 0

    with col3:
        total_charges = st.number_input("Total Charges", value=0.0, key="tc", format="%.2f")
        dependents = st.selectbox("Dependents", options=["No", "Yes"], key="dep")
        dependents = 1 if dependents == "Yes" else 0

    # Additional input fields
    col4, col5, col6 = st.columns(3)
    with col4:
        online_security = st.selectbox("Online Security", options=["No", "Yes"], key="os")
        online_security = 1 if online_security == "Yes" else 0
    with col5:
        tech_support = st.selectbox("Tech Support", options=["No", "Yes"], key="ts")
        tech_support = 1 if tech_support == "Yes" else 0
    with col6:
        paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"], key="pb")
        paperless_billing = 1 if paperless_billing == "Yes" else 0

    col7, col8 = st.columns(2)
    with col7:
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], key="contract")
        contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
    with col8:
        payment_method = st.selectbox("Payment Method", options=[
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="pm")
        payment_method = {
            "Electronic check": 0, 
            "Mailed check": 1, 
            "Bank transfer (automatic)": 2, 
            "Credit card (automatic)": 3
        }[payment_method]

    # Predict button
    if st.button("Predict"):
        try:
            # Prepare input data
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
            
            # Scale and predict
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = churn_feature_scaler.transform(
                input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
            )
            prediction = churn_feature_model.predict(input_df.values)
            st.write("Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")
        
        except ValueError as ve:
            st.error(f"Error: {str(ve)}. Please enter valid numeric values for Monthly Charges, Tenure, and Total Charges.")

    # Display and refresh sample DataFrame for churn classes
    sample_display = st.empty()
    sample_display.dataframe(st.session_state.df_sample_tab1)

    if st.button("Refresh Sample"):
        churn_class_1_sample = churn_class_1.sample(3)
        churn_class_0_sample = churn_class_0.sample(3)
        st.session_state.df_sample_tab1 = pd.concat([churn_class_1_sample, churn_class_0_sample])
        sample_display.dataframe(st.session_state.df_sample_tab1)

# Second Tab: Feedback Section
with tab2:
    st.header("Provide Feedback")
    feedback = st.text_area("Enter your feedback here", key="feedback")
    feedback_tenure = st.number_input("Tenure", value=0, key="fb_tenure", format="%d")
    feedback_monthly_charges = st.number_input("Monthly Charges", value=0.0, key="fb_mc", format="%.2f")

    if st.button("Submit Feedback", key="submit_feedback"):
        try:
            if not feedback:
                st.error("Feedback cannot be empty.")
            else:
                cleaned_feedback = preprocess(feedback)
                feedback_vector = nlp(cleaned_feedback).vector
                
                feedback_tenure = float(feedback_tenure)
                feedback_monthly_charges = float(feedback_monthly_charges)

                feedback_data = np.array([[*feedback_vector, feedback_tenure, feedback_monthly_charges]])
                sentiment_input_scaled = churn_sentiment_scaler.transform(feedback_data)
                
                prediction = churn_sentiment_model.predict(sentiment_input_scaled)
                st.write("Feedback Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")
        
        except ValueError as ve:
            st.error(f"Error: {str(ve)}. Please enter valid numeric values for Tenure and Monthly Charges.")

    # Display and refresh sample DataFrame for feedback classes
    feedback_display = st.empty()
    feedback_display.dataframe(st.session_state.df_sample_tab2)

    if st.button("Refresh Feedback Sample"):
        churn_feedback_class_1_sample = churn_feedback_class_1.sample(3)
        churn_feedback_class_0_sample = churn_feedback_class_0.sample(3)
        st.session_state.df_sample_tab2 = pd.concat([churn_feedback_class_1_sample, churn_feedback_class_0_sample])
        feedback_display.dataframe(st.session_state.df_sample_tab2)