import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

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

# Tabs for different sections
tab1, tab2 = st.tabs(["Predict Customer Churn", "Feedback Section"])

# First Tab: Predict Customer Churn Section
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
            # Convert inputs to correct numeric format
            monthly_charges = float(monthly_charges)
            tenure = float(tenure)
            total_charges = float(total_charges)
            
            # Prepare the input data
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
            
            # Scale numeric data
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = churn_feature_scaler.transform(
                input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
            )
            
            # Predict churn
            prediction = churn_feature_model.predict(input_df.values)
            st.write("Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")
        
        except ValueError:
            st.error("Please enter valid numeric values for Monthly Charges, Tenure, and Total Charges.")
    
    def reverser(data):
        # Define the columns based on their encoding needs
        columns_len_2 = ['SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'Churn']
        columns_len_3 = ['Contract', 'PaymentMethod']
        
        # Apply transformations for columns with binary values
        for col in columns_len_2:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: 'No' if x == 0 else 'Yes')
        
        # Apply specific mappings for Contract and PaymentMethod columns
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

    # Load sample dataframe
    sample_dataframe = pd.read_csv("./Datasets/Ready_data_for_model.csv")

    # Apply reverser function
    sample_dataframe = reverser(sample_dataframe)

    # Initialize session state for this tab's DataFrame if it doesn’t already exist
    if "df_sample_tab1" not in st.session_state:
        st.session_state.df_sample_tab1 = sample_dataframe.sample(5)

    # Display the DataFrame
    st.dataframe(st.session_state.df_sample_tab1)

    # Refresh button below the DataFrame with a unique key
    if st.button("Refresh Sample (Tab 1)", key="refresh_sample_tab1"):
        # Resample 5 rows and update session state
        st.session_state.df_sample_tab1 = sample_dataframe.sample(5)


# Second Tab: Feedback Section
with tab2:
    st.header("Provide Feedback")
    feedback = st.text_area("Enter your feedback here", key="feedback")
    feedback_tenure = st.text_input("Tenure", "0", key="fb_tenure")
    feedback_monthly_charges = st.text_input("Monthly Charges", "0", key="fb_mc")

    if st.button("Submit Feedback", key="submit_feedback"):
        try:
            # Process and transform feedback data
            cleaned_feedback = preprocess(feedback)
            feedback_vector = nlp(cleaned_feedback).vector
            feedback_tenure = float(feedback_tenure)
            feedback_monthly_charges = float(feedback_monthly_charges)

            feedback_data = np.array([[*feedback_vector, feedback_tenure, feedback_monthly_charges]])
            sentiment_input_scaled = churn_sentiment_scaler.transform(feedback_data)
            
            # Predict feedback sentiment
            prediction = churn_sentiment_model.predict(sentiment_input_scaled)
            st.write("Feedback Prediction:", "😢 Customer may leave" if prediction[0] == 1 else "😊 Customer likely to stay")
        
        except ValueError:
            st.error("Please enter valid numeric values for Tenure and Monthly Charges.")

    # Load sample DataFrame for feedback section
    sample_dataframe_feedback = pd.read_csv("./Datasets/Ready_data_for_model_feedback.csv")

    # Initialize session state for this tab's DataFrame if it doesn’t already exist
    if "df_sample_tab2" not in st.session_state:
        st.session_state.df_sample_tab2 = sample_dataframe_feedback.sample(5)

    # Display the DataFrame
    st.dataframe(st.session_state.df_sample_tab2)

    # Refresh button below the DataFrame with a unique key
    if st.button("Refresh Sample (Tab 2)", key="refresh_sample_tab2"):
        # Resample 5 rows and update session state
        st.session_state.df_sample_tab2 = sample_dataframe_feedback.sample(5)