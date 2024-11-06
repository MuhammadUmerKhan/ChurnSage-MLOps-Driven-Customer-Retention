import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib
import os
from spacy.cli import download

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main Title */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Section Titles */
        .section-title {
            font-size: 1.8em;
            color: #3498DB;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        /* Section Content */
        .section-content{
            text-align: center;
        }
        /* Home Page Content */
        .intro-title {
            font-size: 2.5em;
            color: #2C3E50;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #34495E;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            color: #2E86C1;
            font-weight: bold;
        }
        /* Recommendation Titles and Descriptions */
        .recommendation-title {
            font-size: 22px;
            color: #2980B9;
        }
        .recommendation-desc {
            font-size: 16px;
            color: #7F8C8D;
        }
        /* Separator Line */
        .separator {
            margin-top: 10px;
            margin-bottom: 10px;
            border-top: 1px solid #BDC3C7;
        }
        /* Footer */
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="intro-title">📊 Welcome to the Customer Churn Prediction Tool 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">Your one-stop solution for retaining loyal customers! 🚀</div>', unsafe_allow_html=True)

# Define preprocessing function
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)


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

# Load models and scalers
churn_feature_model = joblib.load("./models_joblib_files/Churn_feature_classifier.joblib")
churn_feature_scaler = joblib.load("./models_joblib_files/minmax_scaler_for_churn_prediction.joblib")
churn_sentiment_model = joblib.load("./models_joblib_files/Churn_sentiment_classifier.joblib")
churn_sentiment_scaler = joblib.load("./models_joblib_files/scaler_for_sentiment_analysis.joblib")

# Load the sample data
sample_dataframe = pd.read_csv("./Datasets/Ready_data_for_model.csv")
sample_dataframe = reverser(sample_dataframe)

sample_dataframe_feedback = pd.read_csv("./Datasets/Ready_data_for_model_feedback.csv")
sample_dataframe_feedback = reverser(sample_dataframe_feedback)

churn_class_1 = sample_dataframe[sample_dataframe['Churn'] == 'Yes']
churn_class_0 = sample_dataframe[sample_dataframe['Churn'] == "No"]

churn_feedback_class_1 = sample_dataframe_feedback[sample_dataframe_feedback['Churn'] == 'Yes']
churn_feedback_class_0 = sample_dataframe_feedback[sample_dataframe_feedback['Churn'] == "No"]

if "df_sample_tab1" not in st.session_state:
    churn_class_1_sample = churn_class_1.sample(3)
    churn_class_0_sample = churn_class_0.sample(3)
    st.session_state.df_sample_tab1 = pd.concat([churn_class_1_sample, churn_class_0_sample])

if "df_sample_tab2" not in st.session_state:
    churn_feedback_class_1_sample = churn_feedback_class_1.sample(3)
    churn_feedback_class_0_sample = churn_feedback_class_0.sample(3)
    st.session_state.df_sample_tab2 = pd.concat([churn_feedback_class_1_sample, churn_feedback_class_0_sample])


# Tab layout
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📋 Predict Customer Churn", "✍️ Provide Feedback"])

# First Tab: Home
with tab1:
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">Hi! I\'m Muhammad Umer Khan, an aspiring Data Scientist with a passion for Natural Language Processing (NLP). Currently pursuing my Bachelor’s in Computer Science, I have hands-on experience with projects in data science, data scraping, and building intelligent recommendation systems.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project focuses on creating a comprehensive <span class="highlight">Churn Prediction</span> using advanced NLP techniques. Here’s what we achieved:
            <ul>
                <li><span class="highlight">Data Collection 📊</span>: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank" style="color: #2980B9;">Telecom Customer Dataset</a></li>
                <li><span class="highlight">Feature Based Prediction 🔍</span>: Utilized features like tenure, Monthly Charges, Feedbacks etc., to predict customer churn</li>
                <li><span class="highlight">Fine Tuning 📈</span>: Planned for further development to enhance prediction accuracy.</li>
                <li><span class="highlight">Deployment 🌐</span>: Built a user-friendly app with an intuitive interface for customer prediction.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies Used</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            - <span class="highlight">Languages & Libraries</span>: Python, Pandas, Scikit-Learn, Spacy, Word Embeddings, and Streamlit<br>
            - <span class="highlight">Deployment</span>: Streamlit for interactive interface and easy deployment.
        </div>
    """, unsafe_allow_html=True)
# Second Tab: Churn Prediction
with tab2:
    st.header("Predict Customer Churn 🔍")

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
    if st.button("Predict Churn 🚀"):
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
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = churn_feature_scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
            prediction = churn_feature_model.predict(input_df)
            prediction_prob = churn_feature_model.predict_proba(input_df)[0][1]

            # Display result
            if prediction[0] == 1:
                st.write("Prediction:", "😢 Customer may leave")
                st.markdown(f"Probability: **{prediction_prob*100:.2f}%**")
            else:
                st.write("Prediction:", "😊 Customer likely to stay")
                st.markdown(f"Probability: **{(1 - prediction_prob)*100:.2f}%**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    # Display and refresh sample DataFrame for churn classes
    sample_display = st.empty()
    sample_display.dataframe(st.session_state.df_sample_tab1)

    if st.button("Refresh Sample"):
        churn_class_1_sample = churn_class_1.sample(3)
        churn_class_0_sample = churn_class_0.sample(3)
        st.session_state.df_sample_tab1 = pd.concat([churn_class_1_sample, churn_class_0_sample])
        sample_display.dataframe(st.session_state.df_sample_tab1)

# Third Tab: Provide Feedback
with tab3:
    st.header("Analyze Customer Feedback 📝")
    
    feedback = st.text_area("Enter your feedback here", key="feedback")
    
    # Organize Tenure and Monthly Charges in a row
    col1, col2 = st.columns([3, 1])
    with col1:
        feedback_tenure = st.number_input("Tenure", value=0, key="fb_tenure", format="%d")
    with col2:
        feedback_monthly_charges = st.number_input("Monthly Charges", value=0.0, key="fb_mc", format="%.2f")
        
    if st.button("Analyze Feedback 📊"):
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

                feedback_prediction = churn_sentiment_model.predict(sentiment_input_scaled)
                feedback_prob = churn_sentiment_model.predict_proba(sentiment_input_scaled)[0][1]\
                    
                if feedback_prediction[0] == 1:
                    st.write("Prediction:", "😡 Negative Feedback")
                    st.markdown(f"Probability: **{feedback_prob*100:.2f}%**")
                else:
                    st.write("Prediction:", "😊 Positive Feedback")
                    st.markdown(f"Probability: **{(1 - feedback_prob)*100:.2f}%**")
        
        except ValueError as ve:
            st.error(f"Error: {str(ve)}. Please enter valid numeric values for Tenure and Monthly Charges.")
            
    feedback_display = st.empty()
    feedback_display.dataframe(st.session_state.df_sample_tab2)

    if st.button("Refresh Feedback Sample"):
        churn_feedback_class_1_sample = churn_feedback_class_1.sample(3)
        churn_feedback_class_0_sample = churn_feedback_class_0.sample(3)
        st.session_state.df_sample_tab2 = pd.concat([churn_feedback_class_1_sample, churn_feedback_class_0_sample])
        feedback_display.dataframe(st.session_state.df_sample_tab2)
        
        
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by SpaCy and Random Forest Classifier. 🌐
    </div>
""", unsafe_allow_html=True)