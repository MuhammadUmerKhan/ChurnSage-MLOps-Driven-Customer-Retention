import pandas as pd
import joblib
from configs import scaler_path
import os

def preprocess_input(input_df):
    """🔄 Preprocesses input data before model prediction."""
    try:
        print("📌 Preprocessing input data for prediction...")

        # ✅ Ensure input is a DataFrame
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame([input_df])

        # ✅ Convert numerical columns to float (Fix `astype` issue)
        columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in columns_to_scale:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").astype(float)

        # ✅ Apply MinMaxScaler
        scaler = joblib.load(scaler_path)
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
        
        print("✅ Input data successfully preprocessed!")
        return input_df

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        raise e

def decode_categorical_values(data):
    """Maps encoded categorical values back to original labels before saving."""
    contract_mapping = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    payment_mapping = {0: "Electronic check", 1: "Mailed check", 2: "Bank transfer (auto)", 3: "Credit card (auto)"}

    return {
        "SeniorCitizen": "Yes" if data["SeniorCitizen"] == 1 else "No",
        "Partner": "Yes" if data["Partner"] == 1 else "No",
        "Dependents": "Yes" if data["Dependents"] == 1 else "No",
        "OnlineSecurity": "Yes" if data["OnlineSecurity"] == 1 else "No",
        "TechSupport": "Yes" if data["TechSupport"] == 1 else "No",
        "PaperlessBilling": "Yes" if data["PaperlessBilling"] == 1 else "No",
        "Contract": contract_mapping.get(data["Contract"], "Unknown"),
        "PaymentMethod": payment_mapping.get(data["PaymentMethod"], "Unknown"),
        "tenure": data["tenure"],
        "MonthlyCharges": data["MonthlyCharges"],
        "TotalCharges": data["TotalCharges"]
    }

def log_llm_response(user_feedback, llm_prediction, llm_reasoning):
    """Logs the LLM response for debugging and database storage."""
    try:
        print(f"📌 User Feedback: {user_feedback}")
        print(f"🔮 LLM Prediction: {llm_prediction}")
        print(f"📝 LLM Reasoning: {llm_reasoning}")
        
        # ✅ Store in database
        import database
        database.save_llm_feedback(user_feedback, llm_prediction, llm_reasoning)
        
    except Exception as e:
        print(f"❌ Error logging LLM response: {e}")
