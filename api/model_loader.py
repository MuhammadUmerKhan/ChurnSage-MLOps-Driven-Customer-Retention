import os
import mlflow.pyfunc
import langchain_groq
from langchain.schema import HumanMessage
from functools import lru_cache
from config import GROK_API_KEY, mlflow_db_path

# ✅ Set up LLM API (ChatGroq)

llm = langchain_groq.ChatGroq(groq_api_key=GROK_API_KEY, model_name="qwen-2.5-32b")

def load_production_model():
    """🎯 Loads the latest production model from MLflow."""
    try:
        # ✅ Check if mlflow.db exists
        db_path = mlflow_db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"⚠️ MLflow database not found at: {db_path}")

        print(f"🔍 Using MLflow database at: {db_path}")

        # ✅ Set tracking URI
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        # ✅ Load the production model
        model_name = "customer_churn_model"
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

        print(f"✅ Successfully loaded production model: {model_name}")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e

# ✅ Cache LLM Responses to avoid redundant API calls
@lru_cache(maxsize=50)
def predict_churn_with_llm(user_feedback: str):
    """Sends user feedback to the LLM and returns a churn prediction."""
    if not user_feedback:
        return "❌ Please enter some feedback!"

    # ✅ Optimized LLM Prompt
    prompt = f"""
        You are a **Telecom Customer Retention Expert**. Analyze the following customer review and determine churn risk.

        📌 **Customer Review:** "{user_feedback}"

        🎯 **Your Task:**
        - Analyze sentiment & concerns.
        - Predict churn risk (**High Risk or Low Risk**).
        - Give a short but engaging explanation.

        📌 **Format Response:**
        - **Prediction:** ("Customer likely to leave" / "Customer will stay")
        - **Reasoning:** Short explanation (e.g., "The customer is unhappy with billing issues and mentions switching providers, which indicates a high churn risk.")

        🚀 **Make the response professional & engaging!**
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()