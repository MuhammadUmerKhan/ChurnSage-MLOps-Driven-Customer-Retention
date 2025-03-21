import sqlite3
import os

churn_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "churn_predictions.db"))
# ✅ Initialize Database
def init_db():
    """Creates SQLite database if it doesn't exist."""
    conn = sqlite3.connect(churn_db_path)
    cursor = conn.cursor()

    # ✅ Create Table for Customer Features & Model Prediction
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SeniorCitizen TEXT,
            Partner TEXT,
            Dependents TEXT,
            tenure REAL,
            OnlineSecurity TEXT,
            TechSupport TEXT,
            Contract TEXT,
            PaperlessBilling TEXT,
            PaymentMethod TEXT,
            MonthlyCharges REAL,
            TotalCharges REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            Prediction TEXT
        )
    """)

    # ✅ Create Table for LLM Feedback Analysis
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            llm_prediction TEXT,
            llm_reasoning TEXT
        )
    """)

    conn.commit()
    conn.close()

# ✅ Reverse Mapping for Categorical Features
def decode_categorical_values(data):
    """Maps encoded categorical values back to original labels before saving."""
    contract_mapping = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    payment_mapping = {0: "Electronic check", 1: "Mailed check", 2: "Bank transfer (auto)", 3: "Credit card (auto)"}

    data["SeniorCitizen"] = "Yes" if data["SeniorCitizen"] == 1 else "No"
    data["Partner"] = "Yes" if data["Partner"] == 1 else "No"
    data["Dependents"] = "Yes" if data["Dependents"] == 1 else "No"
    data["OnlineSecurity"] = "Yes" if data["OnlineSecurity"] == 1 else "No"
    data["TechSupport"] = "Yes" if data["TechSupport"] == 1 else "No"
    data["PaperlessBilling"] = "Yes" if data["PaperlessBilling"] == 1 else "No"
    data["Contract"] = contract_mapping[data["Contract"]]
    data["PaymentMethod"] = payment_mapping[data["PaymentMethod"]]

    return data

# ✅ Store Customer Data & Churn Prediction
def save_customer_data(data_dict, churn_prediction):
    """Saves customer input features & churn prediction into the database."""
    try:
        conn = sqlite3.connect(churn_db_path)
        cursor = conn.cursor()

        # ✅ Convert numeric encodings back to categorical values
        mapped_data = decode_categorical_values(data_dict)

        cursor.execute("""
            INSERT INTO customer_data (
                SeniorCitizen, Partner, Dependents, tenure, OnlineSecurity, TechSupport, 
                Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Prediction
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mapped_data["SeniorCitizen"], mapped_data["Partner"], mapped_data["Dependents"], 
            mapped_data["tenure"], mapped_data["OnlineSecurity"], mapped_data["TechSupport"],
            mapped_data["Contract"], mapped_data["PaperlessBilling"], mapped_data["PaymentMethod"],
            mapped_data["MonthlyCharges"], mapped_data["TotalCharges"], churn_prediction
        ))

        conn.commit()
        conn.close()
        print("✅ Customer data successfully stored!")

    except Exception as e:
        print(f"❌ Error storing customer data: {e}")

# ✅ Store LLM Feedback & Response
def save_llm_feedback(user_feedback, llm_prediction, llm_reasoning):
    """Saves user feedback & LLM response into the database."""
    try:
        conn = sqlite3.connect(churn_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO llm_feedback (user_feedback, llm_prediction, llm_reasoning) 
            VALUES (?, ?, ?)
        """, (user_feedback, llm_prediction, llm_reasoning))

        conn.commit()
        conn.close()
        print("✅ LLM feedback successfully stored!")

    except Exception as e:
        print(f"❌ Error storing LLM feedback: {e}")

# ✅ Retrieve Customer Data
def get_all_customer_data():
    """Retrieves all customer data stored in the database."""
    try:
        conn = sqlite3.connect(churn_db_path)
        cursor = conn.cursor()

        print("📌 Retrieving all customer records...")

        cursor.execute("SELECT * FROM customer_data")
        records = cursor.fetchall()
        conn.close()

        print(f"✅ Retrieved {len(records)} customer records.")
        return records

    except Exception as e:
        print(f"❌ Error retrieving customer data: {e}")
        return None

# ✅ Retrieve LLM Feedback Data
def get_all_llm_feedback():
    """Retrieves all stored LLM feedback and responses."""
    try:
        conn = sqlite3.connect(churn_db_path)
        cursor = conn.cursor()

        print("📌 Retrieving all LLM feedback records...")

        cursor.execute("SELECT * FROM llm_feedback")
        records = cursor.fetchall()
        conn.close()

        print(f"✅ Retrieved {len(records)} LLM feedback records.")
        return records

    except Exception as e:
        print(f"❌ Error retrieving LLM feedback: {e}")
        return None

# if __name__ == "__main__":
    
    # ✅ Initialize DB on Startup
    # print(init_db())
    # save_data = save_customer_data(
    #     {
    # "SeniorCitizen": 0,
    # "Partner": 0,
    # "Dependents": 0,
    # "tenure": 2,
    # "OnlineSecurity": 0,
    # "TechSupport": 0,
    # "Contract": 0,
    # "PaperlessBilling": 1,
    # "PaymentMethod": 2,
    # "MonthlyCharges": 70.7,
    # "TotalCharges": 151.65
    #     },
    #     1)
    # print(get_all_customer_data())
    # print(get_all_llm_feedback())