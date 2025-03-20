import os
import dotenv
dotenv.load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

churn_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "churn_predictions.db"))
mlflow_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "mlflow.db"))
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_joblib_files", "scaler.pkl"))