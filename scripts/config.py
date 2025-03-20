import os
import dotenv
dotenv.load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "TelecomChustomerChurnDataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
preprocess_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ProcessedData"))
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl"))
mlflow_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "mlflow.db"))
model_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ModelData"))