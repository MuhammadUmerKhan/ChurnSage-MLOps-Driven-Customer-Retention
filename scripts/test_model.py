import mlflow.pyfunc
import pandas as pd
import os

def load_production_model(model_name: str):
    """Loads the latest production model from MLflow."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
    print(f"✅ Model '{model_name}' loaded successfully!")
    return loaded_model

def make_prediction(model, data_path: str):
    """Loads test data and makes predictions using the trained model."""
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")
    
    print("📝 Sample Data for Prediction:")
    print(X_test.iloc[:1], y_test.iloc[:1])
    
    prediction = model.predict(X_test.iloc[:1])
    print(f"🔮 Model Prediction: {prediction}")

if __name__ == "__main__":
    # Define paths
    model_name = "customer_churn_model"
    model_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ModelData"))

    # Load and test the model
    model = load_production_model(model_name)
    make_prediction(model, model_data_path)
