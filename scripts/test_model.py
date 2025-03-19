import mlflow.pyfunc
import pandas as pd
import os

def load_production_model(model_name: str):
    """Loads the latest production model from MLflow with error handling."""
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print(f"🔍 Attempting to load the latest production model: {model_name}...")

        loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
        print(f"✅ Model '{model_name}' loaded successfully!")

        return loaded_model

    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        return None

def make_prediction(model, data_path: str):
    """Loads test data and makes predictions using the trained model."""
    try:
        print(f"📂 Loading test data from: {data_path}")

        # Load test data
        X_test_path = os.path.join(data_path, "X_test.csv")
        y_test_path = os.path.join(data_path, "y_test.csv")

        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            raise FileNotFoundError(f"❌ Missing test data files in {data_path}")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        print("\n📝 Sample Data for Prediction:")
        print(X_test.iloc[:1], y_test.iloc[:1])

        # Make prediction
        print("🔮 Making prediction...")
        prediction = model.predict(X_test.iloc[:1])
        print(f"✅ Model Prediction: {prediction}")

    except FileNotFoundError as fnf_error:
        print(f"❌ File Error: {fnf_error}")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    # Define paths
    model_name = "customer_churn_model"
    model_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ModelData"))

    # Load and test the model
    print("\n🚀 Starting Model Testing Pipeline...")
    model = load_production_model(model_name)

    if model:
        make_prediction(model, model_data_path)
    else:
        print("⚠️ Skipping prediction due to model loading failure.")

    print("🎯 Model testing process completed!")
