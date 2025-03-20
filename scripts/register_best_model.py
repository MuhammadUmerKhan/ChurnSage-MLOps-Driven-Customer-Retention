import mlflow
from mlflow.tracking import MlflowClient
from config import mlflow_db_path

def get_best_model(experiment_name: str):
    """Finds the best model based on the highest F1-score and prints detailed logs."""
    try:
        print(f"🔍 Searching for the best model in experiment: {experiment_name}")

        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ValueError(f"❌ Experiment '{experiment_name}' not found!")

        print(f"✅ Found Experiment '{experiment_name}' (ID: {experiment.experiment_id})")

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_score DESC"])

        if runs.empty:
            raise ValueError("❌ No runs found for this experiment!")

        best_run_id = runs.iloc[0]["run_id"]
        print(f"🏆 Best model found! Run ID: {best_run_id}")

        return best_run_id, experiment.experiment_id

    except Exception as e:
        print(f"❌ Error while retrieving the best model: {e}")
        return None, None

def register_model(model_name: str, best_run_id: str, experiment_id: str):
    """Registers the best model in MLflow Model Registry with detailed logs."""
    try:
        if not best_run_id:
            print("⚠️ No valid model found. Skipping registration.")
            return

        print(f"📌 Registering the best model as '{model_name}'...")

        client = MlflowClient()
        model_uri = f"runs:/{best_run_id}/models"

        try:
            registered_model = client.create_registered_model(model_name)
            print(f"✅ Registered new model: {model_name}")
        except:
            print(f"⚠️ Model '{model_name}' already exists. Continuing...")

        model_version = client.create_model_version(name=model_name, source=model_uri, run_id=best_run_id)
        print(f"✅ Model '{model_name}' registered as version {model_version.version}")

        # Add metadata tags
        print("📌 Adding metadata tags...")
        client.set_registered_model_tag(model_name, "dataset", "Customer Churn Dataset")
        client.set_registered_model_tag(model_name, "training_method", "GridSearchCV")
        client.set_registered_model_tag(model_name, "author", "Muhammad Umer Khan")
        print("✅ Metadata tags added!")

        # Add description
        client.update_registered_model(
            name=model_name,
            description="Customer Churn Prediction Model trained using multiple classifiers. Registered after hyperparameter tuning & evaluation."
        )
        print(f"✅ Model '{model_name}' description updated!")

        # Set model aliases
        print("📌 Assigning model aliases...")
        client.set_registered_model_alias(model_name, "staging", model_version.version)
        print(f"✅ Model '{model_name}' set to 'staging'.")

        client.set_registered_model_alias(model_name, "production", model_version.version)
        print(f"✅ Model '{model_name}' set to 'production'.")

    except Exception as e:
        print(f"❌ Error during model registration: {e}")

if __name__ == "__main__":
    experiment_name = "Customer Churn Prediction (Exp 2)"
    model_name = "customer_churn_model"

    print("\n🚀 Starting Model Registration Pipeline...\n")

    best_run_id, experiment_id = get_best_model(experiment_name)
    register_model(model_name, best_run_id, experiment_id)

    print("\n🎯 Model registration process completed successfully!\n")
