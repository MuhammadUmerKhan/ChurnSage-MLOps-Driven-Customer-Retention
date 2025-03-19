import mlflow
from mlflow.tracking import MlflowClient

# ✅ Set the correct tracking URI (Point to your SQLite DB)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ✅ Get the experiment
experiment_name = "Customer Churn Prediction (Exp 2)"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found!")

# ✅ Find the best model (highest F1-score)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_score DESC"])

if runs.empty:
    raise ValueError("No runs found for this experiment!")

best_run_id = runs.iloc[0]["run_id"]
print(f"✅ Best model run ID: {best_run_id}")

# ✅ Define model name
model_name = "customer_churn_model"

# ✅ Get the model URI from the best run
model_uri = f"runs:/{best_run_id}/models"

# ✅ Create an MLflow client
client = MlflowClient()

# ✅ Register the model
try:
    registered_model = client.create_registered_model(model_name)
    print(f"✅ Registered new model: {model_name}")
except:
    print(f"⚠️ Model '{model_name}' already exists. Continuing...")

# ✅ Create a new model version
model_version = client.create_model_version(name=model_name, source=model_uri, run_id=best_run_id)
print(f"✅ Model '{model_name}' registered as version {model_version.version}")

# ✅ Add metadata tags
client.set_registered_model_tag(model_name, "dataset", "Customer Churn Dataset")
client.set_registered_model_tag(model_name, "training_method", "GridSearchCV")
client.set_registered_model_tag(model_name, "author", "Muhammad Umer Khan")

# ✅ Add a model description
client.update_registered_model(
    name=model_name,
    description="Customer Churn Prediction Model trained using multiple classifiers. Registered after hyperparameter tuning & evaluation."
)

print(f"✅ Model '{model_name}' metadata updated successfully.")

# ✅ Set model alias to 'staging'
client.set_registered_model_alias(model_name, "staging", model_version.version)
print(f"✅ Model '{model_name}' set to 'staging'.")

# ✅ Move model to 'production' if it's the best-performing model
client.set_registered_model_alias(model_name, "production", model_version.version)
print(f"✅ Model '{model_name}' set to 'production'.")