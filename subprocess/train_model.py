import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from hyperparams import logistic_params, decision_tree_params, random_forest_params, knn_params
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import data_preprocessing
import warnings
warnings.filterwarnings('ignore')

# ✅ Set MLflow tracking URI (Make sure it points to mlflow.db)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer Churn Prediction (Exp 2)")

def train_and_track_model(model, X_train, y_train, X_test, y_test, params, model_name):
    with mlflow.start_run(run_name=model_name):  
        # Convert y_train and y_test to 1D arrays
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # Perform hyperparameter tuning
        gridsearch = GridSearchCV(model, param_grid=params, scoring='f1', cv=5)
        gridsearch.fit(X_train, y_train)

        # Get the best model and hyperparameters
        best_model = gridsearch.best_estimator_
        best_params = gridsearch.best_params_

        # Get model predictions
        y_pred = best_model.predict(X_test)
        y_scores = best_model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)

        # Log hyperparameters & metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)

        # Log additional metadata
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset_used", "Customer Churn Dataset")
        mlflow.set_tag("tracking_method", "SQLite Database")

        # ✅ Ensure "models" directory exists
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)  # ✅ Create only if it doesn’t exist

        # ✅ Save hyperparameters in JSON format
        params_path = os.path.join(models_dir, f"{model_name}_params.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        mlflow.log_artifact(params_path, artifact_path="models")

        # ✅ Save dataset details
        dataset_info = {
            "num_samples": len(X_train) + len(X_test),
            "num_features": X_train.shape[1],
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        dataset_path = os.path.join(models_dir, f"{model_name}_dataset_info.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset_info, f, indent=4)
        mlflow.log_artifact(dataset_path, artifact_path="models")

        # ✅ Log model inside MLflow’s managed storage
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="models",
            input_example=X_train.iloc[:1]  # ✅ Fixes missing input schema warning
        )

        print(f"✅ Model '{model_name}' logged to MLflow with Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        return best_model, best_params

if __name__ == "__main__":
    # ✅ Run preprocessing first
    X_train, X_test, y_train, y_test = data_preprocessing.run_preprocessing_pipeline()

    # ✅ Train and track different models
    train_and_track_model(LogisticRegression(), X_train, y_train, X_test, y_test, logistic_params, "Logistic Regression Classifier")
    train_and_track_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, decision_tree_params, "Decision Tree Classifier")
    train_and_track_model(RandomForestClassifier(), X_train, y_train, X_test, y_test, random_forest_params, "Random Forest Classifier")
    train_and_track_model(KNeighborsClassifier(), X_train, y_train, X_test, y_test, knn_params, "KNN Classifier")
