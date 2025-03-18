import os
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

# ✅ Use SQLite for MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer Churn Prediction (Exp 2)")

def train_and_track_model(model, X_train, y_train, X_test, y_test, params, model_name):
    """Train a model, log it to MLflow, and track metrics."""
    
    with mlflow.start_run(run_name=model_name):  
        # ✅ Convert y_train and y_test to 1D arrays
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # ✅ Convert all integer columns to float64 (Fix MLflow Warning)
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        # ✅ Perform hyperparameter tuning with error handling
        try:
            gridsearch = GridSearchCV(model, param_grid=params, scoring='f1', cv=5, error_score='raise')
            gridsearch.fit(X_train, y_train)
        except Exception as e:
            print(f"⚠️ GridSearchCV failed for model: {model_name}")
            print(f"Error: {e}")
            return None, None  # Skip this model if it fails

        # ✅ Get the best model and hyperparameters
        best_model = gridsearch.best_estimator_
        best_params = gridsearch.best_params_

        # ✅ Get model predictions
        y_pred = best_model.predict(X_test)
        y_scores = best_model.predict_proba(X_test)[:, 1]

        # ✅ Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)

        # ✅ Log hyperparameters & metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)

        # ✅ Log additional metadata
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset_used", "Customer Churn Dataset")
        mlflow.set_tag("tracking_method", "SQLite Database")

        # ✅ Ensure "models/" directory exists
        artifact_dir = "models"
        os.makedirs(artifact_dir, exist_ok=True)

        # ✅ Create an input example (First row of X_train)
        input_example = X_train.iloc[:1]

        # ✅ Log model with signature
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=f"{artifact_dir}/{model_name}",
            input_example=input_example
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