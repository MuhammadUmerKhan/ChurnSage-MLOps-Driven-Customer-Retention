import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


mlflow.set_experiment("Customer Churn Prediction")

def train_and_track_model(model, X_train, y_train, X_test, y_test, params, model_name):
    with mlflow.start_run():
        gridsearch = GridSearchCV(model, param_grid=params, scoring='f1', cv=5)
        gridsearch.fit(X_train, y_train)

        best_model = gridsearch.best_estimator_
        best_params = gridsearch.best_params_

        # Log hyperparameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_model.score(X_test, y_test))

        # Save model artifacts
        mlflow.sklearn.log_model(best_model, artifact_path=f"models/{model_name}")

        print(f"Model '{model_name}' logged to MLflow.")

        return best_model, best_params

if __name__ == "__main__":
    # Load processed data
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Train different models and log them
    train_and_track_model(LogisticRegression(), X_train, y_train, X_test, y_test, logistic_params, "logistic_model")
    train_and_track_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, decision_tree_params, "decision_tree_model")
    train_and_track_model(RandomForestClassifier(), X_train, y_train, X_test, y_test, random_forest_params, "random_forest_model")
    train_and_track_model(KNeighborsClassifier(), X_train, y_train, X_test, y_test, knn_params, "knn_model")
