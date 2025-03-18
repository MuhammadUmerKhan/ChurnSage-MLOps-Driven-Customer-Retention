import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import data_ingestion  # ✅ Ensure this module correctly loads data

def load_data():
    """Load dataset from CSV file."""
    return data_ingestion.load_data()

def clean_data(churn_data: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by handling missing values and removing duplicates."""
    churn_data = churn_data.copy()
    
    # Drop customer ID
    churn_data.drop(columns='customerID', inplace=True, errors='ignore')
    
    # Convert 'TotalCharges' to numeric and handle missing values
    churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(' ', np.nan).astype(float)
    churn_data['TotalCharges'].fillna(value=churn_data['TotalCharges'].mean(), inplace=True)

    # Drop duplicates
    churn_data.drop_duplicates(inplace=True)

    return churn_data

def replace_no_service(value):
    """Replace 'No phone service' and 'No internet service' with 'No'."""
    return 'No' if str(value) in ['No phone service', 'No internet service'] else value

def transform_categorical_features(churn_data: pd.DataFrame) -> pd.DataFrame:
    """Transform categorical features using Label Encoding."""
    churn_data = churn_data.copy()
    
    # Apply transformation to selected columns
    columns_to_transform = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for column in columns_to_transform:
        churn_data[column] = churn_data[column].apply(replace_no_service)
    
    # Label encoding for categorical text features
    le = LabelEncoder()
    text_data_features = [col for col in churn_data.columns if churn_data[col].dtype == 'object']

    print("✅ Applying Label Encoding...")
    for col in text_data_features:
        churn_data[col] = le.fit_transform(churn_data[col])
        print(f"🔹 {col}: {churn_data[col].unique()}")

    return churn_data

def split_features_and_target(churn_data: pd.DataFrame):
    """Separate features and target variable."""
    X = churn_data.drop(columns=['Churn', 'OnlineBackup', 'DeviceProtection', 
                                 'InternetService', 'gender', 'PhoneService', 
                                 'MultipleLines', 'StreamingMovies', 'StreamingTV'])
    y = churn_data['Churn']
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ProcessedData"))
    file_name = "Ready_data_for_model.csv"
    
    os.makedirs(base_dir, exist_ok=True)  # ✅ Ensure directory exists
    
    file_path = os.path.join(base_dir, file_name)
    pd.concat([X, y], axis=1).to_csv(file_path, index=False)
    print(f"✅ Processed Data saved to: {file_path}")

    return X, y

def apply_smoteenn(X: pd.DataFrame, y: pd.Series):
    """Handle class imbalance using SMOTEENN."""
    print("✅ Applying SMOTEENN...")
    smote_enn = SMOTEENN()
    return smote_enn.fit_resample(X, y)

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numeric features using MinMaxScaler."""
    columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for df in [X_train, X_test]:
        df[columns_to_scale] = df[columns_to_scale].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale data
    scaler = MinMaxScaler()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train, X_test

def run_preprocessing_pipeline():
    """Runs the full preprocessing pipeline."""
    print("🚀 Starting Data Preprocessing...")

    # Load and clean data
    data = clean_data(load_data())

    # Transform categorical features
    data = transform_categorical_features(data)

    # Separate features and target
    X, y = split_features_and_target(data)

    # Handle class imbalance
    X_resampled, y_resampled = apply_smoteenn(X, y)

    # Split into train & test sets
    X_train, X_test, y_train, y_test = split_train_test(X_resampled, y_resampled)

    # Scale numeric features
    X_train, X_test = scale_numeric_features(X_train, X_test)

    model_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ModelData"))
    os.makedirs(model_data_path, exist_ok=True)  # ✅ Ensure directory exists
    
    # Save processed data
    X_train.to_csv(f"{model_data_path}/X_train.csv", index=False)
    X_test.to_csv(f"{model_data_path}/X_test.csv", index=False)
    y_train.to_csv(f"{model_data_path}/y_train.csv", index=False)
    y_test.to_csv(f"{model_data_path}/y_test.csv", index=False)

    print("✅ Preprocessing completed! Processed data saved to 'Datasets/ModelData/'.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_preprocessing_pipeline()