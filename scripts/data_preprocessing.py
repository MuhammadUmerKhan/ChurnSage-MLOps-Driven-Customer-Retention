import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import data_ingestion
import os

def load_data():
    """Load dataset from CSV file."""
    churn_data = data_ingestion.load_data()
    return churn_data

def clean_data(churn_data: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by handling missing values and removing duplicates."""
    churn_data = churn_data.copy()
    
    # Drop customer ID
    churn_data.drop(columns='customerID', inplace=True)
    
    # Convert 'TotalCharges' to numeric and handle missing values
    churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(' ', np.nan).astype(float)
    churn_data['TotalCharges'].fillna(value=churn_data['TotalCharges'].mean(), inplace=True)
    
    # Remove empty values
    churn_data = churn_data[churn_data['TotalCharges'].notna()]
    
    # Drop duplicates
    churn_data.drop_duplicates(inplace=True)

    return churn_data

def replace_no_service(value: str) -> str:
    """Replace 'No phone service' and 'No internet service' with 'No'."""
    return 'No' if value in ['No phone service', 'No internet service'] else value

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

    print('Label Encoder Transformation')
    for col in text_data_features:
        churn_data[col] = le.fit_transform(churn_data[col])
        print(col, ':', churn_data[col].unique(), '=', le.inverse_transform(churn_data[col].unique()))
    
    return churn_data

def split_features_and_target(churn_data: pd.DataFrame):
    """Separate features and target variable."""
    X = churn_data.drop(columns=['Churn', 'OnlineBackup', 'DeviceProtection', 
                                 'InternetService', 'gender', 'PhoneService', 
                                 'MultipleLines', 'StreamingMovies', 'StreamingTV'])
    y = churn_data['Churn']
    
    base_dir = os.path.join("..", "Datasets", "Processed Data")
    file_name = "Ready_data_for_model.csv"
    
    # Construct the full file path    
    file_path = os.path.join(base_dir, file_name)
    
    pd.concat([X, y], axis=1)[ 
                                ['MonthlyCharges', 'tenure', 'TotalCharges', 
                                'SeniorCitizen', 'Partner', 'Dependents', 
                                'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 
                                'Contract', 'PaymentMethod', 'Churn']
                            ].to_csv(file_path, index=False)
    
    return X, y

def apply_smoteenn(X: pd.DataFrame, y: pd.Series):
    """Handle class imbalance using SMOTEENN."""
    smote_enn = SMOTEENN()
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numeric features using MinMaxScaler."""
    columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Ensure numeric conversion and handle missing values
    X_train[columns_to_scale] = X_train[columns_to_scale].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test[columns_to_scale] = X_test[columns_to_scale].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale data
    scaler = MinMaxScaler()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train, X_test

if __name__ == "__main__":
    # Load data
    data = load_data()

    # Clean data
    data = clean_data(data)

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

    base_dir = os.path.join("..", "Datasets", "Model Data")

    # Save processed data
    X_train.to_csv(f"{base_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{base_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{base_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{base_dir}/y_test.csv", index=False)

    print("Preprocessing completed and files saved!")
