import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import data_ingestion  # ✅ Ensure this module correctly loads data

def load_data():
    """Load dataset from CSV file."""
    try:
        print("📂 Loading data...")
        churn_data = data_ingestion.load_data()

        if churn_data is None or churn_data.empty:
            raise ValueError("⚠️ Dataset is empty or failed to load!")

        print(f"✅ Data Loaded Successfully! Shape: {churn_data.shape}")
        return churn_data

    except Exception as e:
        print(f"❌ Error in loading data: {e}")
        return None

def clean_data(churn_data: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by handling missing values and removing duplicates."""
    try:
        print("🛠️ Cleaning data...")
        churn_data = churn_data.copy()
        
        # Drop customer ID
        churn_data.drop(columns='customerID', inplace=True, errors='ignore')
        
        # Convert 'TotalCharges' to numeric and handle missing values
        churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(' ', np.nan).astype(float)
        churn_data['TotalCharges'].fillna(value=churn_data['TotalCharges'].mean(), inplace=True)

        # Drop duplicates
        churn_data.drop_duplicates(inplace=True)

        print("✅ Data Cleaning Completed!")
        return churn_data

    except Exception as e:
        print(f"❌ Error in cleaning data: {e}")
        return churn_data  # Return raw data in case of failure

def replace_no_service(value):
    """Replace 'No phone service' and 'No internet service' with 'No'."""
    return 'No' if str(value) in ['No phone service', 'No internet service'] else value

def transform_categorical_features(churn_data: pd.DataFrame) -> pd.DataFrame:
    """Transform categorical features using Label Encoding."""
    try:
        print("🔄 Transforming categorical features using Label Encoding...")
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

        for col in text_data_features:
            churn_data[col] = le.fit_transform(churn_data[col])

        print("✅ Categorical Transformation Completed!")
        return churn_data

    except Exception as e:
        print(f"❌ Error in categorical transformation: {e}")
        return churn_data  # Return data without transformation

def split_features_and_target(churn_data: pd.DataFrame):
    """Separate features and target variable."""
    try:
        print("📊 Splitting features and target variable...")

        X = churn_data.drop(columns=['Churn', 'OnlineBackup', 'DeviceProtection', 
                                     'InternetService', 'gender', 'PhoneService', 
                                     'MultipleLines', 'StreamingMovies', 'StreamingTV'])
        y = churn_data['Churn']

        # Save processed data
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "ProcessedData"))
        os.makedirs(base_dir, exist_ok=True)  # ✅ Ensure directory exists
        
        file_path = os.path.join(base_dir, "Ready_data_for_model.csv")
        pd.concat([X, y], axis=1).to_csv(file_path, index=False)
        print(f"✅ Processed Data saved to: {file_path}")

        return X, y

    except Exception as e:
        print(f"❌ Error in splitting features and target: {e}")
        return None, None

def apply_smoteenn(X: pd.DataFrame, y: pd.Series):
    """Handle class imbalance using SMOTEENN."""
    try:
        print("📈 Applying SMOTEENN for balancing data...")
        smote_enn = SMOTEENN()
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        print("✅ SMOTEENN Applied Successfully!")
        return X_resampled, y_resampled

    except Exception as e:
        print(f"❌ Error in applying SMOTEENN: {e}")
        return X, y  # Return original data

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split dataset into training and testing sets."""
    try:
        print("📤 Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=test_size, random_state=42)

    except Exception as e:
        print(f"❌ Error in train-test split: {e}")
        return None, None, None, None

def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numeric features using MinMaxScaler."""
    try:
        print("📏 Scaling numeric features using MinMaxScaler...")
        columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

        for df in [X_train, X_test]:
            df[columns_to_scale] = df[columns_to_scale].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Scale data
        scaler = MinMaxScaler()
        X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        # ✅ Save the scaler for deployment
        scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl"))
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved at: {scaler_path}")

        return X_train, X_test

    except Exception as e:
        print(f"❌ Error in scaling features: {e}")
        return X_train, X_test

def run_preprocessing_pipeline():
    """Runs the full preprocessing pipeline."""
    print("\n🚀 Starting Data Preprocessing Pipeline...\n")

    # Load and clean data
    data = load_data()
    if data is None:
        print("❌ Data loading failed. Exiting pipeline.")
        return None, None, None, None

    data = clean_data(data)

    # Transform categorical features
    data = transform_categorical_features(data)

    # Separate features and target
    X, y = split_features_and_target(data)
    if X is None or y is None:
        print("❌ Feature splitting failed. Exiting pipeline.")
        return None, None, None, None

    # Handle class imbalance
    X_resampled, y_resampled = apply_smoteenn(X, y)

    # Split into train & test sets
    X_train, X_test, y_train, y_test = split_train_test(X_resampled, y_resampled)
    if X_train is None or X_test is None:
        print("❌ Train-test split failed. Exiting pipeline.")
        return None, None, None, None

    # Scale numeric features
    X_train, X_test = scale_numeric_features(X_train, X_test)

    print("\n✅ Preprocessing pipeline completed successfully!\n")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_preprocessing_pipeline()
