import pandas as pd
import os
from config import dataset_path

def load_data():
    """
    Load the dataset from the specified directory.
    - Returns: DataFrame if successful, else None.
    """
    try:
        # ✅ Check if file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ Dataset file not found at: {dataset_path}")

        print(f"📂 Loading dataset from: {dataset_path}")
        churn_data = pd.read_csv(dataset_path)

        # ✅ Check if DataFrame is empty
        if churn_data.empty:
            raise ValueError("⚠️ Loaded dataset is empty!")

        print("✅ Data Loaded Successfully! Shape:", churn_data.shape)
        return churn_data

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except pd.errors.ParserError:
        print("❌ Error parsing the CSV file. Please check its format!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

    return None  # Return None if data loading fails

if __name__ == '__main__':
    churn_data = load_data()
    
    if churn_data is not None:
        print("🎯 Data preview:\n", churn_data.head())
    else:
        print("❌ Data loading failed. Please check the logs.")
