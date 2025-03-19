import pandas as pd
import os

def load_data():
    """
    Load the dataset from the specified directory.
    - Returns: DataFrame if successful, else None.
    """
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "TelecomChustomerChurnDataset"))
        file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        file_path = os.path.join(base_dir, file_name)

        # ✅ Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Dataset file not found at: {file_path}")

        print(f"📂 Loading dataset from: {file_path}")
        churn_data = pd.read_csv(file_path)

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
