import pandas as pd
import os

def load_data():
    # Load the dataset
    
    base_dir = os.path.join("..", "Datasets", "Telecom chustomer churn dataset")
    file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Construct the full file path    
    file_path = os.path.join(base_dir, file_name)

    churn_data = pd.read_csv(file_path)
    return churn_data

if __name__ == '__main__':
    churn_data = load_data()
    print("Data Loaded Successfully")