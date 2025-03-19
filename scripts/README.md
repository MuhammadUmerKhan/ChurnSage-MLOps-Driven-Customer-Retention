# 📜 **Scripts Overview - Customer Churn Prediction with MLOps**  

## 📌 **Introduction**
This document provides an overview of all scripts used in this project, explaining their **roles, functionalities, and dependencies**. It also highlights how data and models are stored and tracked using **MLflow and SQLite**.

---

## 📂 **Scripts and Their Purpose**
Each script plays a crucial role in automating the **MLOps workflow**, ensuring smooth execution from **data ingestion** to **model registration and testing**.

### 1️⃣ `data_ingestion.py` 📥  
**Purpose:** Loads the dataset and prepares it for further preprocessing.  
🔹 Reads the **Telco Customer Churn dataset** from `Datasets/TelecomCustomerChurnDataset/`  
🔹 Handles **file loading errors** with exception handling  
🔹 Returns a **pandas DataFrame** containing raw data  

**Database Interaction:**  
- The dataset is not stored in a database at this stage. It is simply loaded into memory for preprocessing.  

---

### 2️⃣ `data_preprocessing.py` 🛠️  
**Purpose:** Cleans and preprocesses raw data for machine learning models.  
🔹 **Handles missing values** and **duplicates**  
🔹 **Encodes categorical features** using **Label Encoding**  
🔹 **Applies SMOTEENN** to handle **class imbalance**  
🔹 **Scales numerical features** using **MinMaxScaler**  
🔹 **Saves the processed data** in `Datasets/ProcessedData/`  

**Database Interaction:**  
- Stores **processed data** (`X_train`, `X_test`, `y_train`, `y_test`) as `.csv` files in `Datasets/ModelData/`.  
- Saves the **scaler (`MinMaxScaler`) as a `.pkl` file** in `models_joblib_files/` for deployment.  

---

### 3️⃣ `train_model.py` 🎯  
**Purpose:** Trains multiple machine learning models and tracks them in MLflow.  
🔹 Uses **Logistic Regression, Decision Tree, Random Forest, KNN**  
🔹 **Performs Hyperparameter tuning** with `GridSearchCV`  
🔹 Calculates **performance metrics** (Accuracy, F1-Score, Precision, Recall, ROC-AUC)  
🔹 Logs models, parameters, and metrics in **MLflow Experiment Tracking**  

**Database Interaction:**  
- Uses **MLflow (`mlflow.db`)** for experiment tracking.  
- Stores trained models in **MLflow's artifact storage (`mlruns/`)**.  
- Saves **hyperparameters and dataset info** as JSON in `models/`.  

---

### 4️⃣ `register_best_model.py` 🏆  
**Purpose:** Identifies the **best model** and registers it in **MLflow Model Registry**.  
🔹 Fetches **all trained models** from MLflow and sorts by **highest F1-score**  
🔹 Registers the **best-performing model** in **MLflow Model Registry**  
🔹 Assigns aliases:  
   - `staging` → For testing  
   - `production` → For deployment  
🔹 **Adds metadata tags** (dataset, training method, author)  

**Database Interaction:**  
- Uses **MLflow Model Registry** (`mlflow.db`) to store model versions.  
- Saves metadata and aliases for version control.  

---

### 5️⃣ `test_model.py` 🔬  
**Purpose:** Loads the **latest production model** and tests it on unseen data.  
🔹 Fetches the **best registered model** from MLflow  
🔹 Loads `X_test.csv` from `Datasets/ModelData/`  
🔹 Makes **real-time predictions**  
🔹 Prints **sample predictions** for validation  

**Database Interaction:**  
- Loads the **best model** from **MLflow Model Registry** (`mlflow.db`).  
- Reads **test dataset** stored in `Datasets/ModelData/`.  

---

### 6️⃣ `hyperparams.py` 🔢  
**Purpose:** Stores predefined **hyperparameter configurations** for each model.  
🔹 Used by `train_model.py` during **GridSearchCV tuning**  
🔹 Ensures **consistent tuning parameters** across runs  

**Database Interaction:**  
- No direct database interaction.  

**Database Interaction:**  
- Uses **MLflow (`mlflow.db`)** for experiment tracking and model registry.  
- Reads and writes data from `Datasets/`.  

---

## 🗄️ **Database & Storage Overview**  
This project leverages **MLflow and SQLite** for efficient storage and tracking.

| **Component**      | **Storage Location**                      | **Purpose** |
|-------------------|-----------------------------------------|-------------|
| **Raw Dataset**   | `Datasets/TelecomCustomerChurnDataset/` | Original data |
| **Processed Data** | `Datasets/ProcessedData/` & `Datasets/ModelData/` | Cleaned & scaled dataset |
| **Trained Models** | `mlruns/` (MLflow artifact storage) | Stores model artifacts |
| **Model Registry** | `mlflow.db` (SQLite database) | Tracks model versions |
| **Scaler & Artifacts** | `models_joblib_files/` & `models/` | Stores scalers & JSON metadata |


---

## 🔥 **Conclusion**  
This **Scripts.md** file serves as a **comprehensive guide** to understanding how different scripts work together to build a **fully automated, MLOps-driven churn prediction system**. 🚀  

If you have any **questions or suggestions**, feel free to contribute! 🤝

---