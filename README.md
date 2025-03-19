# 📌 Telecom Customer Churn Prediction with MLOps

![churn 1.png](https://miro.medium.com/v2/resize:fit:1024/1*TgciopaOk-C8fwtPmmet3w.png)

## 🚀 Project Overview
In the highly competitive telecom industry, **customer churn** (customers leaving for another provider) is a major challenge. This project leverages **Machine Learning and MLOps methodologies** to develop a predictive model that identifies customers likely to churn. By integrating **MLflow for experiment tracking, automated pipelines, and model registry**, this solution ensures efficient model training, evaluation, and deployment.

## 📑 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Project Features](#project-features)
- [MLOps Workflow](#mlops-workflow)
- [Project Directory Structure](#project-directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [MLflow Tracking and Model Registry](#mlflow-tracking-and-model-registry)
- [Model Testing](#model-testing)
- [Deployment (Coming Soon)](#deployment-coming-soon)
- [Conclusion](#conclusion)

---

## 📌 Problem Statement

Customer churn in the telecom industry leads to significant revenue loss. The challenge is to **predict potential churners** based on their behavior, contract type, and payment history. The solution involves:
- **Building an ML model** to classify customers as "Churn" or "Not Churn."
- **Tracking experiments** with MLflow to find the best-performing model.
- **Automating the MLOps workflow** for efficient retraining and deployment.

---

## ✅ Solution Approach
### **1️⃣ Data Processing & Feature Engineering**
- **Data Cleaning & Preprocessing** (handling missing values, encoding categorical variables, etc.)
- **Feature Selection & Scaling** (using MinMaxScaler & Label Encoding)
- **Handling Class Imbalance** (using **SMOTEENN**)

### **2️⃣ Model Training & Tracking**
- Train multiple models (**Logistic Regression, Decision Tree, Random Forest, KNN**)
- **Hyperparameter tuning** using GridSearchCV
- **Track experiments** using MLflow

### **3️⃣ Model Registry & Testing**
- **Register the best model** in MLflow Model Registry
- Load the best model and **test it on unseen data**

### **4️⃣ Deployment (Upcoming Feature)**
- Deploy the model as an **API using FastAPI or Flask**
- Develop a **Streamlit Web App** for user interaction

---

## 🔥 Project Features
- ✅ **End-to-End MLOps Pipeline** (From Data Ingestion to Model Testing)
- ✅ **MLflow Experiment Tracking & Model Registry**
- ✅ **Automated Model Training & Evaluation**
- ✅ **Best Model Selection using GridSearchCV**
- ✅ **Artifact Storage & Model Versioning**
- ✅ **Simple Automation Pipeline (Single Command Execution)**
- ✅ **Future Deployment with FastAPI & Streamlit**

---

## 🔁 MLOps Workflow
**1️⃣ Data Ingestion → 2️⃣ Data Preprocessing → 3️⃣ Model Training & Tracking → 4️⃣ Model Registration → 5️⃣ Model Testing**

```
+------------------+       +---------------------+      +------------------+
| Data Ingestion  | --->  | Data Preprocessing | ---> | Model Training  |
+------------------+       +---------------------+      +------------------+
                                                 |
+---------------------+      +------------------+
| MLflow Experiment  | ---> | Model Registry  |
+---------------------+      +------------------+
```

---

## 📂 Project Directory Structure
```
Customer-Churn-Prediction-with-NLP-Insights/
│── Datasets/
│── models/
│── scripts/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── register_model.py
│   ├── test_model.py
│── pipeline.py  <-- Runs everything automatically
│── mlflow.db
│── mlruns/
│── requirements.txt
│── README.md
```

---

## ⚙️ Setup and Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights.git
cd Customer-Churn-Prediction-with-NLP-Insights
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Initialize MLflow Tracking**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

---

## 🚀 Running the Pipeline (Automated Execution)
Run the entire **MLOps pipeline** in a **single command**:
```bash
python3 pipeline.py
```

This will execute:
1. **Data Ingestion** (`data_ingestion.py`)
2. **Preprocessing & Feature Engineering** (`data_preprocessing.py`)
3. **Model Training & Experiment Tracking** (`train_model.py`)
4. **Model Registration** (`register_model.py`)
5. **Model Testing** (`test_model.py`)

---

## 📊 MLflow Tracking and Model Registry
### **1️⃣ View Experiment Runs**
```bash
mlflow ui
```
Navigate to **http://localhost:5000** to check experiment logs, metrics, and artifacts.

### **2️⃣ Register the Best Model**
Run:
```bash
python3 scripts/register_model.py
```
This will:
✅ Select the best-performing model
✅ Register it in MLflow Model Registry
✅ Assign aliases like **'production'** and **'staging'**

### **3️⃣ Load the Best Model**
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/customer_churn_model@production")
```

---

## 🧪 Model Testing
Once the model is registered, **test it with new data**:
```bash
python3 scripts/test_model.py
```
✅ This loads the **production model** and makes predictions on **new customer data**.

---

## 🚀 Deployment (Coming Soon)
Future plans include:
- **FastAPI/Flask API** to serve the model
- **Streamlit UI** for user-friendly predictions

---

## 📌 Conclusion
This project demonstrates a complete **MLOps workflow for customer churn prediction**, integrating **MLflow tracking, automated pipelines, and model registry**. Future work will focus on **deployment** to make the model accessible via an API or web application.

🔹 **Want to contribute?** Fork the repo and submit a PR! 🚀
