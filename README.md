# 📌 **📉 ChurnSage: MLOps-Driven Customer Retention**

![churn 1.png](https://miro.medium.com/v2/resize:fit:1024/1*TgciopaOk-C8fwtPmmet3w.png)

## 🚀 **Project Overview**
In the **highly competitive telecom industry**, customer churn (when customers leave for a competitor) is a major business challenge.  
This project **leverages Machine Learning and MLOps** to build a predictive model that identifies **customers likely to churn** based on their **usage behavior, contract type, and payment history**.  

💡 **What makes this project unique?**  
- 👉 **Automated Machine Learning Pipeline** → From data ingestion to model training & deployment
- 👉 **MLOps Integration** → MLflow for experiment tracking, model registry, and artifact storage
- 👉 **Automated Model Selection & Registration** → Tracks the best-performing model dynamically
- 👉 **Integration of LLM with ChatQrok** → Uses AI to predict churn based on customer feedback
- 👉 **Integrates FastAPI** → Real-time predictions
- 👉 **Database Integration** → Stores customer inputs & predictions in SQLite
- 👉 **Bulk CSV Upload & Prediction** → Allows batch processing of customer data
- 👉 **Future-Proof Design** → Prepares the model for scalable deployment via API or Web UI

---

## **📁 Table of Contents**
- [📌 Problem Statement](#-problem-statement)
- [🛠️ Solution Approach](#-solution-approach)
- [🔥 Project Features](#-project-features)
- [📊 Insights](#-Insights)
- [🔁 MLOps Workflow](#-mlops-workflow)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🚀 Running the Automated Pipeline](#-running-the-automated-pipeline)
- [🖥️ Running the FastAPI Server](#-running-the-fastapi-server)
- [📊 MLflow Tracking and Model Registry](#-mlflow-tracking-and-model-registry)
- [🧩 Model & LLM Testing (Postman)](#-model--llm-testing-postman)
- [📦 Database Integration](#-database-integration)
- [📌 Conclusion](#-conclusion)

---

## 🚀 **Project Overview**

In the **highly competitive telecom industry**, customer churn (when customers leave for a competitor) is a major business challenge.\
This project **leverages Machine Learning and MLOps** to build a predictive model that identifies **customers likely to churn** based on their **usage behavior, contract type, and payment history**.

💡 **What makes this project unique?**\
👉 **Automated Machine Learning Pipeline** → From **data ingestion** to **model training & deployment**\
👉 **MLOps Integration** → MLflow for **experiment tracking, model registry, and artifact storage**\
👉 **Automated Model Selection & Registration** → Tracks the **best-performing model dynamically**\
👉 **Integration of LLM with ChatQrok** → Uses AI to predict churn based on customer feedback\
👉 **Database Storage** (SQLite) for customer data & LLM responses\
👉 **Future-Proof Design** → Prepares the model for **scalable deployment via API or Web UI**

---

## 📌 **Problem Statement**

Customer retention is a **critical concern** in the telecom industry due to **high competition and acquisition costs**. The key challenge is:\
**"Can we predict which customers are likely to churn and take proactive actions to retain them?"**

To solve this, we must:\
👉 **Analyze customer behavior** using structured data\
👉 **Identify key features** influencing customer churn\
👉 **Train Machine Learning models** to predict churners accurately\
👉 **Use MLOps to automate & track the model lifecycle**

---

## 🛠️ **Solution Approach**

Our solution **leverages Machine Learning, MLOps & LLM** to build a **reliable, scalable, and automated churn prediction system**.

### **1️⃣ Data Processing & Feature Engineering**

- 👉 **Data Cleaning & Handling Missing Values**
- 👉 **Categorical Feature Encoding** (Label Encoding)
- 👉 **Feature Scaling** using **MinMaxScaler**
- 👉 **Handling Class Imbalance** using **SMOTEENN**

### **2️⃣ Model Training & Experiment Tracking**

- 👉 Train **Logistic Regression, Decision Tree, Random Forest, KNN**
- 👉 Use **GridSearchCV for Hyperparameter Tuning**
- 👉 **Track Experiments** using **MLflow**

### **3️⃣ Model Selection & Registry**

- 👉 **Automatically register the best model** in **MLflow Model Registry**
- 👉 **Assign model aliases** (`staging`, `production`) for streamlined versioning
- 👉 **Store models, hyperparameters, and metrics** in **MLflow artifacts**

### **4️⃣ Customer Feedback Analysis with LLM**

- 👉 Use **ChatQrok LLM** to analyze **customer reviews & predict churn**
- 👉 Implement an **NLP-powered chatbot** for sentiment-based churn prediction

### **5️⃣ Deployment**

- 🚀 **Expose model as an API** using **FastAPI/Flask**
- 🎨 **Create an interactive UI** using **Streamlit**

---

## 🔥 **Project Features**
🔹 **End-to-End Automated MLOps Pipeline**  
🔹 **MLflow for Experiment Tracking & Model Registry**  
🔹 **Automated Model Training, Selection & Evaluation**  
🔹 **Artifact Storage & Model Versioning**  
🔹 **SMOTEENN for Handling Class Imbalance**  
🔹 **Future Deployment with FastAPI & Streamlit UI**  

---
## 📊 **Insights**


| Visualization Type      | Distribution/Insights |
|------------------------|----------------------|
| **Categorical Feature** | ![Feature Importance](https://raw.githubusercontent.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/main/insights/categorical_feature.png) |
| **Churn Analysis**   | ![Confusion Matrix](https://raw.githubusercontent.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/main/insights/churn_analysis.png) |
| **Churn Distribution** | ![Churn Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/main/insights/churn_vs_not_churn.png) |
| **Tech Support**    | ![Customer Tenure Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/main/insights/onlien%20analysis.png) |

---
## 🔁 **MLOps Workflow**
Our project follows an **industry-standard MLOps pipeline**:

```
+------------------+       +---------------------+      +------------------+
| Data Ingestion  | --->  | Data Preprocessing | ---> | Model Training  |
+------------------+       +---------------------+      +------------------+
                                                 |
+---------------------+      +------------------+
| MLflow Experiment  | ---> | Model Registry  |
+---------------------+      +------------------+
```

Each step is **fully automated** and can be executed using **a single command**.

---

## 📂 **Project Directory Structure**
```
Customer-Churn-Prediction-with-NLP-Insights/
│── Datasets/
│── models/
│── scripts/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── register_best_model.py
│   ├── test_model.py
│── pipeline.py  <-- Runs everything automatically
│── mlflow.db
│── mlruns/
│── requirements.txt
│── README.md
```
---

## ⚙️ **Setup and Installation**
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
mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root ./mlruns
```

---

## 🚀 **Running the Automated Pipeline**
Run the **entire project pipeline in one command**:
```bash
python3 pipeline.py
```
This will sequentially execute:
✅ **Data Ingestion** (`data_ingestion.py`)  
✅ **Preprocessing & Feature Engineering** (`data_preprocessing.py`)  
✅ **Model Training & Experiment Tracking** (`train_model.py`)  
✅ **Model Registration** (`register_best_model.py`)  
✅ **Model Testing & Prediction** (`test_model.py`)  

---

## **🖥️ Running the FastAPI Server**
Once the model is trained and registered, run **FastAPI** to serve real-time predictions:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
This starts the FastAPI server on **http://127.0.0.1:8000**.

---


## **🧩 Model & LLM Testing (Postman)**
Use **Postman** to send requests to the FastAPI server.

### **1️⃣ Test Churn Prediction (POST Request)**
- **Endpoint:** `http://127.0.0.1:8000/predict`
- **Request Body:**
```json
{
    "SeniorCitizen": 1,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 24,
    "OnlineSecurity": 1,
    "TechSupport": 0,
    "Contract": 1,
    "PaperlessBilling": 1,
    "PaymentMethod": 3,
    "MonthlyCharges": 79.99,
    "TotalCharges": 1899.99
}
```
- **Response:**
```json
{
    "prediction": "Customer will stay"
}
```

### **2️⃣ Test LLM Feedback Analysis**
- **Endpoint:** `http://127.0.0.1:8000/predict-llm`
- **Request Body:**
```json
{
    "user_feedback": "The service is really slow, and I am considering switching providers!"
}
```
- **Response:**
```json
{
    "llm_prediction": "Customer likely to leave",
    "llm_reasoning": "The customer is unhappy with service speed and considering alternatives, indicating a high churn risk."
}
```

---

## 📊 **MLflow Tracking and Model Registry**
### **1️⃣ View Experiment Runs**
Launch MLflow UI to explore experiment tracking:
```bash
mlflow ui
```
📌 Navigate to **http://localhost:5000** to check experiment logs, metrics, and artifacts.

### **2️⃣ Register the Best Model**
```bash
python3 scripts/register_best_model.py
```
This will:\
✅ **Select the best-performing model**  
✅ **Register it in MLflow Model Registry**  
✅ **Assign `staging` & `production` aliases**  

### **3️⃣ Load the Best Model**
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/customer_churn_model@production")
```

---

## **📦 Database Integration**
This project **stores data in an SQLite database**.

### **1️⃣ Customer Churn Data**
**Table:** `customer_data`
Stores **customer input features & churn predictions**.

| SeniorCitizen | Partner | Dependents | tenure | OnlineSecurity | TechSupport | Contract | PaperlessBilling | PaymentMethod | MonthlyCharges | TotalCharges | Prediction |
|--------------|---------|------------|--------|---------------|------------|---------|-----------------|--------------|--------------|------------|------------|
| No          | Yes     | No         | 24     | Yes           | No         | One year | Yes            | Credit Card  | 79.99        | 1899.99    | Customer will stay |

### **2️⃣ LLM Feedback Analysis**
**Table:** `llm_feedback`
Stores **customer reviews & LLM predictions**.

| user_feedback | llm_prediction | llm_reasoning |
|--------------|---------------|--------------|
| "Service is slow, considering switching providers!" | Customer likely to leave | The customer is unhappy with service speed and considering alternatives. |

---

## 🧪 **Model Testing**
Once the model is registered, **test it with new customer data**:
```bash
python3 scripts/test_model.py
```
✅ This loads the **latest production model** and makes **predictions on unseen data**.

---
#### 🧪 **[Streamlit Deplyed Demo](https://customer-churn-prediction-with-nlp-insights-rlazeungatswsd5e4c.streamlit.app/)**
---

To add Dockerization and Docker Hub usage instructions without touching any of the core content, you can append the following section **at the end of your README**:

---

## 🐳 **Dockerization & Deployment**
- **Build**:
  ```bash
  docker build -t muhammadumerkhan/customer-churn-predictor .
  ```
- **Run**:
  ```bash
  docker run -p 8501:8501 muhammadumerkhan/customer-churn-predictor
  ```
- **Push**:
  ```bash
  docker push muhammadumerkhan/customer-churn-predictor
  ```
- **Pull**:
  ```bash
  docker pull muhammadumerkhan/customer-churn-predictor
  ```

Anyone can pull and run the app using:

```bash
docker pull muhammadumerkhan/customer-churn-predictor
docker run -p 8501:8501 muhammadumerkhan/customer-churn-predictor
```
---
## **📌 Conclusion**
This project demonstrates a **complete MLOps workflow** integrating:<br>
✅ **MLflow for experiment tracking**  
✅ **FastAPI for real-time model deployment**  
✅ **ChatQrok LLM for feedback-based churn prediction**  
✅ **SQLite database for customer data storage**  

💡 **Next Steps:**  
- 🖥️ **Deploy FastAPI** on cloud services  
- 📊 **Improve LLM predictions** using more advanced NLP models  
- 🔧 **Enhance Streamlit UI** for better customer insights  

🔹 **Want to contribute?** Fork the repo and submit a PR! 🚀 
