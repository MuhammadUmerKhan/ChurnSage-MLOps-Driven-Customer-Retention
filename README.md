# 📌 **Telecom Customer Churn Prediction with MLOps**

![churn 1.png](https://miro.medium.com/v2/resize:fit:1024/1*TgciopaOk-C8fwtPmmet3w.png)

## 🚀 **Project Overview**
In the **highly competitive telecom industry**, customer churn (when customers leave for a competitor) is a major business challenge.  
This project **leverages Machine Learning and MLOps** to build a predictive model that identifies **customers likely to churn** based on their **usage behavior, contract type, and payment history**.  

💡 **What makes this project unique?**  
- 👉 Automated Machine Learning Pipeline → From data ingestion to model training & deployment
- 👉 MLOps Integration → MLflow for experiment tracking, model registry, and artifact storage
- 👉 Automated Model Selection & Registration → Tracks the best-performing model dynamically
- 👉 Integration of LLM with ChatQrok → Uses AI to predict churn based on customer feedback
- 👉 Future-Proof Design → Prepares the model for scalable deployment via API or Web UI

---

## 📑 **Table of Contents**
- [📌 Problem Statement](#-problem-statement)
- [🛠 Solution Approach](#-solution-approach)
- [🔥 Project Features](#-project-features)
- [🔁 MLOps Workflow](#-mlops-workflow)
- [📂 Project Directory Structure](#-project-directory-structure)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🚀 Running the Automated Pipeline](#-running-the-automated-pipeline)
- [📊 MLflow Tracking and Model Registry](#-mlflow-tracking-and-model-registry)
- [🧪 Model Testing](#-model-testing)
- [🌍 Deployment (Upcoming)](#-deployment-upcoming)
- [📌 Conclusion](#-conclusion)

---
# 📌 **Telecom Customer Churn Prediction with MLOps**



## 🚀 **Project Overview**

In the **highly competitive telecom industry**, customer churn (when customers leave for a competitor) is a major business challenge.\
This project **leverages Machine Learning and MLOps** to build a predictive model that identifies **customers likely to churn** based on their **usage behavior, contract type, and payment history**.

💡 **What makes this project unique?**\
👉 **Automated Machine Learning Pipeline** → From **data ingestion** to **model training & deployment**\
👉 **MLOps Integration** → MLflow for **experiment tracking, model registry, and artifact storage**\
👉 **Automated Model Selection & Registration** → Tracks the **best-performing model dynamically**\
👉 **Integration of LLM with ChatQrok** → Uses AI to predict churn based on customer feedback\
👉 **Future-Proof Design** → Prepares the model for **scalable deployment via API or Web UI**

---

## 📁 **Table of Contents**

- [📌 Problem Statement](#-problem-statement)
- [🛠️ Solution Approach](#-solution-approach)
- [🔥 Project Features](#-project-features)
- [🔀 MLOps Workflow](#-mlops-workflow)
- [📂 Project Directory Structure](#-project-directory-structure)
- [⚙️ Setup and Installation](#-setup-and-installation)
- [🚀 Running the Automated Pipeline](#-running-the-automated-pipeline)
- [📊 MLflow Tracking and Model Registry](#-mlflow-tracking-and-model-registry)
- [🧩 Model Testing](#-model-testing)
- [🌍 Deployment (Upcoming)](#-deployment-upcoming)
- [📌 Conclusion](#-conclusion)

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

### **5️⃣ Deployment (Upcoming)**

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
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
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
This will:
✅ **Select the best-performing model**  
✅ **Register it in MLflow Model Registry**  
✅ **Assign `staging` & `production` aliases**  

### **3️⃣ Load the Best Model**
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/customer_churn_model@production")
```

---

## 🧪 **Model Testing**
Once the model is registered, **test it with new customer data**:
```bash
python3 scripts/test_model.py
```
✅ This loads the **latest production model** and makes **predictions on unseen data**.

---

## 🌍 **Deployment (Upcoming)**
Future improvements:
- **FastAPI/Flask API** → Serve predictions via REST API  
- **Streamlit Web App** → Interactive UI for churn predictions  
- **CI/CD Integration** → Automate training and deployment  

---

## 📌 **Conclusion**
This project demonstrates a **complete MLOps workflow** for customer churn prediction, integrating **MLflow tracking, automated pipelines, and model registry**.  
Future work will focus on **deployment** to make the model accessible via an API or web application.

💡 **Want to contribute?** Fork the repo and submit a PR! 🚀  

---