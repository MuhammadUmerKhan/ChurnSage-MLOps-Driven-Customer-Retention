### 7️⃣ `pipeline.py` 🔄  
**Purpose:** Automates the **entire MLOps workflow** in a single command.  
🔹 Sequentially executes:  
   1️⃣ **Data Ingestion** (`data_ingestion.py`)  
   2️⃣ **Data Preprocessing** (`data_preprocessing.py`)  
   3️⃣ **Model Training** (`train_model.py`)  
   4️⃣ **Model Registration** (`register_best_model.py`)  
   5️⃣ **Model Testing** (`test_model.py`)  
🔹 Ensures **error handling** for failed steps  


## 📌 **How to Run Everything?**  
Simply execute the **entire MLOps pipeline** using:  
```bash
python3 pipeline.py
```
This will run **all scripts sequentially**, ensuring a smooth **end-to-end** workflow.