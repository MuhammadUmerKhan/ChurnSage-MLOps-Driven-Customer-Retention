# Telecom Customer Churn Attrition Prediction

![churn 1.png](https://miro.medium.com/v2/resize:fit:1024/1*TgciopaOk-C8fwtPmmet3w.png)

In today's competitive telecom industry, retaining customers is crucial. Customer churn, where users leave for another provider, can significantly impact a company's bottom line. This project focuses on building a predictive model to identify customers likely to leave based on usage patterns, behavior, and other factors. By understanding these patterns, companies can take proactive measures to enhance customer satisfaction and reduce churn.

This repository contains all the necessary components for analyzing customer churn, from the dataset to the code for predictions. It also offers insights into key factors driving customer attrition, suggests strategies for improving retention, and explores the role of natural language processing (NLP) in understanding customer sentiments.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Data Insights](#data-insights)
- [Key Findings](#key-findings)
- [References](#references)
- [Usage Instructions](#usage-instructions)
- [Running the Project](#running-the-project)
- [License](#license)

---

## Problem Statement

In the dynamic landscape of the telecommunications industry, customers can choose from a wide range of service providers. Customer satisfaction is vital, as users often base their perceptions of a company on singular interactions. The significance of understanding and mitigating customer churn is highlighted by the direct impact of churn rate on revenue. 

Given the high costs associated with acquiring new customers, in-depth churn analysis is essential. Insights derived from this analysis empower companies to formulate strategic approaches, target specific segments, and enhance service quality. By building predictive models and integrating NLP insights to analyze customer feedback, we can drive sustainable growth and strengthen customer loyalty.

---

## Methodology

1. **Data Collection and Preparation:**
   - Curated a comprehensive dataset comprising customer demographics, service usage details, contract specifics, and billing preferences.
   - Employed rigorous data cleaning and preprocessing techniques to ensure data integrity and consistency.

2. **Exploratory Data Analysis (EDA):**
   - Conducted in-depth EDA to uncover hidden patterns, correlations, and anomalies within the dataset.
   - Visualized key metrics such as customer segmentation by contract type and churn propensity across different customer segments using tools like Power BI for interactive visualizations.

3. **Natural Language Processing (NLP) Insights:**
   - Analyzed customer feedback and interactions to extract sentiments and identify common pain points using NLP techniques.
   - Employed text preprocessing methods, such as tokenization, stemming, and lemmatization, to prepare text data for analysis.
   - Utilized sentiment analysis to correlate customer sentiments with churn rates, gaining deeper insights into factors influencing attrition.

4. **Predictive Modeling:**
   - Developed and fine-tuned machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting.
   - Evaluated model performance based on metrics such as accuracy, precision, recall, and ROC-AUC score to select the optimal model for churn prediction.
---

## Data Insights

Explore profound insights and analytics gleaned from our extensive dataset. Uncover a deeper understanding of customer behaviors, patterns in service usage, and the pivotal factors influencing churn dynamics.

| Feature                                      | Visualization                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Categorical Features                         | ![Categorical Features](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/categorical_feature.png)   |
| Churn Target Variable                        | ![Churn Target Variable](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/churn_vs_not_churn.png)  |
| Customer Information                         | ![Customer Information](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/churning.png)   |
| Distribution Analysis                        | ![Distribution Analysis](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/distro_analysis.png)   |
| Security Distribution                        | ![Security Distribution](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/onlien%20analysis.png)|
| Churn Analysis                               | ![Screenshot](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/churn_analysis.png)          |



## Usage Instructions

### Getting Started

- **Clone the Repository**

  ```` 
   git clone https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights.git
   ````

- Install the project dependencies by running the following command

   ```
   pip install -r requirements.txt
   ```

- Navigate through the project's directories and files to get acquainted with its structure.


- **Dataset Handling:**

   - Load the dataset using Python, R, or your preferred data analysis tool.
   - Explore and preprocess data to prepare for churn prediction modeling.

- **Model Development:**

   - Train and evaluate machine learning models to predict customer churn.
   - Fine-tune models based on performance metrics to optimize predictions.

---

## Running the Project

- **Start the Application:** Execute the following command to run the project.

   ```bash
   streamlit run app.py
   ```

**After you have successfully installed and launched the project, you can utilize it to forecast customer churn. Follow these steps to begin:**

- **Access the Web Interface:** Open your web browser and navigate to:

   ```
   http://localhost:8501/
   ```

**Upon accessing the interface, you'll encounter a user-friendly web application. Take time to familiarize yourself with its features and explore the various options available for predicting outcomes.**


| Details | Prediction |
|---------|---------|
| ![Image 1](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/details.png) | ![Image 2](https://github.com/MuhammadUmerKhan/Customer-Churn-Prediction-with-NLP-Insights/blob/main/insights/prediction.png) |



- **Input Customer Data:** Refer to the on-screen instructions for guidance on entering customer data into the system. Once entered, explore the prediction feature to forecast churn outcomes based on the data provided.
  
- **Analyze Results:** Review the prediction outcomes to gather insights and formulate strategies, such as designing targeted promotions, and effectively applying the project to specific scenarios, such as implementing enticing customer offers.
  
---

Experience the deployed version of the Telecom Churn Prediction project!


### Dataset Usage:

1. Load the dataset using your preferred programming environment (e.g., Python with Pandas).
2. Explore and analyze the dataset attributes and patterns.
3. Utilize the dataset for Telco Customer Churn Classification analysis.

Feel free to adapt the above steps based on your specific use case and programming environment.

---

## Conclusion
By leveraging comprehensive churn analysis, robust predictive models, and interactive Power BI visualizations, the telecommunications company can proactively identify at-risk customers, understand churn drivers, and enhance service offerings. This approach fosters customer retention, reduces acquisition costs, and supports sustainable growth. Ultimately, the integration of predictive insights into strategic decision-making enables companies to maintain a competitive edge in a dynamic industry.
