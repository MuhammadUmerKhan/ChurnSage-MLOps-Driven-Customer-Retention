# Telecom Customer Churn Attrition Prediction

![churn 1.png](https://miro.medium.com/v2/resize:fit:1024/1*TgciopaOk-C8fwtPmmet3w.png)

In today's competitive telecom industry, retaining customers is crucial. Customer churn, where users leave for another provider, can significantly impact a company's bottom line. This project focuses on building a predictive model to identify customers likely to leave based on usage patterns, behavior, and other factors. By understanding these patterns, companies can take proactive measures to enhance customer satisfaction and reduce churn.

This repository contains all the necessary components for analyzing customer churn, from the dataset to the code for predictions. It also offers insights into key factors driving customer attrition, suggests strategies for improving retention, and explores the role of natural language processing (NLP) in understanding customer sentiments.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Data Insights](#data-insights)
- [Power BI Visuals](#power-bi-visuals)
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

5. **Power BI for Visual Insights:**
   - Implemented Power BI to create interactive dashboards and visualizations that provide intuitive insights into churn patterns.
   - Visualized model predictions, feature importance, customer segmentation, and NLP findings to facilitate strategic decision-making and operational planning.

---

## Data Insights

Explore profound insights and analytics gleaned from our extensive dataset. Uncover a deeper understanding of customer behaviors, patterns in service usage, and the pivotal factors influencing churn dynamics.

| Feature                                      | Visualization                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Categorical Features                         | ![Categorical Features](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/ce2e270e-2118-41b5-8207-1fccd2e98982)   |
| Churn Target Variable                        | ![Churn Target Variable](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/681e2805-d61e-4d56-be55-fa0495a5bfd5)  |
| Customer Information                         | ![Customer Information](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/234d902f-f514-4d2f-b28a-6d5813c67909)   |
| Distribution Analysis                        | ![Distribution Analysis](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/e72dbd50-9c94-4c44-bf89-1b3baa090a64)   |
| Mean Tenure                                  | ![Mean Tenure](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/0f8feb7e-d723-4061-beb6-62275d6a54b9)         |
| Churn Tenure Analysis                        | ![Screenshot](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/5942dbb4-47c7-467e-a075-14dc47ca7572)          |

---

## Power BI Visuals

Explore interactive Power BI visualizations designed to enhance data exploration and decision-making. Visualize customer churn trends, contract preferences, and revenue impact through intuitive and actionable dashboards.

| Overview and Key Metrics | Attrition Insights |
|---------------|-------------|
| ![Overview](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/18855f1b-fefe-4317-bede-fb8219d67e9f) | ![Detailed Insights](https://github.com/virajbhutada/Telecom-Customer-Churn-Prediction/assets/143819712/7c068b03-4cdb-4659-b3e3-c15dc481cd59) |
| Gain a high-level view of customer churn trends, contract preferences, and revenue impact. This interactive dashboard provides insights into overall churn metrics and key business indicators. | Explore detailed analytics on customer segments, service usage patterns, and churn predictors. This visualization offers a deeper dive into specific data points and trends influencing churn decisions. |

---

## Key Findings

### Strategic Insights

- **Customer Segmentation:** Identify high-risk customer segments prone to churn.
- **Service Optimization:** Evaluate the impact of service features and contract terms on customer retention.
- **Sentiment Correlation:** Utilize NLP insights to correlate customer sentiment with churn rates, highlighting areas for service improvement.
- **Financial Impact:** Quantify revenue loss due to churn and explore strategies for revenue recovery.

---

## References

- [Average Customer Acquisition Cost by Industry](https://hockeystack.com/blog/average-customer-acquisition-cost-by-industry/)
- [Subscriber Acquisition Cost Examples](https://www.klipfolio.com/resources/kpi-examples/call-center/subscriber-acquisition-cost)
- [Understanding Customer Churn Rate](https://www.zendesk.com/in/blog/customer-churn-rate/#georedirect)
- [Churn Prevention Strategies](https://www.profitwell.com/customer-churn/churn-prevention)

---

## Usage Instructions

### Getting Started

- **Clone the Repository**

   ```bash
   git clone https://github.com/adityakapole/Telecom-Customer-Churn-Prediction.git
   cd Telecom-Customer-Churn-Prediction
