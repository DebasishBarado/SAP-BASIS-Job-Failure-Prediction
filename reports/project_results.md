# Project Results

## Project

**SAP BASIS Job Failure Prediction**

This project demonstrates an AI/ML-based approach for estimating the failure risk of SAP BASIS background jobs before execution.

The project combines simulated SAP background-job data, machine learning, preprocessing, SQL analysis, and an interactive Streamlit dashboard.

## Dataset Results

The project uses a simulated dataset containing:

- **10,000 job executions**
- **5,086 successful jobs**
- **4,914 failed jobs**
- **49.14% overall failure rate**
- **15 columns**
- **0 missing values**

The dataset contains operational information such as CPU usage, memory usage, previous failures, start delays, expected data volume, and dependency status.

## Machine Learning Results

Three classification models were evaluated:

- Gradient Boosting
- Logistic Regression
- Random Forest

The evaluation results were:

| Model | Accuracy | Failed Recall | Failed F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.6810 | 0.62 | 0.66 | 0.756536 |
| Logistic Regression | 0.6685 | 0.63 | 0.65 | 0.742177 |
| Random Forest | 0.6780 | 0.63 | 0.66 | 0.738465 |

The Gradient Boosting model was selected for the dashboard prediction pipeline.

## Final Prediction Pipeline

The final system follows this process:

```text
SAP Job Conditions
        ↓
Feature Preprocessing
        ↓
Gradient Boosting Model
        ↓
Failure Probability
        ↓
Risk Classification
        ↓
BASIS Recommendation