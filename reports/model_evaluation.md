# Model Evaluation

## Project

**SAP BASIS Job Failure Prediction**

This report summarizes the machine learning models evaluated for predicting whether an SAP background job is likely to fail.

## Dataset

The project uses a simulated SAP BASIS background-job dataset containing:

- **10,000 job executions**
- **5,086 successful jobs**
- **4,914 failed jobs**
- **49.14% overall failure rate**

The target variable is:

`job_status`

Possible values:

- `Success`
- `Failed`

## Models Evaluated

Three classification models were evaluated:

1. Gradient Boosting
2. Logistic Regression
3. Random Forest

## Model Comparison

| Model | Accuracy | Failed Recall | Failed F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.6810 | 0.62 | 0.66 | 0.756536 |
| Logistic Regression | 0.6685 | 0.63 | 0.65 | 0.742177 |
| Random Forest | 0.6780 | 0.63 | 0.66 | 0.738465 |

## Gradient Boosting Evaluation

The Gradient Boosting model achieved:

- Accuracy: **68.10%**
- Failed-job recall: **62%**
- Failed-job F1-score: **0.66**
- ROC-AUC: **0.756536**

The Gradient Boosting model was selected for the dashboard prediction pipeline.

## Important Features

The model analysis identified operational variables such as:

- Expected data volume
- Average runtime
- CPU utilization
- Memory utilization
- Previous failure count
- Start delay
- Dependency status

These variables represent operational conditions that can be useful when estimating the risk of an SAP background-job failure.

## Prediction Pipeline

The dashboard uses the following workflow:

```text
Job Conditions
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