# Dataset Analysis

## Project

**SAP BASIS Job Failure Prediction**

This report describes the simulated SAP BASIS background-job dataset used for training and evaluating the machine learning models.

## Dataset Overview

The dataset contains **10,000 SAP background-job executions**.

It contains **15 columns** representing job information, scheduling conditions, historical job performance, system resource usage, dependency conditions, and the final job status.

### Dataset Size

- Total records: **10,000**
- Total features and target columns: **15**
- Missing values: **0**
- Target variable: `job_status`

## Dataset Columns

| Column | Description |
|---|---|
| `execution_id` | Unique identifier for each job execution |
| `job_id` | Identifier of the SAP background job |
| `job_name` | Name of the background job |
| `job_type` | Type of job execution |
| `scheduled_hour` | Hour at which the job is scheduled |
| `day_of_week` | Day on which the job is scheduled |
| `start_delay_minutes` | Delay before the job starts |
| `previous_success_count` | Number of previous successful executions |
| `previous_failure_count` | Number of previous failed executions |
| `average_runtime_minutes` | Average execution time of the job |
| `expected_data_volume` | Expected amount of data processed by the job |
| `cpu_usage_percent` | CPU utilization associated with the job |
| `memory_usage_percent` | Memory utilization associated with the job |
| `dependency_status` | Status of required job dependencies |
| `job_status` | Final execution result: Success or Failed |

## Target Variable Distribution

The target variable is `job_status`.

The dataset contains:

- **5,086 successful jobs**
- **4,914 failed jobs**

The overall failure rate is **49.14%**.

This provides a relatively balanced distribution between successful and failed job executions for classification modeling.

## Failure Rate by Dependency Status

Dependency status showed a noticeable relationship with job failures.

| Dependency Status | Failure Rate |
|---|---:|
| Not Ready | 81.84% |
| Delayed | 64.41% |
| Ready | 39.81% |

Jobs with dependencies marked as **Not Ready** had a higher observed failure rate than jobs whose dependencies were **Ready**.

This indicates that dependency availability is an important operational condition for SAP background-job execution.

## Resource Usage Analysis

CPU and memory utilization were also analyzed based on job status.

### Average CPU Usage

| Job Status | Average CPU Usage |
|---|---:|
| Failed | 60.11% |
| Success | 54.34% |

Failed jobs had a higher average CPU utilization than successful jobs in the simulated dataset.

### Average Memory Usage

| Job Status | Average Memory Usage |
|---|---:|
| Failed | 59.84% |
| Success | 54.70% |

Failed jobs also showed higher average memory utilization than successful jobs.

## Historical Failure Analysis

Previous job failures were analyzed as another operational indicator.

| Job Status | Average Previous Failures |
|---|---:|
| Failed | 5.09 |
| Success | 4.10 |

In the simulated dataset, failed jobs had a higher average number of previous failures.

This historical information can help the prediction system identify jobs that may require additional monitoring.

## Start Delay Analysis

Job start delays were also compared.

| Job Status | Average Start Delay |
|---|---:|
| Failed | 15.94 minutes |
| Success | 13.93 minutes |

Failed jobs showed a higher average start delay than successful jobs.

Start delay can therefore be considered as one of the operational conditions used by the machine learning pipeline.

## Key Dataset Observations

The analysis identified several variables that are useful for understanding job-failure behavior:

- Dependency status
- CPU utilization
- Memory utilization
- Previous failure count
- Start delay
- Expected data volume
- Average runtime

These variables were included in the machine learning feature set because they represent operational conditions that may influence SAP background-job execution.

## Data Preparation

Before model training, the dataset was divided into:

- **80% training data**
- **20% testing data**

This resulted in:

- **8,000 training records**
- **2,000 testing records**

The split used stratification to maintain a similar distribution of successful and failed jobs in both datasets.

Categorical variables were processed using the preprocessing pipeline, while numerical variables were passed through the appropriate numerical preprocessing steps.

## Conclusion

The dataset analysis provides an overview of the operational characteristics of the simulated SAP BASIS background-job executions.

The analysis indicates that conditions such as dependency availability, resource utilization, previous failures, and start delays can provide useful information for estimating job-failure risk.

These observations form the foundation for the machine learning models used in the SAP BASIS Job Failure Prediction project.