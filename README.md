# SAP BASIS Job Failure Prediction

An AI/ML-based SAP BASIS background-job failure risk prediction system that analyzes job execution conditions and estimates the probability of job failure before execution.

## Project Overview

SAP BASIS administrators regularly monitor background jobs that perform tasks such as reporting, data processing, backups, and scheduled operations.

A job can fail because of conditions such as:

- High CPU utilization
- High memory utilization
- Previous job failures
- Start delays
- Large expected data volume
- Unavailable job dependencies

This project uses machine learning to analyze these operational conditions and estimate the failure risk of an SAP background job.

The system provides the predicted job status, failure probability, risk level, and an operational recommendation through an interactive dashboard.

## Project Objective

The main objective is to demonstrate how AI and machine learning can be applied to SAP BASIS operational data to provide an early-warning and decision-support system for background-job monitoring.

## Key Features

- Simulated SAP BASIS background-job dataset
- 10,000 job execution records
- Data analysis and preprocessing
- Multiple machine learning models
- Gradient Boosting prediction model
- Failure probability estimation
- LOW, MEDIUM, and HIGH risk classification
- SAP BASIS-focused operational recommendations
- Interactive Streamlit dashboard
- Job monitoring and analytics
- SQL analysis queries
- Model evaluation reports
- Prediction history
- Saved machine learning model and preprocessing pipeline

## System Workflow

```text
SAP Job Conditions
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
Failure Probability
        ↓
Risk Classification
        ↓
BASIS Recommendation
        ↓
Dashboard
