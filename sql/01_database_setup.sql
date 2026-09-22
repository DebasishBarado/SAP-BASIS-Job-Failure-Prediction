-- SAP BASIS Job Failure Prediction
-- Database Setup

CREATE DATABASE IF NOT EXISTS sap_basis_jobs;

USE sap_basis_jobs;

CREATE TABLE IF NOT EXISTS job_executions (
    execution_id INT PRIMARY KEY,
    job_id VARCHAR(20),
    job_name VARCHAR(100),
    job_type VARCHAR(50),
    scheduled_hour INT,
    day_of_week VARCHAR(20),
    start_delay_minutes INT,
    previous_success_count INT,
    previous_failure_count INT,
    average_runtime_minutes FLOAT,
    expected_data_volume FLOAT,
    cpu_usage_percent FLOAT,
    memory_usage_percent FLOAT,
    dependency_status VARCHAR(20),
    job_status VARCHAR(20)
);