-- SAP BASIS Job Failure Prediction
-- Job Analysis Queries

USE sap_basis_jobs;

-- 1. Total number of job executions
SELECT COUNT(*) AS total_jobs
FROM job_executions;


-- 2. Successful and failed jobs
SELECT
    job_status,
    COUNT(*) AS job_count
FROM job_executions
GROUP BY job_status;


-- 3. Overall failure rate
SELECT
    ROUND(
        SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*),
        2
    ) AS failure_rate_percent
FROM job_executions;


-- 4. Jobs by job type
SELECT
    job_type,
    COUNT(*) AS job_count
FROM job_executions
GROUP BY job_type
ORDER BY job_count DESC;


-- 5. Failure rate by dependency status
SELECT
    dependency_status,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END) AS failed_jobs,
    ROUND(
        SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*),
        2
    ) AS failure_rate_percent
FROM job_executions
GROUP BY dependency_status
ORDER BY failure_rate_percent DESC;


-- 6. Average CPU and memory usage by job status
SELECT
    job_status,
    ROUND(AVG(cpu_usage_percent), 2) AS average_cpu,
    ROUND(AVG(memory_usage_percent), 2) AS average_memory
FROM job_executions
GROUP BY job_status;


-- 7. Average previous failures by job status
SELECT
    job_status,
    ROUND(AVG(previous_failure_count), 2) AS average_previous_failures
FROM job_executions
GROUP BY job_status;


-- 8. Average start delay by job status
SELECT
    job_status,
    ROUND(AVG(start_delay_minutes), 2) AS average_start_delay
FROM job_executions
GROUP BY job_status;