-- SAP BASIS Job Failure Prediction
-- Detailed Failure Analysis

USE sap_basis_jobs;


-- 1. Jobs with the highest number of previous failures
SELECT
    job_id,
    job_name,
    previous_failure_count,
    job_status
FROM job_executions
ORDER BY previous_failure_count DESC
LIMIT 20;


-- 2. Jobs with high CPU usage
SELECT
    job_id,
    job_name,
    cpu_usage_percent,
    job_status
FROM job_executions
WHERE cpu_usage_percent >= 80
ORDER BY cpu_usage_percent DESC;


-- 3. Jobs with high memory usage
SELECT
    job_id,
    job_name,
    memory_usage_percent,
    job_status
FROM job_executions
WHERE memory_usage_percent >= 80
ORDER BY memory_usage_percent DESC;


-- 4. Jobs with high start delay
SELECT
    job_id,
    job_name,
    start_delay_minutes,
    job_status
FROM job_executions
WHERE start_delay_minutes >= 30
ORDER BY start_delay_minutes DESC;


-- 5. Failed jobs with multiple risk conditions
SELECT
    job_id,
    job_name,
    cpu_usage_percent,
    memory_usage_percent,
    previous_failure_count,
    start_delay_minutes,
    dependency_status,
    job_status
FROM job_executions
WHERE cpu_usage_percent >= 80
  AND memory_usage_percent >= 80
  AND previous_failure_count >= 5
  AND start_delay_minutes >= 20
ORDER BY previous_failure_count DESC;


-- 6. Failure rate by job name
SELECT
    job_name,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END) AS failed_jobs,
    ROUND(
        SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*),
        2
    ) AS failure_rate_percent
FROM job_executions
GROUP BY job_name
ORDER BY failure_rate_percent DESC;


-- 7. Failure rate by day of week
SELECT
    day_of_week,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END) AS failed_jobs,
    ROUND(
        SUM(CASE WHEN job_status = 'Failed' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*),
        2
    ) AS failure_rate_percent
FROM job_executions
GROUP BY day_of_week
ORDER BY failure_rate_percent DESC;