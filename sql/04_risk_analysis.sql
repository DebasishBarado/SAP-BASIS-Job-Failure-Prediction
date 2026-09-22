-- SAP BASIS Job Failure Prediction
-- Risk Analysis

USE sap_basis_jobs;


-- 1. Identify jobs with multiple risk indicators
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
WHERE cpu_usage_percent >= 70
   OR memory_usage_percent >= 70
   OR previous_failure_count >= 5
   OR start_delay_minutes >= 20
   OR dependency_status = 'Not Ready'
ORDER BY
    previous_failure_count DESC,
    start_delay_minutes DESC;


-- 2. Count jobs with critical resource usage
SELECT
    COUNT(*) AS high_resource_jobs
FROM job_executions
WHERE cpu_usage_percent >= 80
   OR memory_usage_percent >= 80;


-- 3. Failed jobs with unavailable dependencies
SELECT
    COUNT(*) AS failed_dependency_jobs
FROM job_executions
WHERE job_status = 'Failed'
  AND dependency_status = 'Not Ready';


-- 4. Failed jobs with high CPU and memory usage
SELECT
    COUNT(*) AS high_resource_failures
FROM job_executions
WHERE job_status = 'Failed'
  AND cpu_usage_percent >= 80
  AND memory_usage_percent >= 80;


-- 5. High-risk job summary
SELECT
    COUNT(*) AS high_risk_jobs
FROM job_executions
WHERE (
        cpu_usage_percent >= 80
        AND memory_usage_percent >= 80
      )
   OR previous_failure_count >= 8
   OR start_delay_minutes >= 30
   OR dependency_status = 'Not Ready';


-- 6. Compare average conditions for successful and failed jobs
SELECT
    job_status,
    ROUND(AVG(cpu_usage_percent), 2) AS avg_cpu,
    ROUND(AVG(memory_usage_percent), 2) AS avg_memory,
    ROUND(AVG(previous_failure_count), 2) AS avg_previous_failures,
    ROUND(AVG(start_delay_minutes), 2) AS avg_start_delay
FROM job_executions
GROUP BY job_status;