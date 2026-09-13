import pandas as pd
import numpy as np


# ============================================================
# Part 1: Configuration
# ============================================================

NUM_RECORDS = 10000

# Make generated data reproducible
np.random.seed(42)


# ============================================================
# Part 2: Job Information
# ============================================================

jobs = [
    ("JOB001", "DAILY_SALES"),
    ("JOB002", "MONTHLY_PAYROLL"),
    ("JOB003", "INVENTORY_UPDATE"),
    ("JOB004", "DATA_TRANSFER"),
    ("JOB005", "BACKUP_JOB"),
    ("JOB006", "FINANCE_REPORT"),
    ("JOB007", "CUSTOMER_REPORT"),
    ("JOB008", "SALES_REPORT"),
    ("JOB009", "STOCK_UPDATE"),
    ("JOB010", "ORDER_PROCESSING"),
    ("JOB011", "EMPLOYEE_DATA_SYNC"),
    ("JOB012", "DATABASE_BACKUP"),
    ("JOB013", "LOG_CLEANUP"),
    ("JOB014", "MASTER_DATA_SYNC"),
    ("JOB015", "DAILY_ANALYTICS"),
    ("JOB016", "WAREHOUSE_UPDATE"),
    ("JOB017", "INVOICE_PROCESSING"),
    ("JOB018", "VENDOR_DATA_SYNC"),
    ("JOB019", "SYSTEM_MONITORING"),
    ("JOB020", "ARCHIVE_JOB")
]

job_types = [
    "Reporting",
    "Data Transfer",
    "Backup",
    "Data Processing",
    "Synchronization"
]


# ============================================================
# Part 3: Scheduling Information
# ============================================================

scheduled_hour = np.random.randint(
    0,
    24,
    NUM_RECORDS
)

day_of_week = np.random.choice(
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ],
    NUM_RECORDS
)

start_delay_minutes = np.random.randint(
    0,
    31,
    NUM_RECORDS
)


# ============================================================
# Part 4: Historical Job Performance
# ============================================================

previous_success_count = np.random.randint(
    0,
    30,
    NUM_RECORDS
)

previous_failure_count = np.random.randint(
    0,
    10,
    NUM_RECORDS
)

average_runtime_minutes = np.random.randint(
    5,
    121,
    NUM_RECORDS
)


# ============================================================
# Part 5: Workload and System Conditions
# ============================================================

expected_data_volume = np.random.randint(
    1000,
    500001,
    NUM_RECORDS
)

cpu_usage_percent = np.random.randint(
    20,
    96,
    NUM_RECORDS
)

memory_usage_percent = np.random.randint(
    20,
    96,
    NUM_RECORDS
)


# ============================================================
# Part 6: Dependency Status
# ============================================================

dependency_status = np.random.choice(
    [
        "Ready",
        "Delayed",
        "Not Ready"
    ],
    NUM_RECORDS,
    p=[0.70, 0.20, 0.10]
)


# ============================================================
# Part 7: Failure Risk Calculation
# ============================================================

# Start with a small base probability
failure_risk = np.full(
    NUM_RECORDS,
    0.05
)


# ------------------------------------------------------------
# Previous failures
# More previous failures = higher future failure risk
# ------------------------------------------------------------

failure_risk += (
    previous_failure_count / 9
) * 0.20


# ------------------------------------------------------------
# Previous successful executions
# More successful executions = slightly lower risk
# ------------------------------------------------------------

failure_risk -= (
    previous_success_count / 29
) * 0.08


# ------------------------------------------------------------
# CPU usage
# High CPU creates higher failure risk
# ------------------------------------------------------------

failure_risk += np.where(
    cpu_usage_percent >= 80,
    0.18,
    np.where(
        cpu_usage_percent >= 65,
        0.08,
        0
    )
)


# ------------------------------------------------------------
# Memory usage
# High memory creates higher failure risk
# ------------------------------------------------------------

failure_risk += np.where(
    memory_usage_percent >= 80,
    0.18,
    np.where(
        memory_usage_percent >= 65,
        0.08,
        0
    )
)


# ------------------------------------------------------------
# Large data volume
# ------------------------------------------------------------

failure_risk += np.where(
    expected_data_volume >= 400000,
    0.15,
    np.where(
        expected_data_volume >= 250000,
        0.07,
        0
    )
)


# ------------------------------------------------------------
# Start delay
# Large scheduling delay = higher risk
# ------------------------------------------------------------

failure_risk += np.where(
    start_delay_minutes >= 20,
    0.15,
    np.where(
        start_delay_minutes >= 10,
        0.06,
        0
    )
)


# ------------------------------------------------------------
# Average runtime
# Long-running jobs have higher risk
# ------------------------------------------------------------

failure_risk += np.where(
    average_runtime_minutes >= 90,
    0.12,
    np.where(
        average_runtime_minutes >= 60,
        0.05,
        0
    )
)


# ============================================================
# Part 8: Dependency Risk
# ============================================================

# Ready = low additional risk
# Delayed = medium risk
# Not Ready = high risk

failure_risk += np.where(
    dependency_status == "Delayed",
    0.18,
    0
)

failure_risk += np.where(
    dependency_status == "Not Ready",
    0.40,
    0
)


# ============================================================
# Part 9: Combined Risk Conditions
# ============================================================

# High CPU + High memory
failure_risk += np.where(
    (cpu_usage_percent >= 80) &
    (memory_usage_percent >= 80),
    0.12,
    0
)


# Previous failures + dependency problem
failure_risk += np.where(
    (previous_failure_count >= 5) &
    (dependency_status != "Ready"),
    0.15,
    0
)


# High delay + high CPU
failure_risk += np.where(
    (start_delay_minutes >= 20) &
    (cpu_usage_percent >= 80),
    0.10,
    0
)


# Large workload + long runtime
failure_risk += np.where(
    (expected_data_volume >= 400000) &
    (average_runtime_minutes >= 90),
    0.10,
    0
)


# ============================================================
# Part 10: Add Small Natural Randomness
# ============================================================

failure_risk += np.random.normal(
    0,
    0.03,
    NUM_RECORDS
)


# Keep probability between 2% and 95%
failure_risk = np.clip(
    failure_risk,
    0.02,
    0.95
)


# ============================================================
# Part 11: Generate Actual Job Status
# ============================================================

job_status = np.where(
    np.random.random(NUM_RECORDS) < failure_risk,
    "Failed",
    "Success"
)


# ============================================================
# Part 12: Select Jobs
# ============================================================

selected_jobs = np.random.choice(
    len(jobs),
    NUM_RECORDS
)


# ============================================================
# Part 13: Create DataFrame
# ============================================================

df = pd.DataFrame({

    "execution_id": range(
        1,
        NUM_RECORDS + 1
    ),

    "job_id": [
        jobs[i][0]
        for i in selected_jobs
    ],

    "job_name": [
        jobs[i][1]
        for i in selected_jobs
    ],

    "job_type": np.random.choice(
        job_types,
        NUM_RECORDS
    ),

    "scheduled_hour": scheduled_hour,

    "day_of_week": day_of_week,

    "start_delay_minutes": start_delay_minutes,

    "previous_success_count": previous_success_count,

    "previous_failure_count": previous_failure_count,

    "average_runtime_minutes": average_runtime_minutes,

    "expected_data_volume": expected_data_volume,

    "cpu_usage_percent": cpu_usage_percent,

    "memory_usage_percent": memory_usage_percent,

    "dependency_status": dependency_status,

    "job_status": job_status
})


# ============================================================
# Part 14: Save Dataset
# ============================================================

output_path = "data/raw/sap_job_data.csv"

df.to_csv(
    output_path,
    index=False
)

print(
    f"Dataset created successfully: {output_path}"
)

print(
    f"Total records: {len(df)}"
)


# ============================================================
# Part 15: Validate Dataset
# ============================================================

print("\nFirst 5 records:")
print(df.head())


print("\nDataset shape:")
print(df.shape)


print("\nJob status distribution:")
print(df["job_status"].value_counts())


print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# Part 16: Failure Rate by Dependency
# ============================================================

print("\nFailure rate by dependency status:")

dependency_failure_rate = (
    df.groupby("dependency_status")["job_status"]
    .apply(lambda x: (x == "Failed").mean() * 100)
    .sort_values(ascending=False)
)

print(dependency_failure_rate)


# ============================================================
# Part 17: Average Failure Risk Conditions
# ============================================================

print("\nAverage CPU by job status:")
print(
    df.groupby("job_status")["cpu_usage_percent"].mean()
)

print("\nAverage memory by job status:")
print(
    df.groupby("job_status")["memory_usage_percent"].mean()
)

print("\nAverage previous failures by job status:")
print(
    df.groupby("job_status")["previous_failure_count"].mean()
)

print("\nAverage start delay by job status:")
print(
    df.groupby("job_status")["start_delay_minutes"].mean()
)