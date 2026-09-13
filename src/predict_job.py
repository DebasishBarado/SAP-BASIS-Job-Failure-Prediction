import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/gradient_boosting_model.pkl")

# Load preprocessing transformer
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model and preprocessor loaded successfully")

# New SAP job data
new_job = pd.DataFrame([{
    "job_id": "JOB001",
    "job_name": "DAILY_SALES",
    "job_type": "Reporting",
    "scheduled_hour": 10,
    "day_of_week": "Monday",
    "start_delay_minutes": 20,
    "previous_success_count": 15,
    "previous_failure_count": 6,
    "average_runtime_minutes": 60,
    "expected_data_volume": 300000,
    "cpu_usage_percent": 80,
    "memory_usage_percent": 75,
    "dependency_status": "Delayed"
}])

print("\nNew job data:")
print(new_job)