import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/gradient_boosting_model.pkl")

# Load preprocessing transformer
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model and preprocessor loaded successfully")

# New SAP job data
cpu_usage = float(input("Enter CPU usage (%): "))
memory_usage = float(input("Enter memory usage (%): "))
previous_failures = int(input("Enter previous failure count: "))
start_delay = int(input("Enter start delay (minutes): "))

dependency_status = input("Enter dependency status (Ready / Delayed / Not Ready): ")

new_job = pd.DataFrame([{
    "job_id": "JOB001",
    "job_name": "DAILY_SALES",
    "job_type": "Reporting",
    "scheduled_hour": 10,
    "day_of_week": "Monday",
    "start_delay_minutes": start_delay,
    "previous_success_count": 15,
    "previous_failure_count": previous_failures,
    "average_runtime_minutes": 60,
    "expected_data_volume": 300000,
    "cpu_usage_percent": cpu_usage,
    "memory_usage_percent": memory_usage,
    "dependency_status": dependency_status
}])

print("\nNew job data:")
print(new_job)

# Preprocess the new job data
new_job_processed = preprocessor.transform(new_job)

# Predict job status
prediction = model.predict(new_job_processed)[0]

print("\nPredicted Job Status:", prediction)

# Get failure probability
failure_probability = model.predict_proba(new_job_processed)[0][0]

print("\nPredicted Job Status:", prediction)
print("Failure Probability:", round(failure_probability * 100, 2), "%")

# Determine risk level
if failure_probability >= 0.70:
    risk_level = "HIGH"
elif failure_probability >= 0.40:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

print("Risk Level:", risk_level)

# Generate recommendation
if risk_level == "HIGH":
    recommendation = (
        "Check dependency status, CPU/memory usage, "
        "previous failures, and start delay before execution."
    )
elif risk_level == "MEDIUM":
    recommendation = (
        "Monitor the job closely and check recent job performance."
    )
else:
    recommendation = (
        "Job appears low risk. Continue normal monitoring."
    )

print("Recommendation:", recommendation)