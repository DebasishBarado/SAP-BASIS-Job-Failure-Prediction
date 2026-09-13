import pandas as pd
import joblib

def get_percentage_input(message):
    while True:
        try:
            value = float(input(message))

            if 0 <= value <= 100:
                return value

            print("Please enter a value between 0 and 100.")

        except ValueError:
            print("Please enter a valid number.")

def get_dependency_status():
    while True:
        status = input(
            "Enter dependency status (Ready / Delayed / Not Ready): "
        ).strip().lower()

        if status == "ready":
            return "Ready"

        elif status == "delayed":
            return "Delayed"

        elif status == "not ready":
            return "Not Ready"

        else:
            print("Please enter Ready, Delayed, or Not Ready.")

# Load trained model
model = joblib.load("models/gradient_boosting_model.pkl")

# Load preprocessing transformer
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model and preprocessor loaded successfully")

# New SAP job data
cpu_usage = get_percentage_input("Enter CPU usage (%): ")
memory_usage = get_percentage_input("Enter memory usage (%): ")
previous_failures = int(input("Enter previous failure count: "))
start_delay = int(input("Enter start delay (minutes): "))

dependency_status = get_dependency_status()

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