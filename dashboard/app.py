
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="SAP BASIS AI Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "raw" / "sap_job_data.csv"

REQUIRED_MODEL_COLUMNS = [
    "job_id",
    "job_name",
    "job_type",
    "scheduled_hour",
    "day_of_week",
    "start_delay_minutes",
    "previous_success_count",
    "previous_failure_count",
    "average_runtime_minutes",
    "expected_data_volume",
    "cpu_usage_percent",
    "memory_usage_percent",
    "dependency_status",
]

STATUS_COLUMN_CANDIDATES = ["job_status", "status", "prediction", "result"]


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(37,99,235,.10), transparent 28%),
            radial-gradient(circle at 95% 10%, rgba(14,165,233,.08), transparent 25%),
            #07111f;
        color: #e5edf7;
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.82);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .topbar {
        background: linear-gradient(135deg, #0b1b31 0%, #102b4d 58%, #123a59 100%);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 20px;
        padding: 22px 26px;
        box-shadow: 0 18px 50px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05);
        margin-bottom: 16px;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -.5px;
        color: #f8fbff;
    }

    .brand span {
        color: #42d392;
    }

    .subtitle {
        color: #a9bad0;
        margin-top: 5px;
        font-size: 14px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #f4f8fd;
        margin: 20px 0 12px 2px;
    }

    .section-note {
        color: #8fa4bc;
        font-size: 13px;
        margin: -4px 0 14px 2px;
    }

    .card {
        background: linear-gradient(145deg, rgba(18,34,55,.94), rgba(10,24,41,.94));
        border: 1px solid rgba(148,163,184,.15);
        border-radius: 17px;
        padding: 19px;
        box-shadow: 0 14px 38px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.035);
        min-height: 125px;
    }

    .card:hover {
        border-color: rgba(66,211,146,.28);
        box-shadow: 0 18px 44px rgba(0,0,0,.28), 0 0 0 1px rgba(66,211,146,.05);
    }

    .kpi-label {
        color: #91a5bc;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .kpi-value {
        color: #f8fbff;
        font-size: 29px;
        font-weight: 850;
        margin-top: 8px;
    }

    .kpi-help {
        color: #6f849d;
        font-size: 12px;
        margin-top: 5px;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(16,33,54,.98), rgba(8,21,37,.98));
        border: 1px solid rgba(148,163,184,.15);
        border-radius: 17px;
        padding: 18px;
        box-shadow: 0 12px 32px rgba(0,0,0,.20);
    }

    .metric-head {
        display:flex;
        justify-content:space-between;
        align-items:center;
        color:#a8bbd0;
        font-size:13px;
        font-weight:700;
    }

    .metric-number {
        color:#f8fbff;
        font-size:30px;
        font-weight:850;
        margin:8px 0 10px;
    }

    .track {
        height: 9px;
        background:#17283d;
        border-radius:999px;
        overflow:hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,.35);
    }

    .fill {
        height:100%;
        border-radius:999px;
        background: linear-gradient(90deg,#2dd4bf,#42d392);
        box-shadow: 0 0 14px rgba(66,211,146,.25);
    }

    .fill.warn {
        background: linear-gradient(90deg,#f59e0b,#fbbf24);
        box-shadow: 0 0 14px rgba(245,158,11,.20);
    }

    .fill.danger {
        background: linear-gradient(90deg,#ef4444,#fb7185);
        box-shadow: 0 0 14px rgba(239,68,68,.22);
    }

    .status-pill {
        display:inline-flex;
        align-items:center;
        gap:7px;
        padding:7px 12px;
        border-radius:999px;
        font-weight:800;
        font-size:12px;
        letter-spacing:.3px;
    }

    .pill-green { background:rgba(34,197,94,.12); color:#4ade80; border:1px solid rgba(74,222,128,.20); }
    .pill-yellow { background:rgba(245,158,11,.12); color:#fbbf24; border:1px solid rgba(251,191,36,.20); }
    .pill-red { background:rgba(239,68,68,.12); color:#fb7185; border:1px solid rgba(251,113,133,.20); }
    .pill-blue { background:rgba(59,130,246,.12); color:#60a5fa; border:1px solid rgba(96,165,250,.20); }

    .hero {
        background: linear-gradient(135deg, #0b1d33 0%, #0e2b49 55%, #0d4050 100%);
        border:1px solid rgba(66,211,146,.17);
        border-radius:22px;
        padding:30px;
        box-shadow:0 22px 60px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.05);
        margin-bottom:20px;
    }

    .hero-kicker {
        color:#42d392;
        font-size:12px;
        font-weight:800;
        letter-spacing:1.2px;
        text-transform:uppercase;
    }

    .hero h1 {
        color:#f8fbff;
        font-size:38px;
        line-height:1.08;
        margin:8px 0 10px;
    }

    .hero p {
        color:#a8bbd0;
        font-size:15px;
        max-width:820px;
        line-height:1.65;
    }

    .prediction-card {
        background: linear-gradient(145deg, rgba(16,34,55,.99), rgba(7,20,35,.99));
        border:1px solid rgba(148,163,184,.17);
        border-radius:22px;
        padding:28px;
        box-shadow:0 22px 60px rgba(0,0,0,.30);
        text-align:center;
    }

    .prediction-status {
        font-size:36px;
        font-weight:900;
        margin:8px 0;
    }

    .prediction-percent {
        font-size:48px;
        font-weight:900;
        color:#f8fbff;
        line-height:1;
        margin:12px 0 8px;
    }

    .prediction-muted {
        color:#8196ad;
        font-size:12px;
    }

    .gauge {
        width:190px;
        height:190px;
        border-radius:50%;
        margin:8px auto 16px;
        display:flex;
        align-items:center;
        justify-content:center;
        position:relative;
        background: conic-gradient(var(--gauge-color) calc(var(--gauge-value) * 1%), #17283d 0);
        box-shadow:0 0 32px rgba(0,0,0,.24);
    }

    .gauge:before {
        content:"";
        width:146px;
        height:146px;
        border-radius:50%;
        background:#0a1727;
        position:absolute;
        box-shadow:inset 0 0 25px rgba(0,0,0,.45);
    }

    .gauge-content {
        position:relative;
        z-index:2;
        text-align:center;
    }

    .gauge-number {
        color:#f8fbff;
        font-size:30px;
        font-weight:900;
    }

    .gauge-label {
        color:#8da2b8;
        font-size:11px;
        text-transform:uppercase;
        letter-spacing:.8px;
    }

    .recommendation {
        border-radius:17px;
        padding:18px 20px;
        border:1px solid rgba(245,158,11,.20);
        background:rgba(245,158,11,.08);
    }

    .recommendation.high {
        border-color:rgba(239,68,68,.22);
        background:rgba(239,68,68,.08);
    }

    .recommendation.low {
        border-color:rgba(34,197,94,.20);
        background:rgba(34,197,94,.07);
    }

    .recommendation-title {
        color:#f8fbff;
        font-weight:850;
        margin-bottom:6px;
    }

    .recommendation-text {
        color:#a8bbd0;
        line-height:1.55;
        font-size:13px;
    }

    .explain {
        background:rgba(15,30,49,.82);
        border:1px solid rgba(148,163,184,.13);
        border-radius:15px;
        padding:16px;
        height:100%;
    }

    .explain strong { color:#eaf2fb; }
    .explain p { color:#8fa4bc; font-size:13px; line-height:1.55; margin:7px 0 0; }

    .footer {
        text-align:center;
        color:#60758d;
        font-size:11px;
        padding-top:28px;
        margin-top:35px;
        border-top:1px solid rgba(148,163,184,.10);
    }

    /* Streamlit controls */
    .stButton > button {
        border-radius:12px;
        border:1px solid rgba(66,211,146,.24);
        background:linear-gradient(135deg,#143b39,#11614f);
        color:white;
        font-weight:800;
        min-height:46px;
        box-shadow:0 9px 24px rgba(0,0,0,.22);
    }

    .stButton > button:hover {
        border-color:#42d392;
        transform:translateY(-1px);
        box-shadow:0 12px 28px rgba(66,211,146,.14);
    }

    [data-testid="stRadio"] label {
        color:#a8bbd0 !important;
        font-weight:700 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background:rgba(10,25,42,.75);
        border:1px dashed rgba(66,211,146,.30);
        border-radius:16px;
    }

    .small-note {
        color:#71869d;
        font-size:11px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model_and_preprocessor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PREPROCESSOR_PATH}")
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


try:
    model, preprocessor = load_model_and_preprocessor()
    model_ready = True
    model_error = ""
except Exception as exc:
    model = None
    preprocessor = None
    model_ready = False
    model_error = str(exc)


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data
def load_default_dataset(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


def normalize_status_series(series):
    return series.astype(str).str.strip().str.lower().map(
        lambda x: "Failed" if x in {"failed", "fail", "failure", "1", "true"}
        else "Success" if x in {"success", "successful", "succeeded", "0", "false"}
        else str(x).title()
    )


def dataset_status_column(df):
    for col in STATUS_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def dataset_summary(df):
    rows = len(df)
    cols = len(df.columns)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    status_col = dataset_status_column(df)
    if status_col:
        status = normalize_status_series(df[status_col])
        failed = int((status == "Failed").sum())
        success = int((status == "Success").sum())
    else:
        failed = 0
        success = 0
    return rows, cols, missing, duplicates, failed, success, status_col


def metric_status(value, kind):
    if kind in {"cpu", "memory"}:
        if value >= 80:
            return "danger", "High"
        if value >= 60:
            return "warn", "Moderate"
        return "good", "Normal"
    if kind == "delay":
        if value >= 20:
            return "danger", "High"
        if value >= 10:
            return "warn", "Moderate"
        return "good", "Low"
    if kind == "failures":
        if value >= 6:
            return "danger", "Elevated"
        if value >= 3:
            return "warn", "Moderate"
        return "good", "Low"
    return "good", "Normal"


def render_metric_card(title, value, kind, suffix=""):
    css, label = metric_status(float(value), kind)
    fill_class = "" if css == "good" else css
    width = min(max(float(value), 0), 100) if kind in {"cpu", "memory"} else min(max(float(value) * 5, 0), 100)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-head">
                <span>{title}</span>
                <span class="status-pill {'pill-green' if css == 'good' else 'pill-yellow' if css == 'warn' else 'pill-red'}">
                    {'●'} {label}
                </span>
            </div>
            <div class="metric-number">{value}{suffix}</div>
            <div class="track">
                <div class="fill {fill_class}" style="width:{width:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_details(probability):
    if probability >= 0.70:
        return "HIGH", "🔴", "#fb7185", "high"
    if probability >= 0.40:
        return "MEDIUM", "🟠", "#fbbf24", "medium"
    return "LOW", "🟢", "#4ade80", "low"


def get_failure_probability(model_obj, processed):
    probabilities = model_obj.predict_proba(processed)[0]
    classes = list(getattr(model_obj, "classes_", []))
    if "Failed" in classes:
        return float(probabilities[classes.index("Failed")])
    if "failed" in classes:
        return float(probabilities[classes.index("failed")])
    if len(probabilities) == 2:
        # If class names are unexpected, choose the class that is not Success.
        for i, cls in enumerate(classes):
            if str(cls).strip().lower() not in {"success", "successful", "succeeded"}:
                return float(probabilities[i])
        return float(probabilities[1])
    return float(probabilities[0])


def validate_prediction_inputs(df):
    missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in df.columns]
    return missing


def build_prediction_frame(
    job_id,
    job_name,
    job_type,
    scheduled_hour,
    day_of_week,
    start_delay,
    previous_successes,
    previous_failures,
    average_runtime,
    expected_volume,
    cpu,
    memory,
    dependency,
):
    return pd.DataFrame([{
        "job_id": job_id,
        "job_name": job_name,
        "job_type": job_type,
        "scheduled_hour": scheduled_hour,
        "day_of_week": day_of_week,
        "start_delay_minutes": start_delay,
        "previous_success_count": previous_successes,
        "previous_failure_count": previous_failures,
        "average_runtime_minutes": average_runtime,
        "expected_data_volume": expected_volume,
        "cpu_usage_percent": cpu,
        "memory_usage_percent": memory,
        "dependency_status": dependency,
    }])


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None


# ============================================================
# HEADER / NAVIGATION
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">◈ SAP BASIS <span>AI COMMAND CENTER</span></div>
        <div class="subtitle">
            Background-job monitoring, failure-risk prediction and operational analytics
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav = st.radio(
    "Navigation",
    ["Overview", "Job Prediction", "Data Analytics", "Jobs", "ML Insights", "About"],
    horizontal=True,
    label_visibility="collapsed",
)

# ============================================================
# DATA SOURCE
# ============================================================

default_df = None
try:
    default_df = load_default_dataset(str(DEFAULT_DATASET_PATH))
except Exception:
    default_df = None

if st.session_state.uploaded_df is not None:
    active_df = st.session_state.uploaded_df
    active_source = "Uploaded dataset"
else:
    active_df = default_df
    active_source = "Project dataset"


# ============================================================
# OVERVIEW
# ============================================================

if nav == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI-powered SAP monitoring</div>
            <h1>Know the risk before the job runs.</h1>
            <p>
                Monitor background-job conditions, analyze historical execution data,
                and use machine learning to estimate whether a new SAP job is likely
                to fail before execution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error("ML model is not available.")
        st.info(
            "Check that models/gradient_boosting_model.pkl and "
            "models/preprocessor.pkl exist in the project."
        )

    if active_df is not None:
        rows, cols, missing, duplicates, failed, success, status_col = dataset_summary(active_df)
        total_status = failed + success
        failure_rate = (failed / total_status * 100) if total_status else 0.0

        st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Jobs analyzed</div>'
                f'<div class="kpi-value">{rows:,}</div>'
                f'<div class="kpi-help">{active_source}</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Failed jobs</div>'
                f'<div class="kpi-value">{failed:,}</div>'
                f'<div class="kpi-help">Historical failed executions</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Successful jobs</div>'
                f'<div class="kpi-value">{success:,}</div>'
                f'<div class="kpi-help">Historical successful executions</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Failure rate</div>'
                f'<div class="kpi-value">{failure_rate:.2f}%</div>'
                f'<div class="kpi-help">Based on available status records</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Platform Health</div>', unsafe_allow_html=True)
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(
                f'<div class="card"><div class="kpi-label">ML engine</div>'
                f'<div style="margin-top:14px"><span class="status-pill {"pill-green" if model_ready else "pill-red"}">'
                f'{"● READY" if model_ready else "● ERROR"}</span></div>'
                f'<div class="kpi-help">Gradient Boosting model</div></div>',
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Preprocessor</div>'
                f'<div style="margin-top:14px"><span class="status-pill {"pill-green" if model_ready else "pill-red"}">'
                f'{"● READY" if model_ready else "● ERROR"}</span></div>'
                f'<div class="kpi-help">Feature transformation pipeline</div></div>',
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Dataset</div>'
                f'<div style="margin-top:14px"><span class="status-pill {"pill-green" if active_df is not None else "pill-yellow"}">'
                f'{"● LOADED" if active_df is not None else "● WAITING"}</span></div>'
                f'<div class="kpi-help">{active_source}</div></div>',
                unsafe_allow_html=True,
            )
        with h4:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Prediction engine</div>'
                f'<div style="margin-top:14px"><span class="status-pill {"pill-green" if model_ready else "pill-red"}">'
                f'{"● ONLINE" if model_ready else "● OFFLINE"}</span></div>'
                f'<div class="kpi-help">Ready for job analysis</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Historical Job Distribution</div>', unsafe_allow_html=True)
        if status_col:
            chart_df = normalize_status_series(active_df[status_col]).value_counts().rename_axis("Status").reset_index(name="Jobs")
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(chart_df.set_index("Status"), height=300)
            with c2:
                if "dependency_status" in active_df.columns:
                    dep = active_df.copy()
                    dep["_status_norm"] = normalize_status_series(dep[status_col])
                    dep["_failed"] = (dep["_status_norm"] == "Failed").astype(int)
                    dep_rate = dep.groupby("dependency_status")["_failed"].mean().mul(100).round(2)
                    st.bar_chart(dep_rate, height=300)
                else:
                    st.info("Dependency status data is not available in the active dataset.")
        else:
            st.info("No job status column was found in the active dataset.")

        st.markdown('<div class="section-title">How the system works</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            st.markdown(
                '<div class="explain"><strong>1. Enter job conditions</strong>'
                '<p>Provide job, execution, resource and dependency information.</p></div>',
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                '<div class="explain"><strong>2. ML risk analysis</strong>'
                '<p>The trained preprocessing pipeline and Gradient Boosting model analyze the job.</p></div>',
                unsafe_allow_html=True,
            )
        with c:
            st.markdown(
                '<div class="explain"><strong>3. Take action</strong>'
                '<p>Review failure probability, risk level and the recommended BASIS checks.</p></div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No project dataset found. Open Data Analytics to upload a CSV dataset.")


# ============================================================
# JOB PREDICTION
# ============================================================

elif nav == "Job Prediction":
    st.markdown(
        '<div class="section-title">🤖 Job Failure Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-note">Enter the conditions of a future SAP background job and let the trained model estimate its failure risk.</div>',
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error("Prediction is unavailable because the model or preprocessor could not be loaded.")
        st.code(model_error)
        st.stop()

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="section-title">Job Information</div>', unsafe_allow_html=True)

        job_id = st.text_input("Job ID", value="JOB001")
        job_name = st.selectbox(
            "Job Name",
            ["DAILY_SALES", "MONTHLY_REPORT", "DATA_TRANSFER", "BACKUP_JOB", "CUSTOM_JOB"],
        )
        job_type = st.selectbox(
            "Job Type",
            ["Reporting", "Data Transfer", "Backup", "Maintenance"],
        )
        scheduled_hour = st.slider("Scheduled Hour", 0, 23, 10)
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )

        st.markdown('<div class="section-title">Historical Behavior</div>', unsafe_allow_html=True)
        previous_successes = st.number_input(
            "Previous Success Count",
            min_value=0,
            value=15,
            step=1,
            help="How many previous executions of this job completed successfully.",
        )
        previous_failures = st.number_input(
            "Previous Failure Count",
            min_value=0,
            value=0,
            step=1,
            help="How many previous executions of this job failed.",
        )
        average_runtime = st.number_input(
            "Average Runtime (minutes)",
            min_value=0,
            value=60,
            step=1,
            help="Typical runtime of this job.",
        )

    with right:
        st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)

        cpu_usage = st.slider(
            "CPU Usage (%)",
            0,
            100,
            50,
            help="Current CPU utilization. Higher utilization means greater processing load.",
        )
        memory_usage = st.slider(
            "Memory Usage (%)",
            0,
            100,
            50,
            help="Current memory utilization of the system.",
        )
        start_delay = st.number_input(
            "Start Delay (minutes)",
            min_value=0,
            value=0,
            step=1,
            help="How many minutes the job started after its scheduled time.",
        )
        expected_volume = st.number_input(
            "Expected Data Volume",
            min_value=0,
            value=300000,
            step=1000,
            help="Expected amount of data handled by the job.",
        )
        dependency_status = st.selectbox(
            "Dependency Status",
            ["Ready", "Delayed", "Not Ready"],
            help="Whether required upstream jobs/resources are available.",
        )

        st.markdown('<div class="section-title">Live Condition View</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        with v1:
            render_metric_card("CPU utilization", cpu_usage, "cpu", "%")
        with v2:
            render_metric_card("Memory utilization", memory_usage, "memory", "%")
        v3, v4 = st.columns(2)
        with v3:
            render_metric_card("Previous failures", previous_failures, "failures")
        with v4:
            render_metric_card("Start delay", start_delay, "delay", " min")

        dep_class = "pill-green" if dependency_status == "Ready" else "pill-yellow" if dependency_status == "Delayed" else "pill-red"
        st.markdown(
            f'<div style="margin-top:14px"><span class="status-pill {dep_class}">● {dependency_status.upper()}</span>'
            f'<span class="small-note" style="margin-left:10px">Dependency health</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns([1, 1.2, 1])
    with p2:
        predict_clicked = st.button("🔮 PREDICT JOB FAILURE", use_container_width=True)

    if predict_clicked:
        new_job = build_prediction_frame(
            job_id,
            job_name,
            job_type,
            scheduled_hour,
            day_of_week,
            start_delay,
            previous_successes,
            previous_failures,
            average_runtime,
            expected_volume,
            cpu_usage,
            memory_usage,
            dependency_status,
        )

        missing = validate_prediction_inputs(new_job)
        if missing:
            st.error("The prediction input structure is missing required model features.")
            st.write(", ".join(missing))
            st.stop()

        try:
            processed = preprocessor.transform(new_job)
            prediction = model.predict(processed)[0]
            failure_probability = get_failure_probability(model, processed)
        except Exception as exc:
            st.error("Prediction could not be completed.")
            st.exception(exc)
            st.stop()

        risk_level, icon, risk_color, risk_class = risk_details(failure_probability)
        failure_percentage = failure_probability * 100

        recommendation = {
            "HIGH": "Check dependency status, CPU/memory utilization, previous failures, start delay and the expected data volume before execution.",
            "MEDIUM": "Monitor the job closely and review recent job performance and current resource conditions before execution.",
            "LOW": "The job currently appears to have comparatively low predicted failure risk. Continue normal BASIS monitoring.",
        }[risk_level]

        result = {
            "job_id": job_id,
            "job_name": job_name,
            "job_type": job_type,
            "prediction": str(prediction),
            "failure_probability": failure_probability,
            "failure_percentage": failure_percentage,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "cpu": cpu_usage,
            "memory": memory_usage,
            "previous_failures": previous_failures,
            "start_delay": start_delay,
            "dependency": dependency_status,
        }
        st.session_state.last_prediction = result
        st.session_state.prediction_history.insert(0, result)
        st.session_state.prediction_history = st.session_state.prediction_history[:20]

    result = st.session_state.last_prediction

    if result:
        risk_level = result["risk_level"]
        if risk_level == "HIGH":
            risk_color = "#fb7185"
            icon = "🔴"
            risk_class = "high"
        elif risk_level == "MEDIUM":
            risk_color = "#fbbf24"
            icon = "🟠"
            risk_class = "medium"
        else:
            risk_color = "#4ade80"
            icon = "🟢"
            risk_class = "low"

        st.markdown('<div class="section-title">AI Prediction Result</div>', unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            prediction_failed = str(result["prediction"]).strip().lower() == "failed"
            status_icon = "🔴" if prediction_failed else "🟢"
            status_text = "FAILED" if prediction_failed else str(result["prediction"]).upper()
            status_color = "#fb7185" if prediction_failed else "#4ade80"

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-muted">PREDICTED JOB STATUS</div>
                    <div class="prediction-status" style="color:{status_color}">
                        {status_icon} {status_text}
                    </div>
                    <div class="prediction-muted">{result["job_name"]} · {result["job_id"]}</div>
                    <div class="prediction-percent">{result["failure_percentage"]:.2f}%</div>
                    <div class="prediction-muted">Estimated probability of job failure</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            pct = result["failure_percentage"]
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-muted">FAILURE RISK GAUGE</div>
                    <div class="gauge" style="--gauge-value:{pct:.2f}; --gauge-color:{risk_color};">
                        <div class="gauge-content">
                            <div class="gauge-number">{pct:.1f}%</div>
                            <div class="gauge-label">failure risk</div>
                        </div>
                    </div>
                    <div class="status-pill {'pill-red' if risk_class == 'high' else 'pill-yellow' if risk_class == 'medium' else 'pill-green'}">
                        {icon} {result["risk_level"]} RISK
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            render_metric_card("CPU", result["cpu"], "cpu", "%")
        with a2:
            render_metric_card("Memory", result["memory"], "memory", "%")
        with a3:
            render_metric_card("Previous failures", result["previous_failures"], "failures")
        with a4:
            render_metric_card("Start delay", result["start_delay"], "delay", " min")

        st.markdown('<div class="section-title">BASIS Recommendation</div>', unsafe_allow_html=True)
        rec_class = "high" if result["risk_level"] == "HIGH" else "low" if result["risk_level"] == "LOW" else ""
        st.markdown(
            f"""
            <div class="recommendation {rec_class}">
                <div class="recommendation-title">
                    {"🚨" if result["risk_level"] == "HIGH" else "⚠️" if result["risk_level"] == "MEDIUM" else "✅"}
                    {result["risk_level"]} RISK — Recommended action
                </div>
                <div class="recommendation-text">{result["recommendation"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Prediction Details</div>', unsafe_allow_html=True)
        detail_df = pd.DataFrame([{
            "Job ID": result["job_id"],
            "Job Name": result["job_name"],
            "Job Type": result["job_type"],
            "Prediction": result["prediction"],
            "Failure Probability": f'{result["failure_percentage"]:.2f}%',
            "Risk": result["risk_level"],
            "Dependency": result["dependency"],
        }])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">What the metrics mean</div>', unsafe_allow_html=True)
        e1, e2, e3, e4, e5 = st.columns(5)
        explanations = [
            ("CPU Usage", "Current processor utilization. Higher utilization can indicate greater system load."),
            ("Memory Usage", "Current memory utilization. High usage can reduce available resources."),
            ("Previous Failures", "Historical failure count for the job. Repeated failures are useful risk context."),
            ("Start Delay", "Minutes between the scheduled start and the actual start."),
            ("Dependency", "Whether required upstream jobs or resources are ready."),
        ]
        for col, (title, text) in zip([e1, e2, e3, e4, e5], explanations):
            with col:
                st.markdown(
                    f'<div class="explain"><strong>{title}</strong><p>{text}</p></div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# DATA ANALYTICS
# ============================================================

elif nav == "Data Analytics":
    st.markdown('<div class="section-title">📂 Data Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Upload a CSV dataset. The dashboard will automatically validate it, summarize it and build visual analytics.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload SAP job dataset",
        type=["csv"],
        help="CSV files are supported.",
    )

    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.session_state.uploaded_df = uploaded_df
            active_df = uploaded_df
            active_source = "Uploaded dataset"
            st.success(f"Dataset loaded successfully: {len(uploaded_df):,} rows × {len(uploaded_df.columns)} columns.")
        except Exception as exc:
            st.error("The CSV could not be read.")
            st.exception(exc)
            st.stop()

    if st.session_state.uploaded_df is not None:
        c_clear, c_space = st.columns([1, 4])
        with c_clear:
            if st.button("Clear Uploaded Dataset", use_container_width=True):
                st.session_state.uploaded_df = None
                st.rerun()

    if active_df is None:
        st.info("No dataset is loaded. Upload your sap_job_data.csv to start automatic analytics.")
    else:
        rows, cols, missing, duplicates, failed, success, status_col = dataset_summary(active_df)
        k1, k2, k3, k4, k5 = st.columns(5)
        cards = [
            ("Rows", f"{rows:,}", "Records"),
            ("Columns", f"{cols:,}", "Fields"),
            ("Missing values", f"{missing:,}", "All cells"),
            ("Duplicates", f"{duplicates:,}", "Duplicate rows"),
            ("Failed jobs", f"{failed:,}", "Historical status"),
        ]
        for col, (label, value, help_text) in zip([k1, k2, k3, k4, k5], cards):
            with col:
                st.markdown(
                    f'<div class="card"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value}</div>'
                    f'<div class="kpi-help">{help_text}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(active_df.head(100), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Automatic Visual Analytics</div>', unsafe_allow_html=True)

        if status_col:
            norm_status = normalize_status_series(active_df[status_col])
            status_counts = norm_status.value_counts().rename_axis("Status").reset_index(name="Jobs")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Job status distribution**")
                st.bar_chart(status_counts.set_index("Status"), height=320)
            with c2:
                if "dependency_status" in active_df.columns:
                    temp = active_df.copy()
                    temp["_status"] = norm_status
                    temp["_failed"] = (temp["_status"] == "Failed").astype(int)
                    dep_rate = temp.groupby("dependency_status")["_failed"].mean().mul(100).round(2)
                    st.markdown("**Failure rate by dependency status (%)**")
                    st.bar_chart(dep_rate, height=320)
                else:
                    st.info("No dependency_status column available.")
        else:
            st.warning("No recognizable job-status column was found. Charts requiring job status are unavailable.")

        numeric_candidates = [
            ("cpu_usage_percent", "CPU usage"),
            ("memory_usage_percent", "Memory usage"),
            ("previous_failure_count", "Previous failures"),
            ("start_delay_minutes", "Start delay"),
            ("average_runtime_minutes", "Average runtime"),
            ("expected_data_volume", "Expected data volume"),
        ]

        for column, title in numeric_candidates:
            if column in active_df.columns and status_col:
                plot_df = pd.DataFrame({
                    "Status": normalize_status_series(active_df[status_col]),
                    title: pd.to_numeric(active_df[column], errors="coerce"),
                }).dropna()
                grouped = plot_df.groupby("Status")[title].mean().round(2)
                st.markdown(f"**Average {title.lower()} by job status**")
                st.bar_chart(grouped, height=260)

        st.markdown('<div class="section-title">Column Information</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column": active_df.columns,
            "Data Type": [str(active_df[c].dtype) for c in active_df.columns],
            "Missing": [int(active_df[c].isna().sum()) for c in active_df.columns],
            "Unique": [int(active_df[c].nunique(dropna=True)) for c in active_df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)


# ============================================================
# JOB EXPLORER
# ============================================================

elif nav == "Jobs":
    st.markdown('<div class="section-title">🔎 Job Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Search and filter jobs from the active dataset.</div>',
        unsafe_allow_html=True,
    )

    if active_df is None:
        st.info("Load a dataset from Data Analytics first.")
    else:
        working = active_df.copy()
        status_col = dataset_status_column(working)

        f1, f2, f3 = st.columns(3)
        with f1:
            search = st.text_input("Search job", placeholder="e.g. DAILY_SALES")
        with f2:
            if status_col:
                status_values = ["All"] + sorted(normalize_status_series(working[status_col]).dropna().unique().tolist())
                selected_status = st.selectbox("Status", status_values)
            else:
                selected_status = "All"
        with f3:
            if "job_type" in working.columns:
                type_values = ["All"] + sorted(working["job_type"].dropna().astype(str).unique().tolist())
                selected_type = st.selectbox("Job Type", type_values)
            else:
                selected_type = "All"

        if search:
            mask = pd.Series(False, index=working.index)
            for col in ["job_id", "job_name"]:
                if col in working.columns:
                    mask = mask | working[col].astype(str).str.contains(search, case=False, na=False)
            working = working[mask]

        if status_col and selected_status != "All":
            working = working[normalize_status_series(working[status_col]) == selected_status]

        if "job_type" in working.columns and selected_type != "All":
            working = working[working["job_type"].astype(str) == selected_type]

        st.markdown(
            f'<div class="small-note">Showing {len(working):,} matching records.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(working.head(500), use_container_width=True, hide_index=True)


# ============================================================
# ML INSIGHTS
# ============================================================

elif nav == "ML Insights":
    st.markdown('<div class="section-title">🧠 ML Model Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Technical information about the trained model used by the prediction engine.</div>',
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error("Model information is unavailable.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        model_name = type(model).__name__
        classes = ", ".join(map(str, getattr(model, "classes_", []))) or "Not exposed"

        for col, label, value, help_text in [
            (m1, "Model", model_name, "Loaded trained estimator"),
            (m2, "Prediction", "Binary classification", "Success vs Failed"),
            (m3, "Classes", classes, "Model output classes"),
            (m4, "Pipeline", "Preprocessor + model", "Existing project artifacts"),
        ]:
            with col:
                st.markdown(
                    f'<div class="card"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value" style="font-size:21px">{value}</div>'
                    f'<div class="kpi-help">{help_text}</div></div>',
                    unsafe_allow_html=True,
                )

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
            feature_names = None
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = None

            if feature_names is not None and len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False).head(15)

                st.markdown('<div class="section-title">Top Model Features</div>', unsafe_allow_html=True)
                st.bar_chart(importance_df.set_index("Feature"), height=430)
            else:
                st.info("Feature names are not available from the saved preprocessor, so feature importance cannot be labeled safely.")

        if st.session_state.prediction_history:
            st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
            history_df = pd.DataFrame([
                {
                    "Job ID": x["job_id"],
                    "Job Name": x["job_name"],
                    "Prediction": x["prediction"],
                    "Failure Probability": f'{x["failure_percentage"]:.2f}%',
                    "Risk": x["risk_level"],
                    "Dependency": x["dependency"],
                }
                for x in st.session_state.prediction_history
            ])
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prediction History",
                data=csv,
                file_name="sap_job_prediction_history.csv",
                mime="text/csv",
            )


# ============================================================
# ABOUT
# ============================================================

else:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Project overview</div>
            <h1>SAP BASIS AI Command Center</h1>
            <p>
                A portfolio machine-learning application for analyzing SAP background-job
                execution data and estimating failure risk before execution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            """
            <div class="card">
                <div class="kpi-label">What this project does</div>
                <p style="color:#a8bbd0;line-height:1.7">
                    A user can inspect historical job data, upload a new CSV for automatic
                    analytics, explore jobs, and enter the conditions of a future job.
                    The trained model then returns a predicted status, failure probability,
                    risk classification and BASIS-oriented recommendation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="card">
                <div class="kpi-label">Technology</div>
                <p style="color:#a8bbd0;line-height:1.8">
                    Python · Pandas · Scikit-learn · Gradient Boosting · Joblib · Streamlit
                    · Data visualization
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Important project note</div>', unsafe_allow_html=True)
    st.info(
        "This portfolio project uses simulated SAP background-job data because it is not "
        "connected to a production SAP environment."
    )

    st.markdown('<div class="section-title">Risk rules used by this application</div>', unsafe_allow_html=True)
    risk_df = pd.DataFrame({
        "Risk Level": ["LOW", "MEDIUM", "HIGH"],
        "Failure Probability": ["< 40%", "40% – 69.99%", "≥ 70%"],
        "Meaning": [
            "Comparatively lower predicted risk",
            "Monitor and review conditions",
            "Investigate conditions before execution",
        ],
    })
    st.dataframe(risk_df, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SAP BASIS AI Command Center · Job Failure Prediction · Machine Learning Portfolio Project
    </div>
    """,
    unsafe_allow_html=True,
)
