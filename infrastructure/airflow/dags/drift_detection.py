from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_FEATURES = ["Contract", "InternetService", "PaymentMethod"]

def detect_drift(**context):
    import pandas as pd
    import numpy as np
    month = context["params"].get("month", "apr")
    ref = pd.read_csv("/data/raw/customers_jan.csv")
    current = pd.read_csv(f"/data/raw/customers_{month}.csv")

    # Clean numeric columns
    for col in NUM_FEATURES:
        ref[col] = pd.to_numeric(ref[col], errors="coerce").fillna(0)
        current[col] = pd.to_numeric(current[col], errors="coerce").fillna(0)

    drift_score = 0.0
    for col in NUM_FEATURES:
        ref_mean = ref[col].mean()
        cur_mean = current[col].mean()
        if ref_mean == 0:
            continue
        diff_pct = abs(cur_mean - ref_mean) / ref_mean * 100
        if diff_pct > 5:
            drift_score += 1
        print(f"  {col}: ref={ref_mean:.2f} cur={cur_mean:.2f} drift={diff_pct:.1f}%")

    for col in CAT_FEATURES:
        ref_dist = ref[col].value_counts(normalize=True).to_dict()
        cur_dist = current[col].value_counts(normalize=True).to_dict()
        all_keys = set(list(ref_dist) + list(cur_dist))
        drift = sum(abs(ref_dist.get(k, 0) - cur_dist.get(k, 0)) for k in all_keys)
        if drift > 0.1:
            drift_score += 1
        print(f"  {col}: categorical drift={drift:.3f}")

    if drift_score > 3:
        print(f"\nDRIFT DETECTED! Score: {drift_score}/10")
    else:
        print(f"\nNo significant drift. Score: {drift_score}/10")
    return drift_score

with DAG(
    "drift_detection",
    default_args=default_args,
    description="Monthly data drift check",
    schedule_interval="@monthly",
    catchup=False,
    params={"month": "jun"},
) as dag:
    drift = PythonOperator(task_id="detect_drift", python_callable=detect_drift)
    drift
