from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def ingest_data(**context):
    import pandas as pd
    month = context["params"].get("month", "apr")
    path = f"/data/raw/customers_{month}.csv"
    df = pd.read_csv(path)
    print(f"Ingested {len(df)} rows from {path}")
    return len(df)

def validate_data(**context):
    import pandas as pd
    month = context["params"].get("month", "apr")
    path = f"/data/raw/customers_{month}.csv"
    df = pd.read_csv(path)
    required_cols = ["customerID", "tenure", "MonthlyCharges", "Churn"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    nulls = df["TotalCharges"].isnull().sum()
    if nulls > 0:
        print(f"Warning: {nulls} null TotalCharges values")
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Churn rate: {churn_rate:.2%}")
    return float(churn_rate)

def preprocess_data(**context):
    import pandas as pd
    month = context["params"].get("month", "apr")
    path = f"/data/raw/customers_{month}.csv"
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["tenure_years"] = (df["tenure"] / 12).round(1)
    df["avg_charge_per_month"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).round(2)
    output_path = f"/data/processed/customers_{month}_processed.parquet"
    os.makedirs("/data/processed", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

with DAG(
    "etl_pipeline",
    default_args=default_args,
    description="Monthly ETL: ingest, validate, preprocess",
    schedule_interval="@monthly",
    catchup=False,
    params={"month": "apr"},
) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    validate = PythonOperator(task_id="validate_data", python_callable=validate_data)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    ingest >> validate >> preprocess
