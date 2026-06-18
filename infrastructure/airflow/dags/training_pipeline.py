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

CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

def train_models(**context):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score

    month = context["params"].get("month", "apr")
    path = f"/data/processed/customers_{month}_processed.parquet"
    df = pd.read_parquet(path)

    y = (df["Churn"].map({"Yes": 1, "No": 0}).values)
    X = df[CAT_FEATURES + NUM_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            max_depth=5, n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    os.makedirs("/training/models", exist_ok=True)
    model_path = f"/training/models/churn_model_{month}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path} | F1: {f1:.4f}")
    return float(f1)

with DAG(
    "training_pipeline",
    default_args=default_args,
    description="Train and register best model monthly",
    schedule_interval="@monthly",
    catchup=False,
    params={"month": "apr"},
) as dag:
    train = PythonOperator(task_id="train_models", python_callable=train_models)
    train
