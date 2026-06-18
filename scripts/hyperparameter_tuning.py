import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("telcochurn")

CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

df = pd.read_csv("data/raw/customers_jan.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

y = (df["Churn"].map({"Yes": 1, "No": 0}).values)
X = df[CAT_FEATURES + NUM_FEATURES]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
    ("num", StandardScaler(), NUM_FEATURES),
])

# Logistic Regression tuning
print("Tuning Logistic Regression...")
lr_params = {"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__max_iter": [3000]}
lr_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=42))])
lr_grid = GridSearchCV(lr_pipe, lr_params, cv=3, scoring="roc_auc", n_jobs=-1)
lr_grid.fit(X_train, y_train)

with mlflow.start_run(run_name="logreg_tuned"):
    mlflow.log_params(lr_grid.best_params_)
    mlflow.log_metric("cv_roc_auc", lr_grid.best_score_)
    y_pred = lr_grid.predict(X_test)
    y_prob = lr_grid.predict_proba(X_test)[:, 1]
    mlflow.log_metric("test_accuracy", round(accuracy_score(y_test, y_pred), 4))
    mlflow.log_metric("test_precision", round(precision_score(y_test, y_pred), 4))
    mlflow.log_metric("test_recall", round(recall_score(y_test, y_pred), 4))
    mlflow.log_metric("test_f1", round(f1_score(y_test, y_pred), 4))
    mlflow.log_metric("test_roc_auc", round(roc_auc_score(y_test, y_prob), 4))
    mlflow.sklearn.log_model(lr_grid.best_estimator_, "model")
    print(f"  Best params: {lr_grid.best_params_}")
    print(f"  CV ROC AUC: {lr_grid.best_score_:.4f}")
    print(f"  Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Random Forest tuning
print("Tuning Random Forest...")
rf_params = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [5, 10, 15],
}
rf_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))])
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=3, scoring="roc_auc", n_jobs=-1)
rf_grid.fit(X_train, y_train)

with mlflow.start_run(run_name="rf_tuned"):
    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_metric("cv_roc_auc", rf_grid.best_score_)
    y_pred = rf_grid.predict(X_test)
    y_prob = rf_grid.predict_proba(X_test)[:, 1]
    mlflow.log_metric("test_accuracy", round(accuracy_score(y_test, y_pred), 4))
    mlflow.log_metric("test_precision", round(precision_score(y_test, y_pred), 4))
    mlflow.log_metric("test_recall", round(recall_score(y_test, y_pred), 4))
    mlflow.log_metric("test_f1", round(f1_score(y_test, y_pred), 4))
    mlflow.log_metric("test_roc_auc", round(roc_auc_score(y_test, y_prob), 4))
    mlflow.sklearn.log_model(rf_grid.best_estimator_, "model")
    print(f"  Best params: {rf_grid.best_params_}")
    print(f"  CV ROC AUC: {rf_grid.best_score_:.4f}")
    print(f"  Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\nTraining complete. Check MLflow UI at http://localhost:5000")
