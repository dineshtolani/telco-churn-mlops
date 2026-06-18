import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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

results = []

# 1. Logistic Regression with class_weight='balanced'
print("1. Logistic Regression (balanced)...")
lr_balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced", random_state=42)),
])
lr_balanced.fit(X_train, y_train)
y_pred = lr_balanced.predict(X_test)
y_prob = lr_balanced.predict_proba(X_test)[:, 1]
results.append(("LogReg Balanced", accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, y_test, y_pred, y_prob))

# 2. Random Forest with class_weight='balanced'
print("2. Random Forest (balanced)...")
rf_balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(max_depth=5, n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
])
rf_balanced.fit(X_train, y_train)
y_pred = rf_balanced.predict(X_test)
y_prob = rf_balanced.predict_proba(X_test)[:, 1]
results.append(("RF Balanced", accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, y_test, y_pred, y_prob))

# 3. Logistic Regression with custom weights (heavier penalty on missing churn)
print("3. Logistic Regression (custom weights: churn=5x)...")
lr_custom = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=10.0, max_iter=3000, class_weight={0: 1, 1: 5}, random_state=42)),
])
lr_custom.fit(X_train, y_train)
y_pred = lr_custom.predict(X_test)
y_prob = lr_custom.predict_proba(X_test)[:, 1]
results.append(("LogReg 5x", accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, y_test, y_pred, y_prob))

# 4. Random Forest with custom weights
print("4. Random Forest (custom weights: churn=5x)...")
rf_custom = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(max_depth=5, n_estimators=300, class_weight={0: 1, 1: 5}, random_state=42, n_jobs=-1)),
])
rf_custom.fit(X_train, y_train)
y_pred = rf_custom.predict(X_test)
y_prob = rf_custom.predict_proba(X_test)[:, 1]
results.append(("RF 5x", accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, y_test, y_pred, y_prob))

# 5. XGBoost with scale_pos_weight
print("5. XGBoost (scale_pos_weight=churn_ratio)...")
churn_ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=churn_ratio, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss",
    )),
])
xgb_balanced.fit(X_train, y_train)
y_pred = xgb_balanced.predict(X_test)
y_prob = xgb_balanced.predict_proba(X_test)[:, 1]
results.append(("XGBoost Balanced", accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, y_test, y_pred, y_prob))

# Print results
print("\n" + "=" * 100)
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'ROC AUC':<10} TN    FP    FN    TP")
print("=" * 100)

for name, _, _, _, _, _, y_t, y_p, y_pr in results:
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p)
    rec = recall_score(y_t, y_p)
    f1 = f1_score(y_t, y_p)
    auc = roc_auc_score(y_t, y_pr)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    print(f"{name:<25} {acc:<10.2%} {prec:<12.2%} {rec:<10.2%} {f1:<10.2%} {auc:<10.2%} {tn:<5} {fp:<5} {fn:<5} {tp:<5}")

    with mlflow.start_run(run_name=name.lower().replace(" ", "_")):
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("roc_auc", round(auc, 4))
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

# Save best model (the one with best F1 score)
print("\n" + "=" * 100)
best_idx = np.argmax([f1_score(y_t, y_p) for _, _, _, _, _, _, y_t, y_p, _ in results])
best_name = results[best_idx][0]
print(f"Best model: {best_name}")

# Save the corresponding model
best_models = [lr_balanced, rf_balanced, lr_custom, rf_custom, xgb_balanced]
joblib.dump(best_models[best_idx], "serving/model/model.joblib")
print(f"Saved to serving/model/model.joblib")
