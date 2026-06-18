import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"

print("Loading model...")
model = joblib.load("serving/model/model.joblib")

print("Loading data...")
df = pd.read_csv("data/raw/customers_jan.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

y = (df[TARGET].map({"Yes": 1, "No": 0}).values)
X = df[CAT_FEATURES + NUM_FEATURES]

# Split same way as training (80/20, stratified, random_state=42)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTest set: {len(X_test)} customers\n")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 55)
print(f"  Accuracy:  {acc:.2%}  ({int(acc * len(y_test))}/{len(y_test)} correct)")
print(f"  Precision: {prec:.2%}  (of predicted churners, how many actually churned)")
print(f"  Recall:    {rec:.2%}  (of actual churners, how many did we catch)")
print(f"  F1 Score:  {f1:.2%}")
print(f"  ROC AUC:   {auc:.2%}")
print("=" * 55)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"              Predicted Stay  Predicted Churn")
print(f"Actual Stay   {cm[0,0]:>5}            {cm[0,1]:>5}")
print(f"Actual Churn  {cm[1,0]:>5}            {cm[1,1]:>5}")

# Show 10 random examples
print("\n--- 10 random predictions ---")
indices = np.random.choice(len(X_test), 10, replace=False)
for i, idx in enumerate(indices):
    actual = "Churn" if y_test[idx] else "Stay"
    predicted = "Churn" if y_pred[idx] else "Stay"
    match = "✓" if y_test[idx] == y_pred[idx] else "✗"
    print(f"  {match} Customer: {predicted:>6} (prob: {y_prob[idx]:.1%}) | Actual: {actual}")
