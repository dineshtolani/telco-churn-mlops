import argparse
import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data.preprocess import load_data, split_data
from src.utils.mlflow_setup import setup_mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/customers_jan.csv")
    args = parser.parse_args()

    setup_mlflow()
    df = load_data(args.input)
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run(run_name="baseline") as run:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "DummyClassifier")
        mlflow.log_param("strategy", "most_frequent")
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("roc_auc", round(auc, 4))
        mlflow.sklearn.log_model(model, "model")

        churn_rate = y_train.mean()
        print(f"Baseline (always predict {1 if churn_rate > 0.5 else 0})")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"  F1: {f1:.4f} | ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
