import argparse
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from src.data.preprocess import load_data, split_data, make_preprocessor_xgb
from src.utils.mlflow_setup import setup_mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/customers_jan.csv")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    setup_mlflow()
    df = load_data(args.input)
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run(run_name="random_forest") as run:
        pipe = Pipeline([
            ("preprocessor", make_preprocessor_xgb()),
            ("classifier", RandomForestClassifier(
                n_estimators=args.n_estimators, max_depth=args.max_depth,
                random_state=42, n_jobs=-1
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("roc_auc", round(auc, 4))
        mlflow.sklearn.log_model(pipe, "model")

        print(f"Random Forest (n_estimators={args.n_estimators}, max_depth={args.max_depth})")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"  F1: {f1:.4f} | ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
