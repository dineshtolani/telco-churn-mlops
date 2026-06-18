import subprocess
import sys

MODELS = [
    ("Baseline", "python -m src.models.train_baseline"),
    ("Logistic Regression", "python -m src.models.train_logreg --C 1.0"),
    ("Random Forest", "python -m src.models.train_rf --n_estimators 300 --max_depth 10"),
    ("XGBoost", "python -m src.models.train_xgboost --n_estimators 300 --max_depth 4 --learning_rate 0.05"),
]

def main():
    results = {}
    for name, cmd in MODELS:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
        results[name] = result.returncode

    print(f"\n{'='*60}")
    print("Summary:")
    for name, code in results.items():
        status = "OK" if code == 0 else "FAILED"
        print(f"  {name}: {status}")

if __name__ == "__main__":
    main()
