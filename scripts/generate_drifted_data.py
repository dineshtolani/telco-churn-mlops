import pandas as pd
import numpy as np
import os
import argparse

np.random.seed(42)

DRIFT_CONFIGS = {
    "feb": {"churn_shift": 0.02, "tenure_shift": -2, "fiber_shift": 0.03, "monthly_shift": 2},
    "mar": {"churn_shift": 0.03, "tenure_shift": -3, "fiber_shift": 0.05, "monthly_shift": 3},
    "apr": {"churn_shift": 0.05, "tenure_shift": -4, "fiber_shift": 0.07, "monthly_shift": 5},
    "may": {"churn_shift": 0.04, "tenure_shift": -3, "fiber_shift": 0.06, "monthly_shift": 4},
    "jun": {"churn_shift": 0.06, "tenure_shift": -5, "fiber_shift": 0.08, "monthly_shift": 6},
    "jul": {"churn_shift": 0.08, "tenure_shift": -3, "fiber_shift": 0.04, "monthly_shift": 8},
}

def apply_drift(df, config):
    df = df.copy()
    n = len(df)

    churn_col = df["Churn"].map({"Yes": 1, "No": 0})
    flip_mask = np.random.random(n) < config["churn_shift"]
    churn_col[flip_mask] = 1 - churn_col[flip_mask]
    df["Churn"] = churn_col.map({1: "Yes", 0: "No"})

    df["tenure"] = (df["tenure"] + np.random.randint(config["tenure_shift"], 0, size=n)).clip(0, 72)
    df["MonthlyCharges"] = (df["MonthlyCharges"] + np.random.uniform(0, config["monthly_shift"], size=n)).round(2)

    fiber_mask = df["InternetService"] == "DSL"
    switch_mask = np.random.random(fiber_mask.sum()) < abs(config["fiber_shift"])
    df.loc[fiber_mask[fiber_mask].index[switch_mask], "InternetService"] = "Fiber optic"

    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/telco_churn.csv")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output, exist_ok=True)

    df.to_csv(f"{args.output}/customers_jan.csv", index=False)
    print("Generated customers_jan.csv (base, no drift)")

    for month, config in DRIFT_CONFIGS.items():
        drifted = apply_drift(df, config)
        out_path = f"{args.output}/customers_{month}.csv"
        drifted.to_csv(out_path, index=False)
        churn_rate = (drifted["Churn"] == "Yes").mean()
        print(f"Generated customers_{month}.csv  (churn rate: {churn_rate:.1%})")

if __name__ == "__main__":
    main()
