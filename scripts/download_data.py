import os
import pandas as pd
import argparse

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv"

def download_data(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(DATA_URL)
    df.to_csv(output_path, index=False)
    print(f"Downloaded {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/telco_churn.csv")
    args = parser.parse_args()
    download_data(args.output)
