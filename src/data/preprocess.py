import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    return df

def make_preprocessor_logreg():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])

def make_preprocessor_xgb():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

def split_data(df, test_size=0.2, random_state=42):
    y = (df[TARGET].map({"Yes": 1, "No": 0}).values)
    X = df[CAT_FEATURES + NUM_FEATURES]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
