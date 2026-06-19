import os
import joblib
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel
from serving.app.metrics import PREDICTION_COUNT, PREDICTION_LATENCY, CHURN_PREDICTIONS

router = APIRouter()

encoders: dict = {}

@router.on_event("startup")
def load_encoders():
    global encoders
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "encoders.joblib")
    encoders = joblib.load(path)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: str

@router.post("/predict", response_model=PredictResponse)
def predict(data: CustomerData, request: Request):
    import time
    start = time.time()

    df = pd.DataFrame([data.model_dump()])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    model = request.app.state.model
    prob = model.predict_proba(df)[0, 1]
    pred = "Yes" if prob >= 0.5 else "No"

    latency = time.time() - start
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(latency)
    CHURN_PREDICTIONS.labels(prediction=pred).inc()

    return PredictResponse(churn_probability=round(float(prob), 4), churn_prediction=pred)
