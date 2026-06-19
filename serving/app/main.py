import os
from dotenv import load_dotenv
load_dotenv()

import joblib
from fastapi import FastAPI
from serving.app.predict import router as predict_router
from serving.app.metrics import router as metrics_router
from serving.app.telemetry import setup_telemetry

app = FastAPI(title="TelcoChurn API", version="1.0.0")
app.include_router(predict_router)
app.include_router(metrics_router)

setup_telemetry(app)

@app.on_event("startup")
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "model.joblib")
    app.state.model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

@app.get("/health")
def health():
    return {"status": "ok"}
