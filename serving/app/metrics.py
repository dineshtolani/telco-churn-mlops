from fastapi import APIRouter
from prometheus_client import generate_latest, Counter, Histogram, Gauge
from starlette.responses import Response
import time

router = APIRouter()

PREDICTION_COUNT = Counter("prediction_count", "Total number of predictions")
CHURN_PREDICTIONS = Counter("churn_predictions", "Number of customers predicted to churn", ["prediction"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
MODEL_ACCURACY = Gauge("model_accuracy", "Latest model accuracy")
MODEL_ROC_AUC = Gauge("model_roc_auc", "Latest model ROC AUC")

@router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
