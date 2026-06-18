"""
Prometheus metrics endpoint for model monitoring.
"""
from fastapi import APIRouter
from prometheus_client import generate_latest, Counter, Histogram
from starlette.responses import Response

router = APIRouter()

# Define Prometheus metrics
PREDICTION_COUNT = Counter("prediction_count", "Number of predictions")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

@router.get("/metrics")
def metrics():
    """Prometheus endpoint — scraped by Prometheus server."""
    return Response(content=generate_latest(), media_type="text/plain")
