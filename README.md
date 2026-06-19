# TelcoChurn — Customer Churn Prediction Platform

Production-grade MLOps platform for predicting telecom customer churn, featuring automated retraining, drift detection, and CI/CD.

## What It Predicts

The model predicts **customer churn** — whether a telecom subscriber will cancel their service in the next billing cycle. Using 21 customer attributes (demographics, account info, services subscribed), it outputs a churn probability and a binary Yes/No decision.

### Business Impact for Telcos

| Problem | Solution via this platform |
|---------|--------------------------|
| **Churn costs millions** — acquiring a new customer is 5-7× more expensive than retaining one | Model identifies high-risk customers *before* they leave, enabling targeted retention offers |
| **Retention budgets are limited** — can't discount everyone | Predictions let marketing focus 80% of budget on the ~20% of customers most likely to churn |
| **Churn reasons drift over time** — pricing, competitor offers, service quality change | Monthly drift detection + auto-retraining pipeline keeps predictions accurate |
| **Siloed data & manual workflows** | End-to-end MLOps: automated ETL → training → serving → monitoring |

### Example Retention Use Cases

- **Contract expiry**: Flag customers on month-to-month plans 30 days before their tenure milestone — offer loyalty discounts
- **Service downgrade detection**: If a customer drops premium features (Online Security, Tech Support), the model's churn probability rises — trigger a proactive check-in call
- **High-value churners**: Customers with above-average tenure and high monthly charges who are predicted to churn → escalate to retention team (not just an automated SMS)

## Architecture

```
GitHub → GitHub Actions (Bandit → Trivy → Docker Build → Push)
                              │
                              v
                         DockerHub
                              │
                              v
                          ArgoCD
                              │
                              v
                        Kubernetes
                              │
       ┌──────────────────────┼──────────────────────┐
       v                      v                      v
    Airflow                MLflow                  MinIO
   (ETL + Drift)      (Model Registry)        (DVC Storage)
       │                      │                      │
       v                      v                      v
     Feast               FastAPI               Prometheus
(Feature Store)        (Model Serving)          (Metrics)
       │                      │                      │
       v                      v                      v
     Redis               Customers              Grafana
  (Online Store)       (Predictions)          (Dashboards)
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Data versioning | DVC + MinIO |
| Feature store | Feast + Redis |
| Orchestration | Apache Airflow |
| Experiment tracking | MLflow (Postgres + MinIO) |
| Model serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Drift detection | Statistical + Evidently AI |
| Security | Bandit, Trivy, GitLeaks, Docker Scout |
| CI/CD | GitHub Actions |
| Deployment | Docker Compose / Kubernetes + ArgoCD |
| Tracing | OpenTelemetry |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/dineshtolani/telco-churn-mlops.git
cd telco-churn-mlops

# Environment
cp .env.example .env
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Data
python scripts/download_data.py
python scripts/generate_drifted_data.py

# Infrastructure
docker compose -f infrastructure/compose/docker-compose.yml up -d

# Train
python scripts/run_training.py

# Serve
uvicorn serving.app.main:app --host 0.0.0.0 --port 8888
```

## UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow | http://localhost:5000 | — |
| MinIO | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| Airflow | http://localhost:8080 | `admin` / `admin` |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| FastAPI | http://localhost:8888/docs | — |

## Model Performance

| Model | Accuracy | Recall | F1 | ROC AUC |
|-------|----------|--------|-----|---------|
| Baseline | 73.4% | 0.0% | 0.0 | 0.500 |
| Logistic Regression | 80.5% | 57.2% | 0.61 | 0.836 |
| Random Forest | 79.4% | 50.8% | 0.57 | 0.834 |
| **RF Balanced** | **71.9%** | **82.1%** | **0.61** | **0.836** |
| XGBoost Balanced | 73.5% | 77.8% | 0.61 | 0.832 |

**Best model:** Random Forest with `class_weight="balanced"` — catches 82% of churners.

## DAGs (Airflow)

| DAG | Schedule | Description |
|-----|----------|-------------|
| `etl_pipeline` | Monthly | Ingest → validate → preprocess → save Parquet |
| `training_pipeline` | Monthly | Train models → register best in MLflow |
| `drift_detection` | Monthly | Compare distributions → trigger retrain if drifted |

## API

```bash
curl -X POST http://localhost:8888/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "tenure": 12, "Contract": "Month-to-month",
    "MonthlyCharges": 79.99, "TotalCharges": 959.88, ...
  }'

# Response: {"churn_probability": 0.6919, "churn_prediction": "Yes"}
```

## CI/CD Pipeline

On every push to `main`:
1. **Bandit** — Python security scan
2. **Unit tests** — pytest
3. **GitLeaks** — secret detection
4. **Trivy FS** — filesystem vulnerabilities
5. **Docker build** — multi-stage (builder → runtime, <300MB)
6. **Trivy image** — container vulnerabilities
7. **Docker Scout** — image analysis
8. **Push to GitHub Container Registry**

## Dataset

IBM Telco Customer Churn (7,043 customers, 21 features). Monthly drifted variants are generated synthetically.

## License

MIT
