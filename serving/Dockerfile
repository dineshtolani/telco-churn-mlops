# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

WORKDIR /build
COPY serving/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.12-slim AS runtime

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

WORKDIR /app
COPY serving/ serving/

USER app

EXPOSE 8888

CMD ["python", "-m", "uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8888"]
