# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.12-slim AS runtime

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /root/.local /home/app/.local
ENV PATH=/home/app/.local/bin:$PATH

WORKDIR /app
COPY serving/ serving/
COPY --from=builder /root/.local/lib/python3.12/site-packages /home/app/.local/lib/python3.12/site-packages

USER app

EXPOSE 8888

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8888"]
