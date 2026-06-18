FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16 boto3 psycopg2-binary

EXPOSE 5000

ENTRYPOINT ["mlflow"]
CMD ["server", "--host", "0.0.0.0", "--port", "5000"]
