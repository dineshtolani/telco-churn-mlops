import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

def setup_mlflow(
    tracking_uri: str = None,
    experiment_name: str = "telcochurn",
    s3_endpoint_url: str = None,
):
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)

    s3_url = s3_endpoint_url or os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if s3_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_url

    if os.getenv("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    return mlflow.get_experiment_by_name(experiment_name)
