import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Breast Cancer Classification"

def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)