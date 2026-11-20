import os
import pandas as pd

import mlflow
from mlflow.client import MlflowClient
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from sklearn.pipeline import Pipeline

BASEPATH = Path(__file__).resolve().parents[2]
config_loaded = load_dotenv(BASEPATH / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class CancerDetectorAssistant:

    def __init__(self) -> None:
        self._name = "LogisticRegression_IU_ModelEng"
        self._alias = "champion"
        self._model_uri = f"models:/{self._name}@{self._alias}"
        self._client = MlflowClient()
        self._model_info = self._client.get_model_version_by_alias(self._name, self._alias)
        self._general_model_info = self._get_general_model_info()
        self._metrics = self._get_metrics_from_run()
        self.pipeline = self._load_model()

    def _get_general_model_info(self) -> Dict[str, Any]:
        return {
            "name"    : self._model_info.name,
            "version" : self._model_info.version,
            "tags"    : self._model_info.tags,
        }

    def _get_metrics_from_run(self):
        run = self._client.get_run(self._model_info.run_id)
        return run.data.metrics

    def _load_model(self) -> Pipeline:
        return mlflow.sklearn.load_model(self._model_uri)  # type: ignore

    def reload(self) -> None:
        self._model_info = self._client.get_model_version_by_alias(self._name, self._alias)
        self._general_model_info = self._get_general_model_info()
        self._metrics = self._get_metrics_from_run()
        self._model_uri = f"models:/{self._name}@{self._alias}"
        self.pipeline = self._load_model()

    def predict(self, X:Dict) -> Dict[str, Any]:
        df = pd.DataFrame([X])
        df = df.drop(["patient_id", "date_diagnosis"], axis=1)
        # diagnosis prediction
        prediction = self.pipeline.predict(df)[0]        
        diagnosis = "M" if prediction == 1 else "B"
        # probabilty
        classes = list(self.pipeline.classes_)
        malignant_idx = classes.index(1)
        probability_malignant = self.pipeline.predict_proba(df)[0, malignant_idx]
        return {
            "patient_id" : X["patient_id"],
            "timestamp"  : datetime.now(),
            "diagnosis"  : diagnosis,
            "probability_malignant" : probability_malignant
        }

    def train(self) -> None:
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        
        model_general = self._get_general_model_info()
        model_general["metrics"] = self._metrics
        return model_general

    def get_params(self) -> Dict[str, Any]:
        model_general = self._get_general_model_info()
        model_general["params"] = self.pipeline["logistic_regression"].get_params()
        return model_general
