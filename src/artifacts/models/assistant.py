import joblib
import pandas as pd

from pathlib import Path
from typing import Dict
from datetime import datetime
from sklearn.pipeline import Pipeline

class CancerDetectorAssistant:

    def __init__(self) -> None:
        self.name = "CancerDetectorAssistant"
        self.pipeline : Pipeline = joblib.load(Path(__file__).parent.joinpath("logistic_regression.joblib"))

    def predict(self, X:Dict) -> Dict:
        df = pd.DataFrame([X])
        df = df.drop(["patient_id", "date_diagnosis"], axis=1)
        # diagnosis prediction
        prediction = self.pipeline.predict(df)[0]        
        diagnosis = "M" if prediction == 1 else "B"
        # probabilty
        classes = list(self.pipeline.classes_)
        malignant_idx = classes[-1]
        probability_malignant = self.pipeline.predict_proba(df)[0, malignant_idx]
        return {
            "patient_id" : X["patient_id"],
            "timestamp"  : datetime.now(),
            "diagnosis"  : diagnosis,
            "probability_malignant" : probability_malignant
        }

    def train(self) -> None:
        raise NotImplementedError
