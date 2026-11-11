from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from api.dataschemas import TumorData, PredictionResponse
from artifacts.models.assistant import CancerDetectorAssistant

model = CancerDetectorAssistant()

app = FastAPI(title="Tumor diagnosis assistant")

@app.post("/predict")
def predict(measurement:TumorData):
    prediction = model.predict(measurement.model_dump())
    return PredictionResponse(**prediction)