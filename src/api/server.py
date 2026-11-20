from fastapi import FastAPI
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any

from api.dataschemas import TumorData, PredictionResponse, VitalSigns
from artifacts.models.assistant import CancerDetectorAssistant

model = CancerDetectorAssistant()

app = FastAPI(title="Tumor diagnosis assistant")

@app.post("/predict")
def predict(measurement:TumorData) -> PredictionResponse:
    prediction = model.predict(measurement.model_dump())
    return PredictionResponse(**prediction)

@app.get("/model/params")
def model_info() -> Dict[str, Any]:
    return model.get_params()

@app.get("/model/metrics")
def model_metrics() -> Dict[str, Any]:
    return model.get_metrics()

@app.get("/health")
def health_check() -> VitalSigns:
    healthy = model.pipeline is not None
    if not healthy:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return VitalSigns(
        timestamp=datetime.now(),
        is_healthy=True
    )