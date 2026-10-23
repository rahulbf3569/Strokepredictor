"""FastAPI service for stroke risk prediction."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PatientInput, PredictionResponse, BatchRequest, BatchResponse, HealthResponse
from stroke_risk.predict import predict_one, predict_batch, load_model, load_metadata

app = FastAPI(
    title="Stroke Risk Prediction API",
    description="MLOps demo: predict stroke risk from patient features.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except Exception as e:
        return HealthResponse(status=f"error: {e}", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        return predict_one(payload.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch_endpoint(payload: BatchRequest):
    try:
        preds = predict_batch([r.model_dump() for r in payload.records])
        return BatchResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
def metadata():
    try:
        return load_metadata()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))