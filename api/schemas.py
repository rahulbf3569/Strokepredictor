"""Pydantic schemas for the prediction API."""

from typing import Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    gender: str = Field(..., examples=["Male"])
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: str = Field(..., examples=["Yes"])
    work_type: str = Field(..., examples=["Private"])
    Residence_type: str = Field(..., examples=["Urban"])
    avg_glucose_level: float = Field(..., ge=0)
    bmi: Optional[float] = Field(None, ge=0)
    smoking_status: Optional[str] = Field(None, examples=["never smoked"])


class PredictionResponse(BaseModel):
    stroke_probability: float
    stroke_risk: str
    threshold: float
    model: str


class BatchRequest(BaseModel):
    records: list[PatientInput]


class BatchResponse(BaseModel):
    predictions: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool