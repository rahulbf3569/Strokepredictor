"""Inference helpers."""

import json
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, METADATA_PATH, ALL_FEATURES


def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


def predict_one(record: Dict[str, Any]) -> Dict[str, Any]:
    model = load_model()
    meta = load_metadata()
    threshold = meta["threshold"]
    df = pd.DataFrame([record])
    # ensure all features present
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ALL_FEATURES]
    proba = float(model.predict_proba(df)[:, 1][0])
    label = int(proba >= threshold)
    return {
        "stroke_probability": round(proba, 4),
        "stroke_risk": "high" if label == 1 else "low",
        "threshold": threshold,
        "model": meta["best_model"],
    }


def predict_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model = load_model()
    meta = load_metadata()
    threshold = meta["threshold"]
    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ALL_FEATURES]
    probas = model.predict_proba(df)[:, 1]
    return [
        {
            "stroke_probability": round(float(p), 4),
            "stroke_risk": "high" if p >= threshold else "low",
        }
        for p in probas
    ]