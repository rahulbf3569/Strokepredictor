"""Evaluate saved model on the test split."""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from stroke_risk.data import load_processed
from stroke_risk.features import split_xy
from stroke_risk.models import evaluate
from stroke_risk.predict import load_model, load_metadata
import joblib

def main():
    train_df, test_df = load_processed()
    X_test, y_test = split_xy(test_df)
    model = load_model()
    meta = load_metadata()
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= meta["threshold"]).astype(int)
    m = evaluate(y_test, preds, proba)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()