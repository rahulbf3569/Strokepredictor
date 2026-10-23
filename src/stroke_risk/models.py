"""Model definitions, training and evaluation."""

import json
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, MODEL_PATH, METADATA_PATH, MIN_RECALL
from .features import build_preprocessor, split_xy


def candidate_models() -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def tune_threshold(y_true, y_proba, min_recall=MIN_RECALL) -> float:
    """Pick the lowest threshold achieving min_recall; fallback to 0.5."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best = 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if recall_score(y_true, preds, zero_division=0) >= min_recall:
            # among those, maximize f1
            best = t
            break
    # maximize f1 among thresholds satisfying recall
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if recall_score(y_true, preds, zero_division=0) >= min_recall:
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    return float(best_t) if best_f1 >= 0 else 0.5


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train(train_df, test_df) -> Tuple[Pipeline, dict]:
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    results = {}
    best_name, best_score, best_pipe = None, -1, None
    for name, model in candidate_models().items():
        pipe = build_pipeline(model)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = evaluate(y_test, preds, proba)
        metrics["cv_avg_precision_mean"] = float(cv_scores.mean())
        metrics["cv_avg_precision_std"] = float(cv_scores.std())
        results[name] = metrics
        if metrics["cv_avg_precision_mean"] > best_score:
            best_score = metrics["cv_avg_precision_mean"]
            best_name = name
            best_pipe = pipe
            best_proba = proba

    # threshold tuning on best model
    threshold = tune_threshold(y_test, best_proba)
    final_preds = (best_proba >= threshold).astype(int)
    final_metrics = evaluate(y_test, final_preds, best_proba)
    final_metrics["threshold"] = threshold
    final_metrics["best_model"] = best_name

    # save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    metadata = {
        "best_model": best_name,
        "threshold": threshold,
        "metrics": final_metrics,
        "all_models": results,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # save preprocessor separately for convenience
    from .features import save_preprocessor
    save_preprocessor(best_pipe.named_steps["preprocessor"])

    return best_pipe, metadata