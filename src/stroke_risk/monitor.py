"""Monitoring: data drift via Population Stability Index (PSI)."""

import json
import math
from typing import Dict

import numpy as np
import pandas as pd

from .config import REFERENCE_STATS_PATH, PSI_THRESHOLD, ALL_FEATURES


def _psi(expected, actual, buckets: int = 10) -> float:
    """Compute PSI between two 1D arrays."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # handle categoricals by mapping to codes
    if expected.dtype.kind not in "fcui" or actual.dtype.kind not in "fcui":
        codes, uniques = pd.factorize(np.concatenate([expected.astype(str), actual.astype(str)]))
        expected = codes[: len(expected)].astype(float)
        actual = codes[len(expected):].astype(float)
    else:
        expected = expected.astype(float)
        actual = actual.astype(float)
    # quantile bins from expected
    quantiles = np.linspace(0, 1, buckets + 1)
    bins = np.unique(np.quantile(expected, quantiles))
    if len(bins) < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    # avoid zeros
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def build_reference_stats(train_df: pd.DataFrame) -> dict:
    stats = {}
    for col in ALL_FEATURES:
        stats[col] = {
            "values": train_df[col].astype(str).tolist() if train_df[col].dtype == object else train_df[col].tolist(),
        }
    stats["_target_rate"] = float(train_df["stroke"].mean()) if "stroke" in train_df else None
    return stats


def save_reference_stats(train_df, path=REFERENCE_STATS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    # store compact summary instead of full values for memory
    summary = {}
    for col in ALL_FEATURES:
        s = train_df[col]
        if pd.api.types.is_numeric_dtype(s):
            summary[col] = {"type": "numeric", "values": s.dropna().tolist()}
        else:
            summary[col] = {"type": "categorical", "values": s.astype(str).tolist()}
    if "stroke" in train_df:
        summary["_target_rate"] = float(train_df["stroke"].mean())
    with open(path, "w") as f:
        json.dump(summary, f)


def load_reference_stats(path=REFERENCE_STATS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def detect_drift(current_df: pd.DataFrame, reference_path=REFERENCE_STATS_PATH) -> Dict[str, float]:
    ref = load_reference_stats(reference_path)
    psi_scores = {}
    for col in ALL_FEATURES:
        if col not in current_df.columns or col not in ref:
            continue
        ref_vals = np.array(ref[col]["values"])
        cur_vals = current_df[col]
        if ref[col]["type"] == "categorical":
            ref_vals = ref_vals.astype(str)
            cur_vals = cur_vals.astype(str).to_numpy()
        else:
            cur_vals = pd.to_numeric(cur_vals, errors="coerce").to_numpy()
        psi_scores[col] = _psi(ref_vals, cur_vals)
    # overall drift flag
    drifted = [c for c, v in psi_scores.items() if v > PSI_THRESHOLD]
    return {
        "psi": psi_scores,
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "drift_detected": len(drifted) > 0,
    }