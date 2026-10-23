import json
from pathlib import Path

import pandas as pd

from stroke_risk.monitor import save_reference_stats, detect_drift


def test_drift_detection_no_drift(sample_df, tmp_path, monkeypatch):
    import stroke_risk.monitor as m
    ref_path = tmp_path / "ref.json"
    monkeypatch.setattr(m, "REFERENCE_STATS_PATH", ref_path)
    save_reference_stats(sample_df, path=ref_path)
    # current = same distribution
    res = detect_drift(sample_df.copy(), reference_path=ref_path)
    assert "psi" in res
    assert isinstance(res["drift_detected"], bool)


def test_drift_detection_with_drift(sample_df, tmp_path, monkeypatch):
    import stroke_risk.monitor as m
    ref_path = tmp_path / "ref.json"
    monkeypatch.setattr(m, "REFERENCE_STATS_PATH", ref_path)
    save_reference_stats(sample_df, path=ref_path)
    cur = sample_df.copy()
    cur["age"] = cur["age"] + 30  # induce drift
    res = detect_drift(cur, reference_path=ref_path)
    assert res["psi"]["age"] > 0