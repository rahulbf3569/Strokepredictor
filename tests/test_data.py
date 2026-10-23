import pandas as pd

from stroke_risk.data import validate, clean, split
from stroke_risk.config import TARGET


def test_validate(sample_df):
    rep = validate(sample_df)
    assert rep["n_rows"] == len(sample_df)
    assert rep["missing_columns"] == []
    assert len(rep["target_counts"]) == 2


def test_clean_removes_id(sample_df):
    out = clean(sample_df)
    assert "id" not in out.columns
    assert "stroke" in out.columns


def test_split_stratified(sample_df, tmp_path, monkeypatch):
    import stroke_risk.data as d
    monkeypatch.setattr(d, "PROCESSED_TRAIN", tmp_path / "train.csv")
    monkeypatch.setattr(d, "PROCESSED_TEST", tmp_path / "test.csv")
    tr, te = split(sample_df)
    assert len(tr) + len(te) == len(sample_df)
    # stratification preserves rate approx
    assert abs(tr[TARGET].mean() - sample_df[TARGET].mean()) < 0.05