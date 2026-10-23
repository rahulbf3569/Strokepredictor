"""Data loading, validation and splitting."""

import json
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_FILE,
    TARGET,
    ID_COL,
    ALL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    PROCESSED_TRAIN,
    PROCESSED_TEST,
)


def load_raw(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python data/download_data.py` "
            "or place the Kaggle CSV there."
        )
    df = pd.read_csv(path)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Basic data-quality checks; returns a report dict."""
    report = {}
    report["n_rows"] = len(df)
    report["n_cols"] = df.shape[1]
    report["columns"] = list(df.columns)
    report["missing"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["target_counts"] = df[TARGET].value_counts().to_dict() if TARGET in df else {}
    report["target_rate"] = float(df[TARGET].mean()) if TARGET in df else None
    report["duplicates"] = int(df.duplicated().sum())
    # schema check
    expected = set(ALL_FEATURES + [TARGET, ID_COL])
    report["missing_columns"] = sorted(expected - set(df.columns))
    report["unexpected_columns"] = sorted(set(df.columns) - expected)
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize smoking "Unknown" -> NaN for imputation
    if "smoking_status" in df.columns:
        df["smoking_status"] = df["smoking_status"].replace("Unknown", pd.NA)
    # bmi: Kaggle uses "N/A" string sometimes
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    # drop id
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df


def split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = clean(df)
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    PROCESSED_DIR = PROCESSED_TRAIN.parent
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_TRAIN, index=False)
    test.to_csv(PROCESSED_TEST, index=False)
    return train, test


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not PROCESSED_TRAIN.exists() or not PROCESSED_TEST.exists():
        raise FileNotFoundError("Processed splits not found. Run split() first.")
    return pd.read_csv(PROCESSED_TRAIN), pd.read_csv(PROCESSED_TEST)


if __name__ == "__main__":
    df = load_raw()
    rep = validate(df)
    print(json.dumps(rep, indent=2, default=str))
    train, test = split(df)
    print(f"train={len(train)} test={len(test)}")