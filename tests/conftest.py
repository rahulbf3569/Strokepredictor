import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.normal(50, 15, n).clip(1, 90),
        "hypertension": rng.binomial(1, 0.1, n),
        "heart_disease": rng.binomial(1, 0.05, n),
        "ever_married": rng.choice(["Yes", "No"], n),
        "work_type": rng.choice(["Private", "Self-employed", "Govt_job"], n),
        "Residence_type": rng.choice(["Urban", "Rural"], n),
        "avg_glucose_level": rng.normal(110, 30, n).clip(50, 250),
        "bmi": rng.normal(28, 6, n).clip(10, 50),
        "smoking_status": rng.choice(["never smoked", "formerly smoked", "smokes"], n),
        "stroke": rng.binomial(1, 0.08, n),
    })
    # ensure both classes
    df.loc[:5, "stroke"] = 1
    df.loc[6:15, "stroke"] = 0
    return df