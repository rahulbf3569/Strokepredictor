"""Generate a small SYNTHETIC sample mimicking the stroke dataset schema.

Useful for CI / smoke tests without Kaggle credentials. NOT real data.
Designed so a classifier can learn a realistic signal (age, hypertension,
heart disease, glucose, smoking) with ~6% positive rate.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"

def main(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.normal(52, 17, n).clip(1, 90)
    hypertension = rng.binomial(1, 0.10, n)
    heart_disease = rng.binomial(1, 0.06, n)
    avg_glucose = rng.normal(108, 38, n).clip(55, 280)
    bmi = rng.normal(28.5, 6.5, n).clip(14, 55)
    smoking = rng.choice(["never smoked", "formerly smoked", "smokes", "Unknown"],
                        n, p=[0.48, 0.22, 0.15, 0.15])
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "gender": rng.choice(["Male", "Female"], n, p=[0.42, 0.58]),
        "age": age.round(1),
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": rng.choice(["Yes", "No"], n, p=[0.68, 0.32]),
        "work_type": rng.choice(["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                                n, p=[0.58, 0.15, 0.13, 0.10, 0.04]),
        "Residence_type": rng.choice(["Urban", "Rural"], n),
        "avg_glucose_level": avg_glucose.round(2),
        "bmi": bmi.round(1),
        "smoking_status": smoking,
    })
    # inject missing bmi (~4%) and mark smoking Unknown -> NA in clean()
    mask = rng.random(n) < 0.04
    df.loc[mask, "bmi"] = np.nan

    # logistic risk model
    logit = (
        -5.0
        + 0.06 * (age - 40)
        + 0.9 * hypertension
        + 1.1 * heart_disease
        + 0.008 * (avg_glucose - 100)
        + 0.04 * (bmi - 25)
        + np.where(smoking == "smokes", 0.5, 0.0)
        + np.where(smoking == "formerly smoked", 0.25, 0.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["stroke"] = (rng.random(n) < prob).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote synthetic sample: {OUT} (stroke rate={df['stroke'].mean():.4f}, n={len(df)})")

if __name__ == "__main__":
    main()