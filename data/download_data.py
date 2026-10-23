"""Download the Stroke Prediction Dataset from Kaggle.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or env vars).
Alternatively, download manually from:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
and place the CSV as data/raw/healthcare-dataset-stroke-data.csv
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stroke_risk.config import RAW_DIR, RAW_DATA_FILE

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILE.exists():
        print(f"Data already present: {RAW_DATA_FILE}")
        return
    try:
        import kaggle  # noqa
    except ImportError:
        print("kaggle package not installed. pip install kaggle")
        print("Or download manually and place at:", RAW_DATA_FILE)
        return
    # use kaggle CLI via subprocess
    import subprocess
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "fedesoriano/stroke-prediction-dataset",
         "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )
    # rename if needed
    for f in RAW_DIR.glob("*.csv"):
        if f.name != RAW_DATA_FILE.name:
            f.rename(RAW_DATA_FILE)
            break
    print("Downloaded to", RAW_DATA_FILE)

if __name__ == "__main__":
    main()