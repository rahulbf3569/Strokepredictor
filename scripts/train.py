"""Train the stroke risk model end-to-end."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_risk.data import load_raw, validate, split
from stroke_risk.models import train
from stroke_risk.monitor import save_reference_stats
from stroke_risk.utils import get_logger

log = get_logger("train")

def main():
    log.info("Loading raw data...")
    df = load_raw()
    rep = validate(df)
    log.info(f"rows={rep['n_rows']} target_rate={rep['target_rate']:.4f} missing_cols={rep['missing_columns']}")
    log.info("Splitting...")
    train_df, test_df = split(df)
    log.info("Training models...")
    pipe, metadata = train(train_df, test_df)
    save_reference_stats(train_df)
    log.info(f"Best model: {metadata['best_model']}")
    log.info(f"Threshold: {metadata['threshold']}")
    log.info(f"Metrics: {metadata['metrics']}")
    print("\n=== Training complete ===")
    print(f"Model saved to models/stroke_model.joblib")

if __name__ == "__main__":
    main()