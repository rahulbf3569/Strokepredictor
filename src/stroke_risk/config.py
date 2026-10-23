"""Project configuration: paths, schema, hyperparameters."""

from pathlib import Path
import os

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_FILE = RAW_DIR / "healthcare-dataset-stroke-data.csv"
PROCESSED_TRAIN = PROCESSED_DIR / "train.csv"
PROCESSED_TEST = PROCESSED_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "stroke_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"

# ---- Schema ----
TARGET = "stroke"
ID_COL = "id"

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- Training ----
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# In medical screening we prefer catching positives -> tune threshold for recall
MIN_RECALL = 0.70

# ---- Monitoring ----
PSI_THRESHOLD = 0.2  # >0.2 = significant drift
DRIFT_FEATURE_THRESHOLD = 0.1

# ---- API ----
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))