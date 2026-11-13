"""Shared settings for the Smart Manufacturing module."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "smart_manufacturing" / "raw"
DATASET_PATH = RAW_DATA_DIR / "ai4i2020.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "smart_manufacturing"
REPORT_DIR = PROJECT_ROOT / "reports" / "smart_manufacturing"

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

RANDOM_STATE = 42
TARGET_COLUMN = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
REQUIRED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
    + FAILURE_MODE_COLUMNS
)

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
