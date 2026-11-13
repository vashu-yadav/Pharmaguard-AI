"""Shared paths and image settings for Tablet Quality Control."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "quality_control" / "raw"
DATASET_DIR = RAW_DATA_DIR / "PharmaVision"
DATASET_ARCHIVE = RAW_DATA_DIR / "PharmaVision.rar"
MODEL_DIR = PROJECT_ROOT / "models" / "quality_control"
REPORT_DIR = PROJECT_ROOT / "reports" / "quality_control"
MODEL_METADATA = MODEL_DIR / "model_metadata.json"

ZENODO_RECORD_ID = "22773411"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
IMAGE_SIZE = 224
CLASS_NAMES = ("Normal", "Defective")
CLASS_TO_INDEX = {"normal": 0, "defective": 1}
RANDOM_STATE = 42
BATCH_SIZE = 16
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
MODEL_NAMES = ("custom_cnn", "mobilenet_v2")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
