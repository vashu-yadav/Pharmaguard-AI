"""Shared paths and cached readers for saved PharmaGuard artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SMART_MODELS = ROOT / "models" / "smart_manufacturing"
SMART_REPORTS = ROOT / "reports" / "smart_manufacturing"
SMART_DATA = ROOT / "data" / "smart_manufacturing" / "raw" / "ai4i2020.csv"
QC_REPORTS = ROOT / "reports" / "quality_control"


def read_json(path: Path) -> dict:
    """Read a JSON artifact and report an actionable error if it is absent/bad."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required saved project file was not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read project file {path}: {error}") from error


@st.cache_data(show_spinner=False)
def load_metrics(path_string: str) -> dict:
    return read_json(Path(path_string))


@st.cache_resource(show_spinner="Loading the saved machine model...")
def load_smart_pipeline(model_name: str):
    """Load a fitted pipeline selected in model_metadata.json."""
    metadata = read_json(SMART_MODELS / "model_metadata.json")
    model_files = metadata.get("model_files", {})
    if model_name not in model_files:
        raise ValueError(f"No saved pipeline is registered for {model_name!r}.")
    model_path = SMART_MODELS / model_files[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved machine pipeline is missing: {model_path}")
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_shap_background() -> pd.DataFrame:
    """Load a deterministic sample from the saved training partition only."""
    from src.smart_manufacturing.config import FEATURE_COLUMNS, TARGET_COLUMN

    split = read_json(SMART_REPORTS / "split_indices.json")
    train_indices = split.get("train")
    if not train_indices:
        raise ValueError("Saved split file does not contain training row indices for SHAP.")
    if not SMART_DATA.is_file():
        raise FileNotFoundError(f"AI4I source data needed for SHAP is missing: {SMART_DATA}")
    frame = pd.read_csv(SMART_DATA)
    positions = [int(index) for index in train_indices[:100]]
    background = frame.iloc[positions][FEATURE_COLUMNS].copy()
    if TARGET_COLUMN in background.columns or background.empty:
        raise ValueError("Could not create the training-only SHAP background.")
    return background


def metric_value(metrics: dict, model_name: str, split: str, metric: str):
    """Return an actual saved metric, or None when that metric is unavailable."""
    return metrics.get(model_name, {}).get(split, {}).get(metric)


def format_metric(value) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"
