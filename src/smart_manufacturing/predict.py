"""Load a saved model pipeline and predict machine failure for new inputs."""

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, MODEL_DIR


def predict_machine(input_data, model_name: str | None = None) -> dict:
    """Predict failure for one row using a saved model and its preprocessing.

    ``input_data`` may be a mapping or a one-row pandas DataFrame. The returned
    dictionary contains the predicted class, failure probability, and threshold.
    A real trained pipeline must exist; this function never invents a result.
    """
    model_directory = Path(MODEL_DIR)
    metadata_path = model_directory / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained Smart Manufacturing model metadata found at {metadata_path}. "
            "Run `python -m src.smart_manufacturing.train` first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_model = model_name or metadata.get("recommended_model")
    model_files = metadata.get("model_files", {})
    if selected_model not in model_files:
        available = ", ".join(sorted(model_files)) or "none"
        raise ValueError(
            f"Unknown model {selected_model!r}. Available trained models: {available}."
        )

    if isinstance(input_data, pd.DataFrame):
        rows = input_data.copy()
    elif isinstance(input_data, dict):
        rows = pd.DataFrame([input_data])
    else:
        raise TypeError("input_data must be a mapping or a one-row pandas DataFrame.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in rows.columns]
    if missing_features:
        raise ValueError("Missing required input features: " + ", ".join(missing_features))
    if len(rows) != 1:
        raise ValueError("predict_machine currently accepts exactly one input row.")

    features = rows[FEATURE_COLUMNS].copy()
    if features["Type"].isna().any() or not set(features["Type"]).issubset({"L", "M", "H"}):
        raise ValueError("Type must be one of the product variants 'L', 'M', or 'H'.")

    relative_model_path = Path(model_files[selected_model])
    model_path = relative_model_path if relative_model_path.is_absolute() else model_directory / relative_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model pipeline not found: {model_path}")

    pipeline = joblib.load(model_path)
    classifier = pipeline.named_steps.get("classifier")
    if classifier is not None and classifier.__class__.__module__.startswith("xgboost"):
        # The preprocessing pipeline produces CPU arrays. Keep inference on
        # CPU even when the saved XGBoost booster was trained using CUDA.
        classifier.set_params(device="cpu")
    failure_probability = float(pipeline.predict_proba(features)[0, 1])
    threshold = float(metadata["thresholds"][selected_model])
    prediction = int(failure_probability >= threshold)

    return {
        "model": selected_model,
        "prediction": prediction,
        "label": "failure" if prediction == 1 else "no failure",
        "failure_probability": failure_probability,
        "threshold": threshold,
    }
