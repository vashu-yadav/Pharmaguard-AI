"""Download data, train models, evaluate them, and create SHAP reports.

Run from the repository root with:
    python -m src.smart_manufacturing.train
"""

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version

import joblib

from .config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_NAMES,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
)
from .download_data import ensure_dataset
from .evaluation import evaluate_models
from .explainability import generate_shap_reports
from .models import fit_models
from .preprocessing import (
    count_failure_mode_disagreements,
    load_and_prepare_data,
    make_train_validation_test_splits,
)


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _class_counts(labels) -> dict:
    counts = labels.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def run_pipeline() -> dict:
    """Run the full workflow on the official dataset and save genuine outputs."""
    dataset_path = ensure_dataset(DATASET_PATH)
    dataframe = load_and_prepare_data(dataset_path)
    splits = make_train_validation_test_splits(dataframe)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    split_summary = {}
    for split_name in ("train", "validation", "test"):
        labels = splits[f"y_{split_name}"]
        split_summary[split_name] = {
            "rows": int(len(labels)),
            "class_counts": _class_counts(labels),
        }

    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset_summary = {
        "source": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "file": str(dataset_path.relative_to(DATASET_PATH.parents[2])),
        "sha256": digest,
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "class_counts": _class_counts(dataframe[TARGET_COLUMN]),
        "failure_mode_union_disagreements": count_failure_mode_disagreements(dataframe),
        "splits": split_summary,
        "random_state": RANDOM_STATE,
    }
    (REPORT_DIR / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )
    split_manifest = {
        "random_state": RANDOM_STATE,
        "dataset_sha256": digest,
        **splits["indices"],
    }
    (REPORT_DIR / "split_indices.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    fitted_models, devices = fit_models(splits)
    model_files = {}
    for name, pipeline in fitted_models.items():
        filename = f"{name}_pipeline.joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        model_files[name] = filename

    results = evaluate_models(fitted_models, splits, REPORT_DIR)
    recommended_model = results["recommended_model"]
    shap_files = generate_shap_reports(
        fitted_models[recommended_model],
        splits["X_train"],
        splits["X_test"],
        splits["y_test"],
        REPORT_DIR / "shap",
    )

    thresholds = {
        name: float(values["threshold_selected_on_validation"])
        for name, values in results.items()
        if name in MODEL_NAMES
    }
    metadata = {
        "source": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "dataset_sha256": digest,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "split_sizes": split_summary,
        "recommended_model": recommended_model,
        "model_files": model_files,
        "thresholds": thresholds,
        "training_device": devices,
        "package_versions": {
            package: _package_version(package)
            for package in (
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
                "shap",
                "joblib",
            )
        },
        "shap_reports": shap_files,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(dataframe)} | SHA-256: {digest}")
    print("Model test results:")
    for name in MODEL_NAMES:
        metrics = results[name]["test"]
        print(
            f"  {name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
            f"threshold={metrics['threshold']:.4f}, device={devices[name]}"
        )
    print(f"Recommended model (validation F1, then validation ROC-AUC): {recommended_model}")
    print(f"Models: {MODEL_DIR}")
    print(f"Reports: {REPORT_DIR}")
    return {"metadata": metadata, "results": results}


if __name__ == "__main__":
    run_pipeline()
