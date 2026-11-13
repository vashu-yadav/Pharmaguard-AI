"""Train, evaluate, and explain the PharmaVision tablet classifiers.

Run from the repository root with:
    python -m src.quality_control.train
"""

import hashlib
import json
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .config import (
    CLASS_NAMES,
    DATASET_ARCHIVE,
    DATASET_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_DIR,
    MODEL_NAMES,
    RANDOM_STATE,
    REPORT_DIR,
)
from .download_data import ensure_dataset
from .evaluation import evaluate_model, save_evaluation_reports, save_training_history
from .models import load_model, train_model
from .predict import predict_tablet
from .preprocessing import prepare_image_dataset, validate_image_dataset


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_gradcam_example(loaders: dict, report_dir: Path) -> dict:
    """Save an actual held-out image beside its model-generated Grad-CAM."""
    test_dataset = loaders["test"].dataset
    sample_index = next(
        (index for index, (_path, label) in enumerate(test_dataset.samples) if label == 1),
        0,
    )
    image_path, actual_label = test_dataset.samples[sample_index]
    with Image.open(image_path) as source:
        original = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    prediction = predict_tablet(original)
    explanation = prediction["gradcam_explanation"]
    canvas = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE + 34), "white")
    canvas.paste(original, (0, 34))
    canvas.paste(explanation, (IMAGE_SIZE, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original test image", fill="black")
    draw.text((IMAGE_SIZE + 8, 8), "Grad-CAM overlay", fill="black")
    output_path = Path(report_dir) / "gradcam_example.png"
    canvas.save(output_path)

    details = {
        "source_image": str(image_path),
        "actual_class": CLASS_NAMES[int(actual_label)],
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "model": prediction["model"],
        "scope_note": prediction["scope_note"],
    }
    details_path = Path(report_dir) / "gradcam_example.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return {"image": output_path.name, "details": details_path.name, **details}


def run_training() -> dict:
    """Download/validate data, train both models, evaluate, and save reports."""
    dataset_root = ensure_dataset()
    dataset_summary = validate_image_dataset(dataset_root)
    loaders = prepare_image_dataset(dataset_root)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PharmaVision dataset: {dataset_root}")
    print(f"Training device: {device}")

    training_runs = {}
    checkpoint_paths = {}
    training_devices = {}
    evaluation_devices = {}
    for model_name in MODEL_NAMES:
        checkpoint_path = MODEL_DIR / f"{model_name}_best.pth"
        try:
            run = train_model(model_name, loaders, checkpoint_path, device=device)
            training_devices[model_name] = str(device)
            evaluation_devices[model_name] = device
        except RuntimeError as error:
            if device.type != "cuda":
                raise
            warnings.warn(
                f"{model_name} failed on CUDA; retrying on CPU. Details: {error}",
                RuntimeWarning,
            )
            checkpoint_path.unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            run = train_model(
                model_name,
                loaders,
                checkpoint_path,
                device=torch.device("cpu"),
            )
            training_devices[model_name] = "cpu (CUDA fallback)"
            evaluation_devices[model_name] = torch.device("cpu")

        training_runs[model_name] = run
        checkpoint_paths[model_name] = checkpoint_path
        save_training_history(model_name, run["history"], REPORT_DIR)

    (REPORT_DIR / "training_history.json").write_text(
        json.dumps(
            {
                name: {
                    **run,
                    "checkpoint_path": Path(run["checkpoint_path"]).name,
                }
                for name, run in training_runs.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model_results = {}
    for model_name, checkpoint_path in checkpoint_paths.items():
        evaluation_device = evaluation_devices[model_name]
        model = load_model(model_name, checkpoint_path, device=evaluation_device)
        model_results[model_name] = {
            "validation": evaluate_model(model, loaders["validation"], device=evaluation_device),
            "test": evaluate_model(model, loaders["test"], device=evaluation_device),
        }

    save_evaluation_reports(model_results, REPORT_DIR)
    recommended_model = max(
        model_results,
        key=lambda name: (
            model_results[name]["validation"]["f1"],
            model_results[name]["validation"]["roc_auc"],
        ),
    )
    archive_hash = _sha256(DATASET_ARCHIVE) if DATASET_ARCHIVE.exists() else None

    metadata = {
        "dataset": "PharmaVision (PV), Zenodo record 22773411",
        "dataset_root": str(dataset_root.relative_to(DATASET_DIR.parents[1])),
        "dataset_archive_sha256": archive_hash,
        "class_names": list(CLASS_NAMES),
        "positive_class": "Defective",
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "random_state": RANDOM_STATE,
        "training_device": training_devices,
        "recommended_model": recommended_model,
        "model_files": {name: path.name for name, path in checkpoint_paths.items()},
        "best_epochs": {name: run["best_epoch"] for name, run in training_runs.items()},
        "package_versions": {
            package: _package_version(package)
            for package in ("torch", "torchvision", "pillow", "scikit-learn")
        },
        "dataset_summary": dataset_summary["splits"],
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    (REPORT_DIR / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )
    gradcam_result = _save_gradcam_example(loaders, REPORT_DIR)
    (REPORT_DIR / "gradcam_example.json").write_text(
        json.dumps(gradcam_result, indent=2), encoding="utf-8"
    )

    print("Test-set model results:")
    for model_name in MODEL_NAMES:
        metrics = model_results[model_name]["test"]
        print(
            f"  {model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
            f"device={training_devices[model_name]}, "
            f"best_epoch={training_runs[model_name]['best_epoch']}"
        )
    print(f"Recommended model (validation F1, then validation ROC-AUC): {recommended_model}")
    print(f"Saved models: {MODEL_DIR}")
    print(f"Saved reports: {REPORT_DIR}")
    return {"metadata": metadata, "results": model_results}


if __name__ == "__main__":
    run_training()
