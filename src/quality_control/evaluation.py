"""Evaluation metrics and plots for the original PharmaVision splits."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def evaluate_model(model, image_data, labels=None, device=None) -> dict:
    """Calculate binary metrics and probabilities for a model and data loader.

    The defective class has index 1 and is treated as the positive class.
    ``image_data`` should be a PyTorch DataLoader using a non-augmented split.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    true_labels = []
    probabilities = []
    with torch.no_grad():
        for images, batch_labels in image_data:
            logits = model(images.to(device, non_blocking=True))
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probabilities.cpu().tolist())
            true_labels.extend(batch_labels.cpu().tolist())

    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    truth = np.asarray(true_labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "true_labels": truth,
        "probabilities": scores,
        "predictions": predictions,
    }


def save_training_history(model_name: str, history: dict, report_dir: Path) -> Path:
    """Save loss and accuracy curves for one model's train/validation history."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Training")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    fig.suptitle(f"{model_name.replace('_', ' ').title()} training history")
    fig.tight_layout()
    output_path = report_dir / f"{model_name}_training_history.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_evaluation_reports(results: dict, report_dir: Path) -> None:
    """Write test metrics, confusion matrices, ROC curves, and comparison plot."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for model_name, splits in results.items():
        serializable[model_name] = {
            split_name: {
                key: value
                for key, value in metrics.items()
                if key not in {"true_labels", "probabilities", "predictions"}
            }
            for split_name, metrics in splits.items()
        }
    (report_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )

    with (report_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        columns = ["model", "split", *METRIC_NAMES]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for model_name, splits in serializable.items():
            for split_name, metrics in splits.items():
                writer.writerow(
                    {
                        "model": model_name,
                        "split": split_name,
                        **{metric: metrics[metric] for metric in METRIC_NAMES},
                    }
                )

    _plot_confusion_matrices(results, report_dir)
    _plot_roc_curves(results, report_dir)
    _plot_comparison(results, report_dir)


def _plot_confusion_matrices(results: dict, report_dir: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        matrix = np.asarray(results[name]["test"]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1], labels=["Normal", "Defective"])
        axis.set_yticks([0, 1], labels=["Normal", "Defective"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def _plot_roc_curves(results: dict, report_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, splits in results.items():
        truth = splits["test"]["true_labels"]
        scores = splits["test"]["probabilities"]
        false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name.replace('_', ' ').title()} (AUC={splits['test']['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("PharmaVision test-set ROC curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def _plot_comparison(results: dict, report_dir: Path) -> None:
    names = list(results)
    positions = np.arange(len(names))
    bar_width = 0.15
    fig, axis = plt.subplots(figsize=(11, 6))
    for metric_index, metric in enumerate(METRIC_NAMES):
        values = [results[name]["test"][metric] for name in names]
        offset = (metric_index - (len(METRIC_NAMES) - 1) / 2) * bar_width
        axis.bar(positions + offset, values, bar_width, label=metric.upper())
    axis.set_xticks(positions, labels=[name.replace("_", " ").title() for name in names])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("CNN vs. MobileNetV2 on the original test split")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "model_comparison.png", dpi=160)
    plt.close(fig)
