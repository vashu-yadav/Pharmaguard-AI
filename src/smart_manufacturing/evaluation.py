"""Threshold selection, real-data metrics, and evaluation plots."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def select_f1_threshold(y_true, probabilities) -> float:
    """Choose the validation threshold with best failure-class F1.

    Ties are resolved in favor of the higher threshold, which tends to reduce
    false alarms. This is a validation decision, never a test-set decision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_f1 = np.max(f1_values)
    tied = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = tied[-1]
    return float(thresholds[best_index])


def _calculate_metrics(y_true, probabilities, threshold) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def evaluate_models(models: dict, splits: dict, report_dir: Path) -> dict:
    """Evaluate each fitted model and write metrics/plots from real splits."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    test_probabilities = {}

    for name, model in models.items():
        validation_prob = model.predict_proba(splits["X_validation"])[:, 1]
        threshold = select_f1_threshold(splits["y_validation"], validation_prob)
        validation_metrics = _calculate_metrics(
            splits["y_validation"], validation_prob, threshold
        )

        test_prob = model.predict_proba(splits["X_test"])[:, 1]
        test_probabilities[name] = test_prob
        test_metrics = _calculate_metrics(splits["y_test"], test_prob, threshold)
        results[name] = {
            "threshold_selected_on_validation": threshold,
            "validation": validation_metrics,
            "test": test_metrics,
        }

    _save_metrics(results, report_dir)
    _save_confusion_matrices(results, splits, test_probabilities, report_dir)
    _save_roc_curve(splits["y_test"], test_probabilities, report_dir)
    _save_comparison_graph(results, report_dir)

    best_name = max(
        results,
        key=lambda name: (
            results[name]["validation"]["f1"],
            results[name]["validation"]["roc_auc"],
        ),
    )
    results["recommended_model"] = best_name
    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _save_metrics(results: dict, report_dir: Path) -> None:
    """Write machine-readable validation/test metrics."""
    json_results = {
        name: values for name, values in results.items() if name != "recommended_model"
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(json_results, indent=2), encoding="utf-8"
    )

    with (report_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["model", "split", "threshold", *METRIC_NAMES]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for name, values in json_results.items():
            for split_name in ("validation", "test"):
                writer.writerow(
                    {
                        "model": name,
                        "split": split_name,
                        **values[split_name],
                    }
                )


def _save_confusion_matrices(results, splits, test_probabilities, report_dir) -> None:
    fig, axes = plt.subplots(1, len(test_probabilities), figsize=(5 * len(test_probabilities), 4))
    axes = np.atleast_1d(axes)
    for axis, (name, probabilities) in zip(axes, test_probabilities.items()):
        threshold = results[name]["threshold_selected_on_validation"]
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(splits["y_test"], predictions, labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{name.replace('_', ' ').title()}\nthreshold={threshold:.3f}")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_xticks([0, 1], labels=["No failure", "Failure"])
        axis.set_yticks([0, 1], labels=["No failure", "Failure"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def _save_roc_curve(y_true, test_probabilities, report_dir) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, probabilities in test_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        score = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name.replace('_', ' ').title()} (AUC={score:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Test-set ROC curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def _save_comparison_graph(results, report_dir) -> None:
    names = [name for name in results if name != "recommended_model"]
    x_positions = np.arange(len(names))
    metric_width = 0.15
    fig, axis = plt.subplots(figsize=(11, 6))
    for metric_index, metric in enumerate(METRIC_NAMES):
        values = [results[name]["test"][metric] for name in names]
        offset = (metric_index - (len(METRIC_NAMES) - 1) / 2) * metric_width
        axis.bar(x_positions + offset, values, metric_width, label=metric.upper())
    axis.set_xticks(x_positions, labels=[name.replace("_", " ").title() for name in names])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Test-set model comparison")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "model_comparison.png", dpi=160)
    plt.close(fig)
