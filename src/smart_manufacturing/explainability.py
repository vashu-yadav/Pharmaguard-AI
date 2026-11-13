"""SHAP explanations for the trained Smart Manufacturing model."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def explain_model(model, background_data, instances, max_background_rows: int = 100):
    """Explain failure probabilities using a model-agnostic SHAP explainer.

    ``model`` is a fitted sklearn Pipeline containing both preprocessing and
    classifier steps. The explanation values are contributions to the
    probability of ``Machine failure = 1``.
    """
    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("Expected a fitted pipeline with preprocessor and classifier steps.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    background = preprocessor.transform(background_data)
    explained = preprocessor.transform(instances)

    if hasattr(background, "toarray"):
        background = background.toarray()
    if hasattr(explained, "toarray"):
        explained = explained.toarray()

    background = np.asarray(background)
    explained = np.asarray(explained)
    if len(background) > max_background_rows:
        background = background[:max_background_rows]

    background_frame = pd.DataFrame(background, columns=feature_names)
    explained_frame = pd.DataFrame(explained, columns=feature_names)

    def failure_probability(transformed_rows):
        return classifier.predict_proba(np.asarray(transformed_rows))[:, 1]

    masker = shap.maskers.Independent(background_frame, max_samples=max_background_rows)
    explainer = shap.Explainer(
        failure_probability,
        masker,
        algorithm="permutation",
        output_names=["Machine failure probability"],
    )
    minimum_evaluations = 2 * len(feature_names) + 1
    return explainer(explained_frame, max_evals=minimum_evaluations)


def generate_shap_reports(
    model,
    background_data,
    test_features,
    test_labels,
    report_dir: Path,
    max_explanation_rows: int = 300,
) -> dict:
    """Save a summary, global importance, and one real test-row explanation."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if len(test_features) > max_explanation_rows:
        sampled = test_features.sample(
            n=max_explanation_rows,
            random_state=42,
        )
        explanation_labels = test_labels.loc[sampled.index]
    else:
        sampled = test_features
        explanation_labels = test_labels.loc[sampled.index]

    explanation = explain_model(model, background_data, sampled)
    values = np.asarray(explanation.values)
    feature_names = list(explanation.feature_names)
    mean_absolute_values = np.mean(np.abs(values), axis=0)

    summary_path = report_dir / "shap_summary.png"
    shap.plots.beeswarm(explanation, max_display=len(feature_names), show=False)
    plt.gcf().tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    importance_path = report_dir / "shap_global_importance.png"
    shap.plots.bar(explanation, max_display=len(feature_names), show=False)
    plt.gcf().tight_layout()
    plt.savefig(importance_path, dpi=160, bbox_inches="tight")
    plt.close()

    importance_csv = report_dir / "shap_feature_importance.csv"
    order = np.argsort(mean_absolute_values)[::-1]
    with importance_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["feature", "mean_absolute_shap_value"])
        for index in order:
            writer.writerow([feature_names[index], float(mean_absolute_values[index])])

    positive_positions = np.flatnonzero(explanation_labels.to_numpy() == 1)
    chosen_position = int(positive_positions[0]) if len(positive_positions) else 0
    chosen_row = explanation[chosen_position]
    individual_path = report_dir / "shap_individual_explanation.png"
    shap.plots.waterfall(chosen_row, max_display=len(feature_names), show=False)
    plt.gcf().tight_layout()
    plt.savefig(individual_path, dpi=160, bbox_inches="tight")
    plt.close()

    individual_details = {
        "test_row_index": str(sampled.index[chosen_position]),
        "actual_machine_failure": int(explanation_labels.iloc[chosen_position]),
        "predicted_failure_probability": float(
            model.predict_proba(sampled.iloc[[chosen_position]])[0, 1]
        ),
        "base_value": float(np.asarray(chosen_row.base_values).reshape(-1)[0]),
        "feature_contributions": {
            feature_names[index]: float(values[chosen_position, index])
            for index in range(len(feature_names))
        },
        "explanation_output": "failure probability",
    }
    individual_json = report_dir / "shap_individual_explanation.json"
    individual_json.write_text(json.dumps(individual_details, indent=2), encoding="utf-8")

    return {
        "summary_plot": str(summary_path),
        "global_importance_plot": str(importance_path),
        "importance_csv": str(importance_csv),
        "individual_plot": str(individual_path),
        "individual_details": str(individual_json),
    }
