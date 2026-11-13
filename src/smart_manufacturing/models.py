"""Model definitions and shared training for Smart Manufacturing."""

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def _cuda_is_available() -> bool:
    """Ask PyTorch whether a CUDA device is available without requiring it."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _pipeline(classifier, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", classifier),
        ]
    )


def fit_models(splits: dict) -> tuple[dict, dict]:
    """Fit all three models on the exact same training partition.

    The XGBoost CUDA attempt falls back to CPU if its installed build or GPU
    runtime cannot train. Scikit-learn's Logistic Regression and Random Forest
    run on CPU.
    """
    positive_count = int(splits["y_train"].sum())
    negative_count = int(len(splits["y_train"]) - positive_count)
    if positive_count == 0:
        raise ValueError("The training split contains no failure examples.")
    positive_weight = negative_count / positive_count

    estimators = {
        "logistic_regression": _pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
    }

    use_cuda = _cuda_is_available()
    xgb_common = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "scale_pos_weight": positive_weight,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
    }
    xgb_device = "cuda" if use_cuda else "cpu"
    estimators["xgboost"] = _pipeline(
        XGBClassifier(**xgb_common, device=xgb_device),
        scale_numeric=False,
    )

    fitted = {}
    devices = {"logistic_regression": "cpu", "random_forest": "cpu"}
    for name, estimator in estimators.items():
        try:
            estimator.fit(splits["X_train"], splits["y_train"])
            fitted[name] = estimator
            if name == "xgboost" and xgb_device == "cuda":
                # Training used CUDA. Inputs are transformed to CPU arrays by
                # scikit-learn, so switch XGBoost's prediction setting to CPU
                # to avoid device-mismatch copies during evaluation/inference.
                estimator.named_steps["classifier"].set_params(device="cpu")
                devices[name] = "cuda training; cpu prediction"
            else:
                devices[name] = "cpu"
        except Exception as error:
            if name != "xgboost" or xgb_device != "cuda":
                raise
            warnings.warn(
                "XGBoost could not train with CUDA; retrying on CPU. "
                f"CUDA error: {error}",
                RuntimeWarning,
            )
            cpu_estimator = _pipeline(
                XGBClassifier(**xgb_common, device="cpu"),
                scale_numeric=False,
            )
            cpu_estimator.fit(splits["X_train"], splits["y_train"])
            fitted[name] = cpu_estimator
            devices[name] = "cpu (CUDA fallback)"

    return fitted, devices
