"""Dataset validation, splitting, and leakage-safe preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Raise a clear error if a dataframe is not the expected AI4I dataset."""
    if dataframe.empty:
        raise ValueError("The AI4I dataset has no rows.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "The AI4I dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"The target column {TARGET_COLUMN!r} contains missing labels.")

    numeric_target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if numeric_target.isna().any():
        raise ValueError(f"The target column {TARGET_COLUMN!r} contains non-numeric labels.")
    target_values = set(numeric_target)
    if target_values != {0, 1}:
        raise ValueError(
            f"{TARGET_COLUMN!r} must contain both binary labels 0 and 1; "
            f"found {sorted(target_values)}."
        )

    for column in FAILURE_MODE_COLUMNS:
        if dataframe[column].isna().any():
            raise ValueError(f"Failure-mode label {column!r} contains missing values.")
        numeric_modes = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric_modes.isna().any():
            raise ValueError(f"Failure-mode label {column!r} contains non-numeric values.")
        values = set(numeric_modes)
        if not values.issubset({0, 1}):
            raise ValueError(f"Failure-mode label {column!r} must contain only 0 or 1.")

    invalid_types = set(dataframe["Type"].dropna().unique()) - {"L", "M", "H"}
    if invalid_types:
        raise ValueError(f"Unexpected values in 'Type': {sorted(invalid_types)}.")

    # Numeric feature columns may contain missing values: the model pipeline
    # imputes them using statistics learned from the training partition only.
    for column in NUMERIC_FEATURES:
        pd.to_numeric(dataframe[column], errors="raise")


def count_failure_mode_disagreements(dataframe: pd.DataFrame) -> int:
    """Count rows where the overall label differs from the five mode flags.

    Keep ``Machine failure`` as the official binary target. Some released rows
    may have an overall positive label without a corresponding mode flag; this
    check reports the discrepancy but never rewrites labels.
    """
    modes = dataframe[FAILURE_MODE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mode_union = modes.max(axis=1).astype(int)
    overall = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce").astype(int)
    return int((mode_union != overall).sum())


def load_and_prepare_data(data_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read and validate the local dataset; this function does not download it."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found at {data_path}. Run the training entry point "
            "to download it automatically from UCI."
        )

    try:
        dataframe = pd.read_csv(data_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read AI4I CSV at {data_path}: {error}") from error

    validate_dataset(dataframe)
    return dataframe


def make_train_validation_test_splits(dataframe: pd.DataFrame) -> dict:
    """Create reproducible stratified 70/15/15 splits using row positions."""
    validate_dataset(dataframe)
    row_indices = np.arange(len(dataframe))
    labels = dataframe[TARGET_COLUMN].astype(int)

    train_indices, remaining_indices = train_test_split(
        row_indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=labels.iloc[remaining_indices],
    )

    def partition(indices):
        rows = dataframe.iloc[indices]
        features = rows[FEATURE_COLUMNS].copy()
        target = rows[TARGET_COLUMN].astype(int).copy()
        return features, target, indices.tolist()

    x_train, y_train, train_rows = partition(train_indices)
    x_validation, y_validation, validation_rows = partition(validation_indices)
    x_test, y_test, test_rows = partition(test_indices)

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_validation": x_validation,
        "y_validation": y_validation,
        "X_test": x_test,
        "y_test": y_test,
        "indices": {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        },
    }


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build imputers/encoders that are fitted only when the model is fitted."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
