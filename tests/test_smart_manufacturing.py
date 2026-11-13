"""Focused checks for the real-data Smart Manufacturing pipeline."""

import json
import unittest

import pandas as pd

from src.smart_manufacturing.config import DATASET_PATH, FEATURE_COLUMNS, MODEL_DIR
from src.smart_manufacturing.predict import predict_machine
from src.smart_manufacturing.preprocessing import (
    load_and_prepare_data,
    make_train_validation_test_splits,
    validate_dataset,
)


class SmartManufacturingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Run the pipeline first to download the UCI dataset.")
        cls.dataframe = load_and_prepare_data(DATASET_PATH)

    def test_schema_and_labels_validate(self):
        validate_dataset(self.dataframe)
        self.assertEqual(set(self.dataframe["Machine failure"].unique()), {0, 1})

    def test_feature_selection_excludes_identifiers_and_failure_modes(self):
        splits = make_train_validation_test_splits(self.dataframe)
        self.assertEqual(list(splits["X_train"].columns), FEATURE_COLUMNS)
        self.assertNotIn("UDI", splits["X_train"].columns)
        self.assertNotIn("Product ID", splits["X_train"].columns)
        for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
            self.assertNotIn(mode, splits["X_train"].columns)

    def test_splits_are_reproducible_and_stratified(self):
        first = make_train_validation_test_splits(self.dataframe)
        second = make_train_validation_test_splits(self.dataframe)
        self.assertEqual(first["indices"], second["indices"])
        self.assertEqual(
            sum(len(first[f"y_{name}"]) for name in ("train", "validation", "test")),
            len(self.dataframe),
        )
        for name in ("train", "validation", "test"):
            self.assertGreater(int(first[f"y_{name}"].sum()), 0)

    def test_saved_pipeline_loads_and_predicts_a_real_row(self):
        metadata_path = MODEL_DIR / "model_metadata.json"
        if not metadata_path.exists():
            self.skipTest("Run the training pipeline before testing saved predictions.")
        row = self.dataframe.iloc[0][FEATURE_COLUMNS].to_dict()
        result = predict_machine(row)
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["failure_probability"], 0.0)
        self.assertLessEqual(result["failure_probability"], 1.0)
        self.assertTrue((MODEL_DIR / json.loads(metadata_path.read_text())["model_files"][result["model"]]).exists())


if __name__ == "__main__":
    unittest.main()
