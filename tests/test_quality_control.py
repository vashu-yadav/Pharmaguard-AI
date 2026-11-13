"""Basic real-data checks for tablet preprocessing and saved prediction."""

import unittest

from PIL import Image
import torch

from src.quality_control.config import DATASET_DIR, MODEL_DIR
from src.quality_control.predict import predict_tablet
from src.quality_control.preprocessing import (
    build_image_transform,
    find_dataset_root,
    validate_image_dataset,
)


class QualityControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.dataset_root = find_dataset_root(DATASET_DIR)
        except FileNotFoundError as error:
            raise unittest.SkipTest(f"Download PharmaVision first: {error}")
        cls.summary = validate_image_dataset(cls.dataset_root)

    def test_original_split_counts_and_classes(self):
        splits = self.summary["splits"]
        self.assertEqual(splits["train"], {"defective": 190, "normal": 190})
        self.assertEqual(splits["validation"], {"defective": 30, "normal": 30})
        self.assertEqual(splits["test"], {"defective": 100, "normal": 100})

    def test_preprocessing_returns_normalized_224_square_tensor(self):
        image_path = next((self.dataset_root / "Train").rglob("*.jpg"), None)
        if image_path is None:
            image_path = next(
                path
                for path in (self.dataset_root / "Train").rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpeg", ".png", ".bmp", ".webp"}
            )
        with Image.open(image_path) as image_file:
            tensor = build_image_transform(training=False)(image_file.convert("RGB"))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all().item())
        transform_names = [type(item).__name__ for item in build_image_transform(False).transforms]
        self.assertNotIn("RandomRotation", transform_names)
        self.assertNotIn("ColorJitter", transform_names)

    def test_saved_model_prediction_and_gradcam(self):
        if not (MODEL_DIR / "model_metadata.json").exists():
            self.skipTest("Train the tablet models before checking saved-model prediction.")
        image_path = next(
            path
            for path in (self.dataset_root / "Test").rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        result = predict_tablet(image_path)
        self.assertIn(result["predicted_class"], {"Normal", "Defective"})
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIsInstance(result["gradcam_explanation"], Image.Image)
        self.assertEqual(result["gradcam_explanation"].size, (224, 224))
        self.assertIn("does not verify", result["scope_note"])


if __name__ == "__main__":
    unittest.main()
