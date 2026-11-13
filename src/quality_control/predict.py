"""Saved-model prediction for a single uploaded tablet image."""

import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from .config import CLASS_NAMES, MODEL_DIR, MODEL_METADATA
from .explainability import explain_prediction
from .models import load_model
from .preprocessing import build_image_transform


def _open_rgb_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        source = Image.open(image)
    elif hasattr(image, "read"):
        if hasattr(image, "seek"):
            image.seek(0)
        source = Image.open(image)
    else:
        raise TypeError("image must be a PIL image, filesystem path, or file-like upload.")

    try:
        source.load()
        return source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read the uploaded tablet image: {error}") from error
    finally:
        source.close()


def predict_tablet(image) -> dict:
    """Return Normal/Defective, confidence, and a Grad-CAM PIL image.

    The prediction describes visible image patterns only. It does not verify
    chemical quality, product identity, dosage, or medicine safety.
    """
    metadata_path = Path(MODEL_METADATA)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained tablet model found at {metadata_path}. Run "
            "`python -m src.quality_control.train` first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = metadata.get("recommended_model")
    model_files = metadata.get("model_files", {})
    if model_name not in model_files:
        raise ValueError("Tablet model metadata does not contain a trained recommended model.")

    checkpoint_path = Path(model_files[model_name])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(MODEL_DIR) / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Saved tablet model checkpoint is missing: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, checkpoint_path, device=device)
    original_image = _open_rgb_image(image)
    image_tensor = build_image_transform(training=False)(original_image)

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()
    predicted_index = int(probabilities.argmax().item())
    gradcam_image = explain_prediction(
        model,
        image_tensor,
        original_image=original_image,
        target_class=predicted_index,
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "class_probabilities": {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "model": model_name,
        "gradcam_explanation": gradcam_image,
        "scope_note": "Visual classification only; does not verify chemical quality or medicine safety.",
    }
