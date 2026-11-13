"""Tablet image upload and saved MobileNetV2 prediction page."""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.quality_control.predict import predict_tablet

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def _validate_upload(uploaded_file) -> Image.Image:
    """Decode a bounded image upload and return an orientation-correct RGB copy."""
    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Please upload an image smaller than 10 MB.")
    try:
        with Image.open(BytesIO(content)) as check:
            if check.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                raise ValueError("Use a JPEG, PNG, WebP, BMP, or TIFF image.")
            if check.width * check.height > MAX_IMAGE_PIXELS:
                raise ValueError("The image is too large to process (maximum 20 megapixels).")
            check.verify()
        with Image.open(BytesIO(content)) as source:
            return ImageOps.exif_transpose(source).convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"The uploaded file is not a readable image: {error}") from error


def render_tablet_page():
    st.title("Tablet Quality Control")
    st.write(
        "Upload a tablet image to classify its visible image pattern with the saved "
        "MobileNetV2 model. The result is image-based decision support."
    )
    uploaded = st.file_uploader(
        "Choose a tablet image", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        help="Maximum size 10 MB and 20 megapixels.",
    )
    if uploaded is not None:
        st.session_state.pop("tablet_image", None)
        st.session_state.pop("tablet_result", None)
        try:
            image = _validate_upload(uploaded)
            with st.spinner("Loading the saved tablet model and calculating Grad-CAM..."):
                result = predict_tablet(image)
            st.session_state["tablet_image"] = image
            st.session_state["tablet_result"] = result
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
            st.error(f"Could not analyze this image: {error}")
        except Exception as error:
            st.error(f"Tablet prediction encountered an unexpected issue: {error}")

    image = st.session_state.get("tablet_image")
    result = st.session_state.get("tablet_result")
    if image is not None and result is not None:
        st.divider()
        st.subheader("Prediction result")
        left, right = st.columns([1, 1])
        with left:
            st.metric("Predicted class", result["predicted_class"])
        with right:
            st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.caption(f"Saved model: {result['model']}")
        prob_cols = st.columns(len(result["class_probabilities"]))
        for col, (name, probability) in zip(prob_cols, result["class_probabilities"].items()):
            col.metric(f"{name} score", f"{probability:.1%}")

        original_col, cam_col = st.columns(2)
        with original_col:
            st.image(image, caption="Uploaded tablet image", width="stretch")
        with cam_col:
            st.image(result["gradcam_explanation"], caption="Grad-CAM: highlighted regions influenced this model output", width="stretch")
        st.info(
            "This prediction describes visible image patterns only. It does not verify "
            "chemical composition, product identity, dosage, pharmaceutical quality, or medicine safety."
        )
