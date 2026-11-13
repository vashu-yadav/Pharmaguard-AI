"""Explainability views for tabular SHAP and image Grad-CAM."""

import streamlit as st

from dashboard_utils import QC_REPORTS, SMART_REPORTS
from machine_page import _explain_current_prediction


def render_explainability_page():
    st.title("Explainable AI")
    st.write("The explanations below belong to different models and data types; they are not combined.")
    tabular, image = st.tabs(["Machine sensors | SHAP", "Tablet images | Grad-CAM"])
    with tabular:
        st.subheader("Smart Manufacturing: SHAP")
        st.write(
            "SHAP estimates how each sensor feature moved the model output for a machine input. "
            "A positive contribution pushes the estimated failure probability up; a negative one pushes it down."
        )
        payload = st.session_state.get("machine_payload")
        model_name = st.session_state.get("machine_model")
        if payload and model_name:
            st.markdown("#### Explanation for the latest machine input")
            try:
                with st.spinner("Calculating SHAP explanation..."):
                    _explain_current_prediction(model_name, payload)
            except Exception as error:
                st.error(f"Could not calculate the saved-model SHAP explanation: {error}")
        else:
            st.info("Run a machine prediction to see its individual explanation.")
        for filename, caption in [
            ("shap_summary.png", "Saved SHAP summary across held-out examples"),
            ("shap_global_importance.png", "Saved global mean absolute SHAP importance"),
            ("shap_individual_explanation.png", "Saved example explanation from the evaluation report"),
        ]:
            path = SMART_REPORTS / "shap" / filename
            if path.is_file():
                st.image(str(path), caption=caption, width="stretch")
        st.caption("The global SHAP plots were produced during training/evaluation for the recommended saved model.")

    with image:
        st.subheader("Tablet Quality Control: Grad-CAM")
        st.write(
            "Grad-CAM highlights image regions that contributed to the selected neural-network class. "
            "It is a visual explanation of the model and is not a chemical or safety test."
        )
        result = st.session_state.get("tablet_result")
        original = st.session_state.get("tablet_image")
        if result and original:
            left, right = st.columns(2)
            left.image(original, caption=f"Latest upload | predicted {result['predicted_class']}", width="stretch")
            right.image(result["gradcam_explanation"], caption="Grad-CAM for latest upload", width="stretch")
        else:
            st.info("Upload an image on the Tablet Quality Control page to see its Grad-CAM explanation.")
            path = QC_REPORTS / "gradcam_example.png"
            if path.is_file():
                st.image(str(path), caption="Saved Grad-CAM example from the evaluation report", width="stretch")
        st.warning("Highlighted regions explain image-model attention only; they do not establish tablet identity, composition, or medicine safety.")
