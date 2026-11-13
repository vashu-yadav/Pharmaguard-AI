"""Landing page and system overview."""

import streamlit as st

from dashboard_utils import QC_REPORTS, SMART_REPORTS, format_metric, load_metrics


def render_home_page():
    st.markdown(
        "<div class='hero'><h1>PharmaGuard AI</h1><p>Explainable machine learning "
        "for two independent pharmaceutical manufacturing workflows.</p></div>",
        unsafe_allow_html=True,
    )
    st.write(
        "Explore predictive maintenance signals from machine sensor readings and "
        "visible tablet-image defects. Each module uses its own dataset and saved model."
    )
    try:
        smart = load_metrics(str(SMART_REPORTS / "metrics.json"))
        quality = load_metrics(str(QC_REPORTS / "metrics.json"))
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Saved evaluation results could not be loaded: {error}")
        smart, quality = {}, {}

    left, right = st.columns(2)
    with left:
        st.markdown(
            "<div class='module-card'><h3>Smart Manufacturing</h3>"
            "<p>AI4I 2020 sensor data | Logistic Regression, Random Forest and XGBoost | SHAP</p>"
            "<p class='muted'>Classifies recorded machine failure risk from six sensor features.</p></div>",
            unsafe_allow_html=True,
        )
        st.caption("Recommended saved model: XGBoost")
        st.metric("XGBoost test F1", format_metric(smart.get("xgboost", {}).get("test", {}).get("f1")))
    with right:
        st.markdown(
            "<div class='module-card'><h3>Tablet Quality Control</h3>"
            "<p>PharmaVision image data | Custom CNN and MobileNetV2 | Grad-CAM</p>"
            "<p class='muted'>Classifies visible image patterns as Normal or Defective.</p></div>",
            unsafe_allow_html=True,
        )
        st.caption("Recommended saved model: MobileNetV2")
        st.metric("MobileNetV2 test F1", format_metric(quality.get("mobilenet_v2", {}).get("test", {}).get("f1")))

    st.subheader("System overview")
    a, b, c = st.columns(3)
    a.metric("Independent modules", "2")
    b.metric("Saved model families", "5")
    c.metric("Explainability methods", "SHAP | Grad-CAM")
    st.info(
        "These are research decision-support models evaluated on their provided datasets. "
        "A tablet image prediction describes visible patterns only; it does not verify "
        "chemical composition, dosage, or medicine safety."
    )
