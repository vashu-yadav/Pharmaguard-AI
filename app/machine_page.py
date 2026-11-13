"""Interactive Smart Manufacturing inference page."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard_utils import (
    SMART_MODELS,
    SMART_REPORTS,
    format_metric,
    load_metrics,
    load_shap_background,
    load_smart_pipeline,
)
from src.smart_manufacturing.config import FEATURE_COLUMNS
from src.smart_manufacturing.predict import predict_machine

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _explain_current_prediction(model_name: str, payload: dict):
    """Render one local SHAP waterfall using the saved training split as background."""
    from src.smart_manufacturing.explainability import explain_model

    pipeline = load_smart_pipeline(model_name)
    background = load_shap_background()
    instance = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    explanation = explain_model(pipeline, background, instance)
    import shap

    shap.plots.waterfall(explanation[0], max_display=12, show=False)
    figure = plt.gcf()
    try:
        figure.tight_layout()
        st.pyplot(figure, width="stretch", clear_figure=True)
    finally:
        plt.close(figure)


def render_machine_page():
    st.title("Smart Manufacturing")
    st.write(
        "Enter one machine's sensor readings. The saved pipeline estimates whether "
        "the AI4I dataset's machine-failure label is likely for this input."
    )
    metadata_path = SMART_MODELS / "model_metadata.json"
    try:
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trained_models = [name for name in MODEL_LABELS if name in metadata.get("model_files", {})]
        if not trained_models:
            raise ValueError("No trained machine pipelines are listed in model metadata.")
    except (OSError, ValueError) as error:
        st.error(f"Cannot read saved Smart Manufacturing model configuration: {error}")
        return

    selected = st.selectbox(
        "Model", trained_models,
        index=trained_models.index(metadata.get("recommended_model"))
        if metadata.get("recommended_model") in trained_models else 0,
        format_func=lambda name: MODEL_LABELS[name],
    )
    with st.form("machine_sensor_form"):
        st.subheader("Machine sensor readings")
        variant = st.selectbox("Product variant", ["L", "M", "H"], help="AI4I product quality variant.")
        col1, col2 = st.columns(2)
        with col1:
            air = st.number_input("Air temperature [K]", min_value=250.0, max_value=350.0, value=300.0, step=0.1)
            process = st.number_input("Process temperature [K]", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
            speed = st.number_input("Rotational speed [rpm]", min_value=0, max_value=10000, value=1500, step=10)
        with col2:
            torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=1000.0, value=40.0, step=0.5)
            wear = st.number_input("Tool wear [min]", min_value=0, max_value=1000, value=100, step=1)
        submitted = st.form_submit_button("Estimate machine failure", type="primary", width="stretch")

    if submitted:
        st.session_state.pop("machine_result", None)
        st.session_state.pop("machine_payload", None)
        st.session_state.pop("machine_model", None)
        payload = {
            "Type": variant,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }
        try:
            with st.spinner("Running the saved model and preparing the explanation..."):
                result = predict_machine(payload, model_name=selected)
            st.session_state["machine_result"] = result
            st.session_state["machine_payload"] = payload
            st.session_state["machine_model"] = selected
        except Exception as error:
            st.error(f"Machine prediction failed: {error}")

    result = st.session_state.get("machine_result")
    payload = st.session_state.get("machine_payload")
    active_model = st.session_state.get("machine_model")
    if result and payload:
        st.divider()
        st.subheader("Prediction result")
        probability = float(result["failure_probability"])
        confidence = probability if result["prediction"] == 1 else 1.0 - probability
        first, second, third = st.columns(3)
        first.metric("Model output", "Failure flagged" if result["prediction"] else "No failure flagged")
        second.metric("Prediction confidence", f"{confidence:.1%}")
        third.metric("Failure probability", f"{probability:.1%}")
        st.caption(f"Model: {MODEL_LABELS.get(result['model'], result['model'])} | Decision threshold: {result['threshold']:.3f}")
        if result["prediction"]:
            st.warning("The model flagged this input. Review the sensor readings and follow your site's maintenance procedures; this estimate is not a diagnosis of equipment condition.")
        else:
            st.info("The model did not flag this input at its saved threshold. Continue routine monitoring; this does not guarantee failure will not occur.")

        st.markdown("#### SHAP explanation for this input")
        st.caption("Positive contributions push the model toward a failure prediction; negative contributions push away from it.")
        try:
            with st.spinner("Calculating SHAP contributions from the saved training data..."):
                _explain_current_prediction(active_model, payload)
        except Exception as error:
            st.error(f"Could not generate the local SHAP explanation: {error}")

    try:
        metrics = load_metrics(str(SMART_REPORTS / "metrics.json"))
        st.divider()
        st.subheader("Saved test-set performance")
        rows = []
        for name in trained_models:
            values = metrics.get(name, {}).get("test", {})
            rows.append({"Model": MODEL_LABELS[name], **{label.upper().replace("_", "-"): values.get(label) for label in METRICS}})
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}", na_rep="N/A"), width="stretch")
        plot_path = SMART_REPORTS / "shap" / "shap_global_importance.png"
        if plot_path.is_file():
            st.image(str(plot_path), caption="Saved global SHAP feature importance for the recommended model", width="stretch")
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Saved machine evaluation results could not be displayed: {error}")
