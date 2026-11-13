# PharmaGuard AI

**An Explainable Hybrid Machine Learning Framework for Smart Pharmaceutical Manufacturing and Intelligent Quality Control**

PharmaGuard AI is a student project with two independent machine learning modules. This repository currently contains only the project foundation and placeholder interfaces. It does not contain datasets, trained models, evaluation results, or implemented predictions.

## Project modules

1. **Smart manufacturing:** planned experiments with the AI4I 2020 Predictive Maintenance Dataset using Logistic Regression, Random Forest, and XGBoost; evaluation plots and SHAP explanations are planned.
2. **Pharmaceutical quality control:** planned tablet image preprocessing and augmentation, a CNN and MobileNetV2 comparison, classification evaluation, and Grad-CAM explanations.

The modules use separate data, models, and reports. No relationship between their samples or production batches is assumed.

## Project structure

```text
PharmaGuard-AI/
├── app/                         # Streamlit dashboard placeholders
├── data/
│   ├── smart_manufacturing/     # raw/ and processed/ machine data
│   └── quality_control/         # raw/ and processed/ tablet images
├── models/
│   ├── smart_manufacturing/     # future machine model artifacts
│   └── quality_control/         # future image model artifacts
├── reports/
│   ├── smart_manufacturing/     # future metrics and plots
│   └── quality_control/         # future metrics and plots
├── src/
│   ├── smart_manufacturing/     # preprocessing, models, evaluation, explainability
│   └── quality_control/         # preprocessing, models, evaluation, explainability
├── AGENTS.md                    # project-specific development guidance
└── requirements.txt             # planned Python dependencies
```

## Windows and VS Code setup

Open this folder in VS Code. To create an isolated environment in the VS Code terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When you are ready to install the listed dependencies, run:

```powershell
py -m pip install -r requirements.txt
```

No packages are installed as part of creating this project foundation. The Streamlit pages are placeholders and are not yet a working prediction dashboard.

## Future prediction interfaces

- `src.smart_manufacturing.predict_machine(input_data)` will accept machine input data. It currently raises `NotImplementedError` because no model has been trained or loaded.
- `src.quality_control.predict_tablet(image)` will accept a tablet image. It currently raises `NotImplementedError` because no image model has been trained or loaded.

## Next steps

1. Select and document the dataset locations and label definitions for each independent module.
2. Implement preprocessing and reproducible train/validation/test workflows.
3. Train and compare the specified models using real dataset results.
4. Add explainability outputs and connect the implemented predictors to the Streamlit pages.
