# PharmaGuard AI

**An Explainable Hybrid Machine Learning Framework for Smart Pharmaceutical Manufacturing and Intelligent Quality Control**

PharmaGuard AI is a student project with two independent machine learning modules. The Smart Manufacturing module implements an AI4I 2020 binary failure-classification workflow. The tablet image quality-control module remains a placeholder. Machine data and tablet images are kept separate and are not assumed to belong to shared batches.

## Project modules

1. **Smart manufacturing:** uses the AI4I 2020 Predictive Maintenance Dataset with Logistic Regression, Random Forest, and XGBoost; it writes evaluation plots/metrics and SHAP explanations from actual downloaded data.
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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When you are ready to install the listed dependencies, run:

```powershell
python -m pip install -r requirements.txt
```

### Smart Manufacturing training

From the repository root, run:

```powershell
python -m src.smart_manufacturing.train
```

The command downloads the official UCI archive only if `data/smart_manufacturing/raw/ai4i2020.csv` is absent, validates it, then trains Logistic Regression, Random Forest, and XGBoost on the same stratified train/validation/test split. Trained model pipelines (including preprocessing) are saved under `models/smart_manufacturing/`. Metrics and plots are saved under `reports/smart_manufacturing/`.

The XGBoost trainer tries CUDA when PyTorch reports a usable GPU and falls back to CPU if XGBoost cannot train with CUDA. Logistic Regression and Random Forest use CPU.

Run the focused checks after training with:

```powershell
python -m unittest discover -s tests -v
```

### Current scope

`predict_machine()` loads the recommended saved pipeline after training. The Smart Manufacturing model is a binary classifier for the dataset's recorded `Machine failure` label; it is not a remaining-useful-life model or a guarantee of future failure forecasting. The tablet quality-control code and Streamlit pages remain placeholders and are not yet a connected dashboard.

## Prediction interfaces

- `src.smart_manufacturing.predict_machine(input_data)` loads the recommended saved Smart Manufacturing pipeline and returns a failure class, probability, and validation-selected threshold. Run the training command above first.
- `src.quality_control.predict_tablet(image)` remains a placeholder and raises `NotImplementedError` until a real image model and preprocessing pipeline are implemented.

## Next steps

1. Connect the verified Smart Manufacturing predictor and its reports to the Streamlit machine page.
2. Implement the independent tablet-image dataset, preprocessing, training, evaluation, prediction, and Grad-CAM workflow when authorized.
3. Add automated checks and setup guidance for clean environment installs and dashboard integration.
