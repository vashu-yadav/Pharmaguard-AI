# PharmaGuard AI

**An Explainable Hybrid Machine Learning Framework for Smart Pharmaceutical Manufacturing and Intelligent Quality Control**

PharmaGuard AI is a student project with two independent machine learning modules. The Smart Manufacturing module implements an AI4I 2020 binary failure-classification workflow. The Tablet Quality Control module trains a custom CNN and pretrained MobileNetV2 on PharmaVision. Machine data and tablet images are kept separate and are not assumed to belong to shared batches.

## Project modules

1. **Smart manufacturing:** uses the AI4I 2020 Predictive Maintenance Dataset with Logistic Regression, Random Forest, and XGBoost; it writes evaluation plots/metrics and SHAP explanations from actual downloaded data.
2. **Pharmaceutical quality control:** trains a custom CNN and pretrained MobileNetV2 on the official fixed PharmaVision splits, with evaluation reports and Grad-CAM image explanations.

The modules use separate data, models, and reports. No relationship between their samples or production batches is assumed.

## Project structure

```text
PharmaGuard-AI/
├── app/                         # Integrated Streamlit pages and shared helpers
├── data/
│   ├── smart_manufacturing/     # raw/ and processed/ machine data
│   └── quality_control/         # PharmaVision original splits
├── models/
│   ├── smart_manufacturing/     # saved machine pipelines and metadata
│   └── quality_control/         # trained tablet model checkpoints
├── reports/
│   ├── smart_manufacturing/     # actual saved metrics, plots, SHAP
│   └── quality_control/         # actual tablet metrics, plots, Grad-CAM
├── src/
│   ├── smart_manufacturing/     # preprocessing, models, evaluation, explainability
│   └── quality_control/         # preprocessing, models, evaluation, explainability
├── AGENTS.md                    # project-specific development guidance
└── requirements.txt             # pinned dashboard and model runtime dependencies
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

### Tablet Quality Control training

From the repository root, run:

```powershell
python -m src.quality_control.train
```

The command downloads PharmaVision from Zenodo record 22773411 into `data/quality_control/raw/`, verifies the archive checksum, extracts the original Train/Val/Test splits, and checks the published 190/30/100 images per class. It trains a custom CNN and an ImageNet-pretrained MobileNetV2 with 224x224 inputs. Only training images are augmented; validation and test images are resized and normalized without augmentation. The best validation-loss checkpoint is saved for each model. Reports, metrics, plots, and an example Grad-CAM overlay are written to `reports/quality_control/`.

PyTorch uses CUDA when available; training retries on CPU if CUDA training fails. MobileNetV2's pretrained weights must be downloadable on the first run. If no supported RAR extractor is available, install bsdtar or 7-Zip before running.

Run both modules' tests after training with:

```powershell
python -m unittest discover -s tests -v
```

### Dashboard

After installing the dependencies and ensuring the saved model checkpoints are available, start the integrated dashboard from the repository root:

```powershell
python -m streamlit run app/app.py
```

The sidebar opens five views: Home Dashboard, Smart Manufacturing, Tablet Quality Control, Explainable AI, and Model Performance. The machine page accepts the six AI4I input features and uses a selected saved pipeline. The tablet page accepts a JPEG, PNG, WebP, BMP, or TIFF image (up to 10 MB and 20 megapixels) and uses the saved recommended MobileNetV2 checkpoint. SHAP explanations use the saved training partition as their background; tablet image explanations use Grad-CAM. Model metrics and plots are read from `reports/` and are not recalculated by the dashboard.

The dashboard does not train models or download datasets. If a model or report is missing, it displays an error indicating which artifact is unavailable. For fresh environments, follow the module setup/training commands above to create these local artifacts before launching.

### Streamlit Community Cloud deployment

The app entry point is `app/app.py`. The root `requirements.txt` pins the versions used by the saved model artifacts. Select **Python 3.12** in Community Cloud's app advanced settings. The dependency file selects CPU-only PyTorch wheels on Linux and keeps the CUDA 13.0 PyTorch wheels for Windows development. Community Cloud does not need a GPU for inference.

Create the Community Cloud app using this repository, the `main` branch, and `app/app.py` as the entry point. The app reads models and reports already stored in the repository. It does not extract or load the PharmaVision archive during startup, and inference does not download pretrained weights or datasets.

### Current scope

`predict_machine()` and `predict_tablet()` load their own module-specific saved models after training. Smart Manufacturing classifies the dataset's recorded `Machine failure` label; it is not a remaining-useful-life model. Tablet predictions and Grad-CAM describe visible image patterns only. They do not verify chemical composition, medicine identity, dosage, pharmaceutical quality, or patient safety. The dashboard is a research prototype and its predictions should not replace validated manufacturing controls or qualified review.

## Prediction interfaces

- `src.smart_manufacturing.predict_machine(input_data)` loads the recommended saved Smart Manufacturing pipeline and returns a failure class, probability, and validation-selected threshold. Run the training command above first.
- `src.quality_control.predict_tablet(image)` returns a Normal/Defective class, confidence, and Grad-CAM image after the tablet training command creates its checkpoints.

## Dashboard limitations

- The dashboard expects the saved model files and evaluation reports generated by each module's training pipeline to exist locally.
- Smart Manufacturing predictions estimate the dataset's machine-failure class from user-entered sensor values; a non-failure result is not a guarantee.
- Tablet classification is based on visible image patterns and may be affected by lighting, camera, tablet type, and image quality.
