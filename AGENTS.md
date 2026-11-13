# PharmaGuard AI project guidance

## Project boundaries
- Keep smart manufacturing and pharmaceutical quality control as independent modules.
- Do not assume machine data and tablet images belong to the same production batches.
- Do not create synthetic datasets, fabricated metrics, or placeholder model artifacts presented as real results.
- Do not download datasets or train models unless explicitly requested.

## Development
- Use Python and paths relative to the repository root.
- Keep code modular and beginner-friendly; explain non-obvious decisions.
- Keep reusable module code under `src/` and the Streamlit interface under `app/`.
- Keep saved models under `models/` and evaluation outputs under `reports/`.
- The prediction functions should not return made-up predictions. Raise `NotImplementedError` until a real model and preprocessing pipeline are available.
