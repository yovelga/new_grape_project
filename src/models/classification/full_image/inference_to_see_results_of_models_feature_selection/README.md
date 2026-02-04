# Binary Classification Inference System

## Overview
Self-contained inference application for binary/multi-class classification of hyperspectral and RGB images.

**Key Features:**
- ✅ Hyperspectral (ENVI) and RGB image support
- ✅ Sklearn model support (LDA, XGBoost, Random Forest, SVM)
- ✅ PyTorch CNN support (EfficientNet, custom architectures)
- ✅ Per-pixel probability mapping
- ✅ Grid-based patch analysis
- ✅ SNV normalization (local implementation)
- ✅ Wavelength filtering
- ✅ Optuna hyperparameter tuning
- ✅ Advanced visualizations
- ✅ **100% self-contained - no external imports**

## Quick Start

### 1. Setup Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (or use defaults)
```

### 2. Run Smoke Test (Recommended First)
```bash
# Verify all components work together
python scripts/smoke_test.py

# With custom CSVs
python scripts/smoke_test.py --trainval path/to/trainval.csv --test path/to/test.csv
```

### 3. Run the GUI Application
```bash
# Requires PyQt5: pip install PyQt5
python binary_class_inference_ui.py
```

### 4. Programmatic Usage
```python
from app.config.settings import settings
from app.data.dataset import load_and_prepare_splits
from app.tuning import run_full_tuning_pipeline

# Load datasets with seed for reproducibility
train_df, val_df, test_df = load_and_prepare_splits(
    trainval_csv_path="path/to/trainval.csv",
    test_csv_path="path/to/test.csv",
    val_size=0.30,
    random_state=settings.random_seed  # Reproducibility!
)

# Run tuning pipeline
results = run_full_tuning_pipeline(
    train_df, val_df, test_df,
    prob_map_fn=your_prob_map_function,
    n_trials=50,
    seed=settings.random_seed,
    metric="f2"
)
```

## Project Structure
```
inference_to_see_results_of_models/
├── binary_class_inference_ui.py      # Main GUI entrypoint (SINGLE UI)
├── .env.example                      # Example environment configuration
├── .env                              # Your config (copy from .env.example)
├── SCOPE.md                          # Project boundaries and guidelines
├── scripts/
│   ├── smoke_test.py                 # Quick validation script
│   └── README.md                     # Script documentation
├── tests/                            # Test suite
│   ├── test_functional.py            # Functional tests
│   ├── test_settings.py              # Settings tests
│   └── validate_structure.py         # Structure validation
├── archive_legacy/                   # Historical reference (not used)
├── logs/                             # Application logs (auto-created)
├── results/                          # Output results (auto-created)
├── models/                           # Place trained models here
└── app/
    ├── config/
    │   ├── settings.py               # Configuration from .env
    │   └── types.py                  # Dataclass definitions
    ├── data/
    │   └── dataset.py                # CSV loading, train/val/test splits
    ├── io/
    │   ├── envi.py                   # ENVI hyperspectral reader
    │   └── rgb.py                    # RGB image reader
    ├── preprocess/
    │   └── spectral.py               # SNV, wavelength filtering
    ├── models/
    │   ├── loader_new.py             # Model loading utilities
    │   └── adapters_new.py           # Model adapters
    ├── inference/
    │   └── prob_map.py               # Probability map generation
    ├── postprocess/
    │   ├── pipeline.py               # Postprocessing pipeline
    │   ├── morphology.py             # Morphological operations
    │   └── blob_filters.py           # Connected component filtering
    ├── tuning/
    │   └── optuna_runner.py          # Optuna hyperparameter tuning
    ├── metrics/
    │   └── classification.py         # F1, F2, accuracy metrics
    └── utils/
        ├── logging.py                # Console + file logging
        └── results_io.py             # JSON/CSV I/O utilities
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `./models` | Directory containing trained models |
| `RESULTS_DIR` | `./results` | Output directory for results |
| `LOG_DIR` | `./logs` | Directory for log files |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |
| `VAL_SPLIT_SIZE` | `0.30` | Validation split fraction |
| `DEVICE` | `cpu` | PyTorch device (cpu, cuda, mps) |
| `LOG_LEVEL` | `INFO` | Logging level |

See `.env.example` for all available options.

## Reproducibility

All randomness is controlled by seed:
- **Dataset splitting**: Uses `RANDOM_SEED` from settings
- **Optuna sampling**: Uses seed parameter in `run_optuna()`

```python
# Example: reproducible workflow
from app.config.settings import settings
train_df, val_df = split_train_val(df, random_state=settings.random_seed)
best_params, _ = run_optuna(..., seed=settings.random_seed)
```

## Logging

Logs are written to both console and file:
- Console: Real-time feedback
- File: `RESULTS_DIR/logs/inference_YYYYMMDD_HHMMSS.log`

Log format:
```
2026-01-21 10:30:45 | inference            | INFO     | Message
```

## Testing

### Smoke Test
```bash
python scripts/smoke_test.py
```

### Module Tests
```bash
python -m app.utils.logging
python -m app.metrics.classification
python -m app.postprocess.pipeline
```

## Dependencies

```bash
# Core
pip install numpy pandas scikit-learn joblib

# Optional
pip install PyQt5          # For GUI
pip install optuna          # For hyperparameter tuning
pip install torch           # For PyTorch models
pip install opencv-python   # For morphological ops (or scipy)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No models found" | Place `.joblib` files in `MODELS_DIR` |
| "Settings errors" | Copy `.env.example` to `.env` |
| "PyQt5 not available" | `pip install PyQt5` or use console mode |

## License
Internal use only.
