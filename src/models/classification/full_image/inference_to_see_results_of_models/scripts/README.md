# Scripts

Utility scripts for the inference application.

## smoke_test.py

Quick validation that all pipeline components work together.

### Usage

```bash
# Run with default settings (uses .env configuration)
python scripts/smoke_test.py

# Run with custom CSV paths
python scripts/smoke_test.py --trainval path/to/trainval.csv --test path/to/test.csv

# Quiet mode (less output)
python scripts/smoke_test.py --quiet
```

### What it checks

1. **Settings** - Loads and validates configuration from .env
2. **Environment completeness** - Ensures .env.example has all required variables
3. **Dataset loading** - Tests CSV loading and train/val splitting
4. **Model loading** - Tests model file loading and adapter creation
5. **Inference pipeline** - Runs inference on synthetic data
6. **Metrics** - Tests F1, F2, accuracy computation
7. **Logging** - Tests logging utilities

### Exit codes

- `0` = All tests passed
- `1` = One or more tests failed

### Troubleshooting

If tests fail:

1. Check that `.env` file exists (copy from `.env.example`)
2. Ensure `MODELS_DIR` contains at least one `.joblib` or `.pkl` model
3. Check that CSV paths are correct if using `--trainval` and `--test` flags
4. Review the error messages for specific failures
