# HSI Crack Detection - Working Setup

## ✅ Status: WORKING

The project is now fully configured and running!

## Quick Start

### Launch UI
Double-click: `start_ui.bat`

Or run manually:
```bash
cd C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model
C:\Users\yovel\Desktop\Grape_Project\.venv\Scripts\python.exe late_detection_ui.py
```

## Configuration

- **Project Root**: `C:\Users\yovel\Desktop\Grape_Project`
- **Working Directory**: `src\models\classification\full_image\infernce_with_new_model`
- **Default Model**: OLD LDA [1=CRACK, 0=regular] (LDA_Balanced.pkl)
- **Dataset**: `src\preprocessing\hole_image\late_detection\late_detection_dataset.csv`
- **Results**: `.\Results\`

## What Was Fixed

1. ✅ Updated model paths - OLD LDA as default (NEW models are empty)
2. ✅ Fixed dataset CSV path
3. ✅ Fixed Results folder to use current directory
4. ✅ Verified all critical paths exist
5. ✅ UI launches successfully
6. ✅ Created start_ui.bat for easy access

## Available Scripts

- `late_detection_ui.py` - Interactive UI (PyQt5)
- `run_late_detection_inference.py` - Batch inference
- `run_blob_patch_grid_search.py` - Grid search
- `run_hyperparameter_tuning.py` - Hyperparameter tuning
- `optuna_blob_patch_opt.py` - Optuna optimization

## Model Status

- ✅ **OLD LDA [1=CRACK, 0=regular]** - Working (default)
- ❌ **NEW LDA Multi-class** - Empty file, needs retraining
- ❌ **LDA 2-class** - Empty file, needs retraining

To use the NEW models, they need to be retrained using the training scripts in:
`src\models\classification\full_image\classification_by_pixel\Train\LDA\`

## Directory Structure

```
infernce_with_new_model/
├── start_ui.bat                      # Quick start script
├── late_detection_ui.py             # Main UI
├── late_detection_core.py           # Core functions
├── run_late_detection_inference.py  # Batch processing
├── grid_search_blob_patch.py        # Grid search
├── optuna_blob_patch_opt.py         # Optuna optimization
├── Results/                         # Output directory
└── prob_maps_late_detection/        # Probability maps
```

## All Set! 🎉

The project is working and ready to use!
