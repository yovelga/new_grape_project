"""
Quick test script for the Feature Selection Pipeline.

This script runs a dry-run of the feature selection pipeline with a reduced
sample size for quick validation before running on the full dataset.

Usage:
    python quick_test_feature_selection.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_selection_pipeline import (
    run_feature_selection_pipeline,
    CSV_PATH_MULTICLASS,
    OUTPUT_DIR,
    TOP_K_SHAP,
    TARGET_FEATURES,
)


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK TEST: Feature Selection Pipeline")
    print("=" * 60)
    print("Running with max_samples=2000 for quick validation")
    print("=" * 60)
    
    # Run with reduced samples for quick test
    test_output_dir = OUTPUT_DIR.parent / f"feature_selection_TEST_{OUTPUT_DIR.name.split('_')[-1]}"
    
    top_10_df, summary = run_feature_selection_pipeline(
        csv_path=CSV_PATH_MULTICLASS,
        output_dir=test_output_dir,
        top_k_shap=20,  # Reduced for quick test
        target_features=TARGET_FEATURES,
        max_samples=2000,  # Reduced sample size for testing
    )
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nIf everything looks good, run the full pipeline with:")
    print("    python feature_selection_pipeline.py")
