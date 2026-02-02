"""
Quick test script for the Feature Selection Pipelines.

This script runs a dry-run of the feature selection pipelines with a reduced
sample size for quick validation before running on the full dataset.

Available modes:
    --mode legacy     : Original SHAP + RFECV pipeline (feature_selection_pipeline.py)
    --mode multiclass : Thesis multiclass-only pipeline (feature_selection_multiclass.py)

Usage:
    python quick_test_feature_selection.py                  # Default: multiclass mode
    python quick_test_feature_selection.py --mode multiclass
    python quick_test_feature_selection.py --mode legacy
    python quick_test_feature_selection.py --max_samples 1000
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_legacy_test(max_samples: int = 2000):
    """Run the legacy SHAP + RFECV pipeline in dry-run mode."""
    from feature_selection_pipeline import (
        run_feature_selection_pipeline,
        CSV_PATH_MULTICLASS,
        OUTPUT_DIR,
        TARGET_FEATURES,
    )
    
    print("=" * 60)
    print("QUICK TEST: Legacy Feature Selection Pipeline (SHAP + RFECV)")
    print("=" * 60)
    print(f"Running with max_samples={max_samples} for quick validation")
    print("=" * 60)
    
    # Run with reduced samples for quick test
    test_output_dir = OUTPUT_DIR.parent / f"test_legacy_{datetime.now().strftime('%H-%M-%S')}"
    
    top_10_df, summary = run_feature_selection_pipeline(
        csv_path=CSV_PATH_MULTICLASS,
        output_dir=test_output_dir,
        top_k_shap=20,  # Reduced for quick test
        target_features=TARGET_FEATURES,
        max_samples=max_samples,
    )
    
    print("\n" + "=" * 60)
    print("LEGACY TEST COMPLETE")
    print("=" * 60)
    print("\nIf everything looks good, run the full pipeline with:")
    print("    python feature_selection_pipeline.py")
    
    return top_10_df, summary


def run_multiclass_test(max_samples: int = 2000, k_grid: str = "1,2,5,10,20"):
    """Run the multiclass-only feature selection pipeline in dry-run mode."""
    from feature_selection_multiclass import run_feature_selection, CSV_PATH
    
    print("=" * 60)
    print("QUICK TEST: Multiclass Feature Selection (CRACK PR-AUC)")
    print("=" * 60)
    print(f"Running with max_samples={max_samples}, k_grid={k_grid}")
    print("NOTE: LOGO on grape images, SNV applied, optimized for CRACK PR-AUC")
    print("=" * 60)
    
    # Parse K grid
    k_list = [int(k.strip()) for k in k_grid.split(',')]
    
    result = run_feature_selection(
        csv_path=CSV_PATH,
        k_grid=k_list,
        max_samples=max_samples,
        use_gpu=True,
    )
    
    print("\n" + "=" * 60)
    print("MULTICLASS TEST COMPLETE")
    print("=" * 60)
    print("\nIf everything looks good, run the full pipeline with:")
    print("    python feature_selection_multiclass.py")
    print("\nOr with custom K grid:")
    print("    python feature_selection_multiclass.py --k_grid 1,2,5,10,20,40,60,80,100")
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quick test for feature selection pipelines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', type=str, default='multiclass',
        choices=['legacy', 'multiclass'],
        help='Which pipeline to test: legacy (SHAP+RFECV) or multiclass (thesis)'
    )
    parser.add_argument(
        '--max_samples', type=int, default=2000,
        help='Maximum samples for dry-run'
    )
    parser.add_argument(
        '--k_grid', type=str, default='1,2,5,10,20',
        help='Comma-separated K values for multiclass test'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'legacy':
        run_legacy_test(max_samples=args.max_samples)
    else:
        run_multiclass_test(max_samples=args.max_samples, k_grid=args.k_grid)
