"""
quick_test_ga_feature_selection.py

Quick test script for the Multiclass Feature Selection Pipeline (GA-style).
This is a convenience script at project root that calls the main pipeline.

Usage:
    python quick_test_ga_feature_selection.py
    python quick_test_ga_feature_selection.py --max_samples 1000
    python quick_test_ga_feature_selection.py --k_max 20
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src" / "models" / "classification" / "pixel_level" / "feature_selection"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run
from quick_test_feature_selection import run_multiclass_test, parse_args

if __name__ == "__main__":
    args = parse_args()
    run_multiclass_test(max_samples=args.max_samples, k_max=args.k_max)
