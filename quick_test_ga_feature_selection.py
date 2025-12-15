"""
Quick Test Script for GA Feature Selection
===========================================

This script tests the GA feature selection with a smaller configuration
for faster execution and validation before running the full optimization.

Usage:
    python quick_test_ga_feature_selection.py

Expected runtime: 2-5 minutes
"""

import os
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main GA module
from src.models.classification.full_image.Train.reduce_chanels import genetic_feature_selection_crack_f1 as ga_module

# ============================================================================
# TEST CONFIGURATION (Smaller, faster settings)
# ============================================================================

# Override GA configuration for quick test
ga_module.GA_CONFIG = {
    'population_size': 20,        # Reduced from 60
    'num_generations': 10,        # Reduced from 40
    'crossover_rate': 0.8,
    'mutation_rate': 0.10,
    'tournament_size': 3,         # Reduced from 4
    'elitism_count': 2,
    'min_features': 5,
    'max_features': 30,           # Reduced from 50
    'random_seed': 42
}

# Update results folder for test run
ga_module.RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\results\ga_feature_selection_TEST"

print("\n" + "="*80)
print(" QUICK TEST - GENETIC ALGORITHM FEATURE SELECTION ")
print("="*80)
print("\nTest Configuration:")
print(f"  Population Size: {ga_module.GA_CONFIG['population_size']}")
print(f"  Generations: {ga_module.GA_CONFIG['num_generations']}")
print(f"  Max Features: {ga_module.GA_CONFIG['max_features']}")
print(f"  Results: {ga_module.RESULTS_FOLDER}")
print("="*80 + "\n")

# Run the main script
if __name__ == "__main__":
    # Check if data file exists
    if not os.path.exists(ga_module.DATA_PATH):
        print(f"\n❌ ERROR: Data file not found at:")
        print(f"   {ga_module.DATA_PATH}")
        print(f"\nPlease update DATA_PATH in genetic_feature_selection_crack_f1.py")
        print(f"or create a symbolic link to your dataset.\n")
        sys.exit(1)

    try:
        ga_module.main()
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"Check results in: {ga_module.RESULTS_FOLDER}")
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

