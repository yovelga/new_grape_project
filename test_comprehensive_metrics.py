"""
Quick Test: Verify Comprehensive Metrics Are Calculated

This script tests that evaluate_single_combination returns all 7 metrics:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. F2 Score (NEW)
6. ROC AUC
7. PR AUC (NEW)
"""

import sys
import os
import numpy as np

# Add project path
project_path = r"C:\Users\yovel\Desktop\Grape_Project"
sys.path.insert(0, project_path)

# Import the function
from src.models.classification.full_image.infernce_with_pathes_with_F1_XGBOOST.grid_search_blob_patch import (
    evaluate_single_combination
)

print("=" * 80)
print("TESTING COMPREHENSIVE METRICS CALCULATION")
print("=" * 80)
print()

# Create synthetic test data
print("Creating synthetic probability maps and labels...")
np.random.seed(42)

prob_maps_dict = {}
labels_dict = {}

# Create 10 samples: 5 positive (cracked), 5 negative (healthy)
for i in range(10):
    sample_id = f"sample_{i}"

    # Create random probability map (100x100)
    if i < 5:
        # Positive samples (cracked) - higher probabilities
        prob_map = np.random.uniform(0.6, 0.95, size=(100, 100))
        labels_dict[sample_id] = 1  # Cracked
    else:
        # Negative samples (healthy) - lower probabilities
        prob_map = np.random.uniform(0.05, 0.4, size=(100, 100))
        labels_dict[sample_id] = 0  # Healthy

    prob_maps_dict[sample_id] = prob_map

print(f"✓ Created {len(prob_maps_dict)} samples")
print(f"  - Positive (cracked): {sum(labels_dict.values())} samples")
print(f"  - Negative (healthy): {len(labels_dict) - sum(labels_dict.values())} samples")
print()

# Test evaluate_single_combination with simple parameters
print("Running evaluate_single_combination...")
print()

metrics = evaluate_single_combination(
    prob_maps_dict=prob_maps_dict,
    labels_dict=labels_dict,
    prob_thr=0.5,
    min_blob_size=10,
    circularity_min=None,
    circularity_max=None,
    aspect_ratio_min=None,
    aspect_ratio_limit=None,
    solidity_min=None,
    solidity_max=None,
    patch_size=16,
    patch_pixel_ratio=0.1,
    global_threshold=0.05,
    morph_size=0
)

print("=" * 80)
print("RESULTS: ALL METRICS")
print("=" * 80)
print()

# Check all expected metrics are present
expected_metrics = [
    'accuracy', 'precision', 'recall', 'f1_score', 'f2_score', 'roc_auc', 'pr_auc'
]

print("Checking for all expected metrics...")
all_present = True
for metric in expected_metrics:
    if metric in metrics:
        value = metrics[metric]
        status = "✓" if not np.isnan(value) else "⚠ (NaN)"
        print(f"  {status} {metric:15s}: {value:.4f}")
    else:
        print(f"  ✗ {metric:15s}: MISSING!")
        all_present = False

print()
print("=" * 80)

if all_present:
    print("✅ SUCCESS: All 7 metrics are present!")
    print()
    print("Detailed Results:")
    print(f"  • Accuracy:   {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision:  {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  • Recall:     {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"  • F1 Score:   {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    print(f"  • F2 Score:   {metrics['f2_score']:.4f}  ({metrics['f2_score']*100:.2f}%) ← NEW!")
    print(f"  • ROC AUC:    {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    print(f"  • PR AUC:     {metrics['pr_auc']:.4f}  ({metrics['pr_auc']*100:.2f}%) ← NEW!")
    print()
    print("Note: F2 Score emphasizes recall (beta=2)")
    print("Note: PR AUC is better for imbalanced datasets than ROC AUC")
else:
    print("❌ FAILURE: Some metrics are missing!")

print("=" * 80)

