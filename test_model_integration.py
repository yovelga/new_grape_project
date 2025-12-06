"""
Test the LDA Best F1 Weighted model with the updated late_detection_core
"""
import sys
import os
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

# Import the core functions
from src.models.classification.full_image.infernce_with_new_model_with_sam2_reduce_bands.late_detection_core import (
    load_model_and_scaler, _classes_any, _get_model_expected_features, _predict_proba_any
)

print("="*70)
print("Testing LDA Best F1 Weighted Model Integration")
print("="*70)

model_path = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_sam2_reduce_bands\lda_model_best_f1_weighted.joblib'

print(f"\n1. Loading model from: {model_path}")
print("-"*70)

try:
    # Test load_model_and_scaler
    lda, scaler, pos_idx, classes = load_model_and_scaler(model_path, scaler_path=None)
    print(f"✓ Model loaded successfully!")
    print(f"  Model type: {type(lda)}")
    print(f"  Is dictionary: {isinstance(lda, dict)}")

    if isinstance(lda, dict):
        print(f"  Dictionary keys: {list(lda.keys())}")
        if 'selected_features' in lda:
            print(f"  Number of selected features: {len(lda['selected_features'])}")
            print(f"  Selected features: {lda['selected_features']}")

    print(f"  Scaler: {scaler}")
    print(f"  Positive class index: {pos_idx}")
    print(f"  Classes: {classes}")

    # Test _classes_any
    print(f"\n2. Testing _classes_any function")
    print("-"*70)
    test_classes = _classes_any(lda)
    print(f"✓ Classes extracted: {test_classes}")

    # Test _get_model_expected_features
    print(f"\n3. Testing _get_model_expected_features function")
    print("-"*70)
    expected_features = _get_model_expected_features(lda)
    print(f"✓ Expected input features: {expected_features}")

    # Test prediction with synthetic data
    print(f"\n4. Testing prediction with synthetic data")
    print("-"*70)

    # Create synthetic data with the expected number of features
    n_samples = 10
    X_test = np.random.rand(n_samples, expected_features)
    print(f"  Created test data: shape={X_test.shape}")

    # Handle feature selection if model is a dictionary
    if isinstance(lda, dict) and 'sfs' in lda:
        print(f"  Applying feature selection...")
        X_test_selected = lda['sfs'].transform(X_test)
        print(f"  After feature selection: shape={X_test_selected.shape}")
        X_test = X_test_selected

    # Test prediction
    probs = _predict_proba_any(lda, X_test)
    print(f"✓ Prediction successful!")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  First prediction: {probs[0]}")

    # Check if CRACK class is found
    crack_found = False
    for i, cls in enumerate(classes):
        if cls in ['CRACK', 'crack', 'Crack']:
            crack_found = True
            print(f"\n5. CRACK class found at index {i}")
            print("-"*70)
            print(f"  Class name: {cls}")
            print(f"  Position index: {pos_idx}")
            break

    if not crack_found:
        print(f"\n5. Warning: CRACK class not found")
        print(f"  Available classes: {classes}")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe model is ready to use in the late detection UI.")
    print("It will automatically handle feature selection during inference.")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

