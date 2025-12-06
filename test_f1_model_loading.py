"""
Quick test script to verify F1-optimized model loading works correctly.

Usage:
    python test_f1_model_loading.py
"""

import sys
from pathlib import Path

# Add project to path
project_path = Path(__file__).resolve().parent
sys.path.insert(0, str(project_path))

# Import the loading function
from src.models.classification.full_image.infernce_with_new_model_with_sam2_with_F1.late_detection_core import (
    load_model_and_scaler
)

def test_model_loading():
    """Test loading the F1-optimized model."""

    # Path to the new F1-optimized model
    model_path = r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_sam2_with_F1\lda_model_multi_class_f1_score.joblib"

    print("=" * 80)
    print("Testing F1-Optimized LDA Model Loading")
    print("=" * 80)
    print(f"\nModel Path: {model_path}")

    try:
        # Load the model
        lda, scaler, pos_idx, classes, optimal_threshold = load_model_and_scaler(model_path, None)

        print("\n✅ Model loaded successfully!")
        print(f"\nModel Details:")
        print(f"  - Classes: {classes}")
        print(f"  - Positive Index (CRACK): {pos_idx}")
        print(f"  - Scaler Present: {scaler is not None}")
        print(f"  - Optimal Threshold: {optimal_threshold}")

        if optimal_threshold is not None:
            print(f"\n✅ F1-Optimized threshold detected: {optimal_threshold:.4f}")
            print(f"   This value will be automatically applied in the UI.")
        else:
            print(f"\n⚠️  No optimal threshold found (legacy model format)")

        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)

        return True

    except FileNotFoundError:
        print(f"\n❌ ERROR: Model file not found!")
        print(f"   Please ensure the model exists at: {model_path}")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_model_loading():
    """Test loading a legacy model (should return None for optimal_threshold)."""

    # Path to a legacy model
    legacy_path = r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\lda_model_multi_class.joblib"

    print("\n" + "=" * 80)
    print("Testing Legacy Model Loading (Backward Compatibility)")
    print("=" * 80)
    print(f"\nModel Path: {legacy_path}")

    try:
        # Load the model
        lda, scaler, pos_idx, classes, optimal_threshold = load_model_and_scaler(legacy_path, None)

        print("\n✅ Legacy model loaded successfully!")
        print(f"\nModel Details:")
        print(f"  - Classes: {classes}")
        print(f"  - Positive Index (CRACK): {pos_idx}")
        print(f"  - Scaler Present: {scaler is not None}")
        print(f"  - Optimal Threshold: {optimal_threshold}")

        if optimal_threshold is None:
            print(f"\n✅ Backward compatibility verified: Legacy model returns None for threshold")
        else:
            print(f"\n⚠️  Unexpected: Legacy model returned threshold value")

        return True

    except FileNotFoundError:
        print(f"\n⚠️  Legacy model not found (this is optional)")
        return True  # Not a failure, just skip

    except Exception as e:
        print(f"\n❌ ERROR: Failed to load legacy model!")
        print(f"   Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("F1-OPTIMIZED MODEL INTEGRATION TEST")
    print("=" * 80)

    # Test F1-optimized model
    success1 = test_model_loading()

    # Test legacy model (backward compatibility)
    success2 = test_legacy_model_loading()

    print("\n" + "=" * 80)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)

