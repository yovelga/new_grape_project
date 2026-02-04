"""
Integration test with real CSV data and model.

Demonstrates complete workflow:
1. Load dataset from CSV
2. Split train/val
3. Load sklearn model
4. Create unified adapter
5. Run predictions

Uses actual data and model files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from app.utils.logging import logger
from app.config.settings import settings
from app.data.dataset import load_dataset_csv, split_train_val, get_class_distribution
from app.models.loader import load_model, get_model_info, find_scaler_file, load_scaler
from app.models.adapters import create_adapter, SklearnAdapter


def test_real_data_workflow():
    """Test complete workflow with real data."""
    logger.info("=" * 60)
    logger.info("Integration Test: Real Data & Model")
    logger.info("=" * 60)

    # Paths to test files (copied locally)
    csv_path = "./data/test_dataset.csv"
    model_path = "./models/test_model.pkl"

    # Check files exist
    if not Path(csv_path).exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Please copy the CSV file to data/test_dataset.csv")
        return False

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please copy the model file to models/test_model.pkl")
        return False

    # Step 1: Load dataset
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Load Dataset from CSV")
    logger.info("=" * 60)

    try:
        df = load_dataset_csv(csv_path)
        logger.info(f"✓ Loaded {len(df)} samples")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Class distribution: {get_class_distribution(df)}")
    except Exception as e:
        logger.error(f"✗ Failed to load CSV: {e}")
        return False

    # Step 2: Split train/val
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Split Train/Val")
    logger.info("=" * 60)

    try:
        train_df, val_df = split_train_val(df, val_size=0.20, random_state=42)
        logger.info(f"✓ Split successful")
        logger.info(f"  Train: {len(train_df)} samples - {get_class_distribution(train_df)}")
        logger.info(f"  Val:   {len(val_df)} samples - {get_class_distribution(val_df)}")
    except Exception as e:
        logger.error(f"✗ Failed to split: {e}")
        return False

    # Step 3: Load model
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Load Model")
    logger.info("=" * 60)

    try:
        model = load_model(model_path)
        info = get_model_info(model)
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Type: {info.get('type', 'unknown')}")
        if 'n_classes' in info:
            logger.info(f"  Classes: {info['n_classes']}")
        if 'classes' in info:
            logger.info(f"  Class labels: {info['classes']}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False

    # Step 4: Check for scaler
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Check for Scaler")
    logger.info("=" * 60)

    scaler_path = find_scaler_file(model_path)
    scaler = None
    if scaler_path:
        try:
            scaler = load_scaler(scaler_path)
            logger.info(f"✓ Scaler loaded: {type(scaler).__name__}")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
    else:
        logger.info("  No scaler found (this is OK)")

    # Step 5: Create adapter
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Create Model Adapter")
    logger.info("=" * 60)

    try:
        adapter = create_adapter(model, model_type="auto", name="XGBoost_Test")
        logger.info(f"✓ Adapter created: {adapter}")
        logger.info(f"  n_classes: {adapter.n_classes}")
        logger.info(f"  is_binary: {adapter.is_binary}")
        if adapter.classes_:
            logger.info(f"  classes: {adapter.classes_}")
    except Exception as e:
        logger.error(f"✗ Failed to create adapter: {e}")
        return False

    # Step 6: Test prediction with synthetic data
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Test Prediction (Synthetic Data)")
    logger.info("=" * 60)

    try:
        # Get expected number of features from model
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            # Try to infer from first prediction
            n_features = 50  # Default guess

        logger.info(f"  Expected features: {n_features}")

        # Create synthetic test data
        X_test = np.random.randn(5, n_features)

        # Apply scaler if available
        if scaler:
            X_test = scaler.transform(X_test)
            logger.info(f"  ✓ Applied scaler")

        # Run prediction
        proba = adapter.predict_proba(X_test)

        logger.info(f"✓ Prediction successful")
        logger.info(f"  Input shape: {X_test.shape}")
        logger.info(f"  Output shape: {proba.shape}")
        logger.info(f"  Probability range: [{proba.min():.4f}, {proba.max():.4f}]")
        logger.info(f"  Sample predictions:")
        for i in range(min(3, len(proba))):
            logger.info(f"    Sample {i+1}: {proba[i]}")

        # Validate
        assert proba.shape[0] == 5, "Wrong number of predictions"
        assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of range"
        assert np.allclose(np.sum(proba, axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"

        logger.info(f"  ✓ All validation checks passed")

    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 7: Summary
    logger.info("\n" + "=" * 60)
    logger.info("Integration Test Summary")
    logger.info("=" * 60)
    logger.info("✓ Dataset loading: PASSED")
    logger.info("✓ Train/val split: PASSED")
    logger.info("✓ Model loading: PASSED")
    logger.info("✓ Adapter creation: PASSED")
    logger.info("✓ Prediction: PASSED")
    logger.info("\n🎉 Complete integration test PASSED!")

    return True


def main():
    """Run integration test."""
    logger.info("=" * 60)
    logger.info("Real Data Integration Test")
    logger.info("=" * 60)
    logger.info("\nThis test demonstrates the complete workflow:")
    logger.info("  1. Load CSV dataset")
    logger.info("  2. Split train/validation")
    logger.info("  3. Load sklearn/XGBoost model")
    logger.info("  4. Create unified adapter")
    logger.info("  5. Run predictions")
    logger.info("\nUsing real files:")
    logger.info("  CSV: data/test_dataset.csv")
    logger.info("  Model: models/test_model.pkl")

    success = test_real_data_workflow()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ Integration test PASSED")
        return 0
    else:
        logger.error("❌ Integration test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
