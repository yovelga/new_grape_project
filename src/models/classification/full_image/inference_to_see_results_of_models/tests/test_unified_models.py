"""
Test unified model loading and adapters.

Tests both sklearn and PyTorch model loading and inference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from app.utils.logging import logger
from app.models.loader import load_model, get_model_info, find_scaler_file
from app.models.adapters import create_adapter, SklearnAdapter


def test_sklearn_adapter():
    """Test sklearn adapter with mock model."""
    logger.info("=" * 60)
    logger.info("Test: Sklearn Adapter")
    logger.info("=" * 60)

    # Create mock sklearn model
    class MockSklearnModel:
        def __init__(self):
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = 10

        def predict(self, X):
            return np.random.randint(0, 2, size=X.shape[0])

        def predict_proba(self, X):
            # Return random probabilities
            proba = np.random.rand(X.shape[0], 2)
            # Normalize to sum to 1
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba

    # Create adapter
    model = MockSklearnModel()
    adapter = SklearnAdapter(model, name="test_model")

    logger.info(f"Adapter created: {adapter}")
    logger.info(f"  n_classes: {adapter.n_classes}")
    logger.info(f"  is_binary: {adapter.is_binary}")
    logger.info(f"  classes: {adapter.classes_}")

    # Test prediction
    X_test = np.random.randn(5, 10)
    proba = adapter.predict_proba(X_test)

    logger.info(f"\nPrediction test:")
    logger.info(f"  Input shape: {X_test.shape}")
    logger.info(f"  Output shape: {proba.shape}")
    logger.info(f"  Output range: [{proba.min():.4f}, {proba.max():.4f}]")
    logger.info(f"  Row sums: {np.sum(proba, axis=1)}")

    # Validate
    assert proba.shape == (5, 2), "Shape mismatch"
    assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities don't sum to 1"
    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of range"

    logger.info("\n✓ Sklearn adapter test PASSED")
    return True


def test_sklearn_pipeline():
    """Test sklearn adapter with pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Sklearn Pipeline")
    logger.info("=" * 60)

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.warning("sklearn not installed, skipping pipeline test")
        return True

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    # Fit with dummy data
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(0, 2, size=20)
    pipeline.fit(X_train, y_train)

    # Create adapter
    adapter = SklearnAdapter(pipeline, name="pipeline_model")

    logger.info(f"Pipeline adapter created: {adapter}")
    logger.info(f"  n_classes: {adapter.n_classes}")

    # Test prediction
    X_test = np.random.randn(3, 5)
    proba = adapter.predict_proba(X_test)

    logger.info(f"\nPrediction test:")
    logger.info(f"  Input shape: {X_test.shape}")
    logger.info(f"  Output shape: {proba.shape}")

    # Validate
    assert proba.shape == (3, 2), "Shape mismatch"
    assert np.allclose(np.sum(proba, axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"

    logger.info("\n✓ Pipeline test PASSED")
    return True


def test_model_without_predict_proba():
    """Test adapter with model that doesn't have predict_proba."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Model Without predict_proba")
    logger.info("=" * 60)

    class MockModelNoProbA:
        def __init__(self):
            self.classes_ = np.array(['A', 'B', 'C'])

        def predict(self, X):
            return np.random.choice(['A', 'B', 'C'], size=X.shape[0])

    model = MockModelNoProbA()
    adapter = SklearnAdapter(model, name="no_proba_model")

    logger.info(f"Adapter created: {adapter}")
    logger.info(f"  n_classes: {adapter.n_classes}")

    # Test prediction
    X_test = np.random.randn(5, 10)
    proba = adapter.predict_proba(X_test)

    logger.info(f"\nPrediction test (fallback to one-hot):")
    logger.info(f"  Output shape: {proba.shape}")
    logger.info(f"  Sample probabilities:\n{proba[:2]}")

    # Should be one-hot encoded
    assert proba.shape == (5, 3), "Shape mismatch"
    assert np.all(np.sum(proba, axis=1) == 1.0), "Not one-hot encoded"

    logger.info("\n✓ Model without predict_proba test PASSED")
    return True


def test_create_adapter():
    """Test automatic adapter creation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Automatic Adapter Creation")
    logger.info("=" * 60)

    # Create mock sklearn model
    class MockModel:
        def __init__(self):
            self.classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(X.shape[0])

        def predict_proba(self, X):
            return np.column_stack([np.ones(X.shape[0]) * 0.7, np.ones(X.shape[0]) * 0.3])

    model = MockModel()

    # Auto-detect sklearn
    adapter = create_adapter(model, model_type="auto", name="auto_sklearn")

    logger.info(f"Auto-created adapter: {adapter}")
    assert isinstance(adapter, SklearnAdapter), "Should create SklearnAdapter"

    logger.info("\n✓ Automatic adapter creation test PASSED")
    return True


def test_validation_checks():
    """Test input/output validation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Validation Checks")
    logger.info("=" * 60)

    class MockModel:
        def __init__(self):
            self.classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.column_stack([np.ones(X.shape[0]) * 0.6, np.ones(X.shape[0]) * 0.4])

    adapter = SklearnAdapter(MockModel())

    # Test wrong input type
    try:
        adapter.predict_proba([1, 2, 3])
        logger.error("Should have raised TypeError")
        return False
    except TypeError as e:
        logger.info(f"✓ Correct error for wrong input type: {e}")

    # Test wrong input shape
    try:
        adapter.predict_proba(np.array([1, 2, 3]))  # 1D array
        logger.error("Should have raised ValueError")
        return False
    except ValueError as e:
        logger.info(f"✓ Correct error for wrong input shape: {e}")

    # Test valid input
    X = np.random.randn(3, 5)
    proba = adapter.predict_proba(X)
    logger.info(f"✓ Valid input processed correctly: {proba.shape}")

    logger.info("\n✓ Validation checks test PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Unified Model Loading & Adapters Tests")
    logger.info("=" * 60)

    tests = [
        ("Sklearn Adapter", test_sklearn_adapter),
        ("Sklearn Pipeline", test_sklearn_pipeline),
        ("Model Without predict_proba", test_model_without_predict_proba),
        ("Automatic Adapter Creation", test_create_adapter),
        ("Validation Checks", test_validation_checks),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.error(f"❌ {failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
