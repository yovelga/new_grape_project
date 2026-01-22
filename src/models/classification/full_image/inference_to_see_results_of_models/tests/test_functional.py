"""
Functional test demonstrating the refactored inference system.

Tests all major components without requiring external data.
"""

import sys
from pathlib import Path
import numpy as np

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.logging import logger
from app.preprocess.spectral import snv_normalize, filter_wavelengths
from app.models.sklearn_models import SklearnModelWrapper
from app.inference.engine import HyperspectralInferenceEngine, GridAnalyzer
from app.visualization.overlays import (
    create_binary_overlay, create_grid_overlay, create_rgb_composite,
    normalize_to_uint8
)


def test_snv_normalization():
    """Test SNV normalization."""
    logger.info("Testing SNV normalization...")

    # Create synthetic spectral data
    spectra = np.random.randn(100, 50) * 10 + 50

    # Apply SNV
    normalized = snv_normalize(spectra)

    # Check normalization (mean~0, std~1)
    means = np.mean(normalized, axis=1)
    stds = np.std(normalized, axis=1)

    assert np.allclose(means, 0, atol=1e-10), "SNV: means should be ~0"
    assert np.allclose(stds, 1, atol=1e-10), "SNV: stds should be ~1"

    logger.info("  ✓ SNV normalization working correctly")


def test_wavelength_filtering():
    """Test wavelength filtering."""
    logger.info("Testing wavelength filtering...")

    # Create synthetic cube
    cube = np.random.randn(50, 50, 100)
    wavelengths = np.linspace(400, 1000, 100)

    # Filter to 450-925 nm
    filtered_cube, filtered_wl = filter_wavelengths(cube, wavelengths, 450, 925)

    assert filtered_cube.shape[2] < cube.shape[2], "Should have fewer bands"
    assert filtered_wl.min() >= 450, "Min wavelength should be >= 450"
    assert filtered_wl.max() <= 925, "Max wavelength should be <= 925"

    logger.info(f"  ✓ Filtered from {cube.shape[2]} to {filtered_cube.shape[2]} bands")


def test_grid_analyzer():
    """Test grid analysis."""
    logger.info("Testing grid analyzer...")

    # Create synthetic binary mask
    binary_mask = np.random.rand(256, 256) > 0.7

    # Analyze grid
    analyzer = GridAnalyzer(cell_size=32, crack_ratio_threshold=0.1)
    viz, stats = analyzer.analyze_grid(binary_mask, return_stats=True)

    assert viz.shape == (256, 256, 3), "Viz should be RGB"
    assert len(stats) > 0, "Should have cell statistics"
    assert all('percent_cracked' in s for s in stats), "Stats should have percentages"

    flagged = [s for s in stats if s['flagged']]
    logger.info(f"  ✓ Grid analysis: {len(flagged)} / {len(stats)} cells flagged")


def test_visualization():
    """Test visualization functions."""
    logger.info("Testing visualization...")

    # Create synthetic data
    base_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    binary_mask = np.random.rand(100, 100) > 0.8

    # Test binary overlay
    overlay = create_binary_overlay(base_img, binary_mask)
    assert overlay.shape == (100, 100, 3), "Overlay should be RGB"

    # Test RGB composite
    cube = np.random.randn(100, 100, 50)
    rgb = create_rgb_composite(cube)
    assert rgb.shape == (100, 100, 3), "RGB composite should be RGB"

    logger.info("  ✓ Visualization functions working")


def test_sklearn_model_wrapper():
    """Test sklearn model wrapper."""
    logger.info("Testing sklearn model wrapper...")

    # Create mock sklearn-like model
    class MockModel:
        def predict(self, X):
            return np.ones(X.shape[0], dtype=int)

        def predict_proba(self, X):
            n_samples = X.shape[0]
            return np.column_stack([
                np.random.rand(n_samples),
                np.random.rand(n_samples)
            ])

    # Wrap model
    mock = MockModel()
    wrapper = SklearnModelWrapper(mock, model_type="test")

    # Test predictions
    X = np.random.randn(10, 20)
    preds = wrapper.predict(X)
    probs = wrapper.predict_proba(X)

    assert preds.shape == (10,), "Predictions shape mismatch"
    assert probs.shape == (10, 2), "Probabilities shape mismatch"

    logger.info("  ✓ Sklearn model wrapper working")


def test_inference_engine():
    """Test hyperspectral inference engine."""
    logger.info("Testing inference engine...")

    # Create mock model
    class MockModel:
        def predict_proba(self, X):
            n_samples = X.shape[0]
            # Return mock probabilities
            return np.column_stack([
                1 - np.random.rand(n_samples) * 0.5,
                np.random.rand(n_samples) * 0.5
            ])

    # Create synthetic cube
    cube = np.random.randn(50, 50, 20)

    # Create engine
    wrapper = SklearnModelWrapper(MockModel())
    engine = HyperspectralInferenceEngine(wrapper, scaler=None, preprocess_fn=None)

    # Run inference
    prob_map = engine.predict_pixel_probabilities(cube, batch_size=1000)

    assert prob_map.shape == (50, 50, 2), "Probability map shape mismatch"
    assert np.all((prob_map >= 0) & (prob_map <= 1)), "Probabilities should be in [0, 1]"

    # Test binary mask
    binary_mask = engine.get_binary_mask(cube, positive_class=1, threshold=0.3)
    assert binary_mask.shape == (50, 50), "Binary mask shape mismatch"
    assert binary_mask.dtype == bool, "Binary mask should be boolean"

    logger.info("  ✓ Inference engine working")


def run_all_tests():
    """Run all functional tests."""
    logger.info("=" * 60)
    logger.info("Running Functional Tests")
    logger.info("=" * 60)

    tests = [
        test_snv_normalization,
        test_wavelength_filtering,
        test_grid_analyzer,
        test_visualization,
        test_sklearn_model_wrapper,
        test_inference_engine,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.error(f"❌ {failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
