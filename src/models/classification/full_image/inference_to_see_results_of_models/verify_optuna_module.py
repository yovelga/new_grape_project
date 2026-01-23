"""
Verification script for Optuna tuning module.

Tests that all components can be imported and basic functionality works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.tuning.dataset_loader import DatasetCSVLoader, Sample
        print("✓ DatasetCSVLoader imported")
        
        from app.tuning.postprocess_patch import (
            PatchClassifierParams, 
            PostprocessPatchClassifier,
            ClassificationResult
        )
        print("✓ PostprocessPatchClassifier imported")
        
        from app.tuning.metrics_report import (
            ConfusionMatrix,
            BinaryMetrics,
            MetricsCalculator,
            MetricsReport
        )
        print("✓ MetricsReport imported")
        
        from app.tuning.inference_cache import InferenceCache
        print("✓ InferenceCache imported")
        
        from app.tuning.optuna_tuner import OptunaTuner
        print("✓ OptunaTuner imported")
        
        from app.tuning.optuna_worker import OptunaWorker
        print("✓ OptunaWorker imported")
        
        from app.ui.optuna_tab import OptunaTabWidget
        print("✓ OptunaTabWidget imported")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of each component."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from app.tuning.dataset_loader import DatasetCSVLoader, Sample
        from app.tuning.postprocess_patch import PatchClassifierParams, PostprocessPatchClassifier
        from app.tuning.metrics_report import MetricsCalculator, MetricsReport
        from app.tuning.inference_cache import InferenceCache
        
        # Test Sample creation
        sample = Sample(path="/path/to/image.hdr", label=1, sample_id="test_001")
        assert sample.label == 1
        print("✓ Sample creation works")
        
        # Test PatchClassifierParams
        params = PatchClassifierParams(
            pixel_threshold=0.9,
            min_blob_area=100,
            morph_size=5,
            patch_size=32,
            patch_crack_pct_threshold=10.0
        )
        params.validate()
        print("✓ PatchClassifierParams validation works")
        
        # Test PostprocessPatchClassifier with dummy data
        classifier = PostprocessPatchClassifier(params)
        dummy_prob_map = np.random.rand(100, 100)
        result = classifier.classify(dummy_prob_map)
        assert result.predicted_label in [0, 1]
        print("✓ PostprocessPatchClassifier classification works")
        
        # Test MetricsCalculator
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        metrics = MetricsCalculator.compute_metrics(y_true, y_pred)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_score <= 1
        print("✓ MetricsCalculator computation works")
        
        # Test MetricsReport
        report = MetricsReport()
        report.add_split('test', y_true, y_pred)
        assert 'test' in report.metrics
        print("✓ MetricsReport works")
        
        # Test InferenceCache
        cache = InferenceCache(max_cache_size_mb=10)
        cache.put("/test/path.hdr", dummy_prob_map)
        cached = cache.get("/test/path.hdr")
        assert cached is not None
        assert np.array_equal(cached, dummy_prob_map)
        print("✓ InferenceCache works")
        
        print("\n✓ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Optuna Tuning Module Verification")
    print("=" * 60)
    
    import_ok = test_imports()
    
    if import_ok:
        func_ok = test_basic_functionality()
    else:
        func_ok = False
    
    print("\n" + "=" * 60)
    if import_ok and func_ok:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe Optuna tuning module is ready to use!")
        print("Launch the UI with: python binary_class_inference_ui.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the errors before using the module.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
