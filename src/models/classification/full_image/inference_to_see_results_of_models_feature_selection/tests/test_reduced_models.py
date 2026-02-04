"""
Smoke tests for reduced model loading and feature alignment.

Tests:
1. Full-feature model loads and runs
2. Reduced model package loads, slices features correctly, and runs
3. Missing feature triggers error
4. Feature alignment maintains correct order
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.feature_alignment import (
    normalize_feature_name,
    wavelengths_to_feature_names,
    align_features_for_model,
    resolve_reduced_package,
    is_reduced_model_package,
    ReducedModelPackageInfo,
)
from app.models.model_manager import (
    ModelManager,
    ModelInfo,
    MODEL_CATEGORY_FULL,
    MODEL_CATEGORY_REDUCED,
)


# ============================================================================
# Feature Name Normalization Tests
# ============================================================================

class TestFeatureNameNormalization:
    """Test feature name normalization."""
    
    def test_normalize_with_nm_suffix(self):
        """Feature name with nm suffix stays normalized."""
        assert normalize_feature_name("452.25nm") == "452.25nm"
    
    def test_normalize_with_space_before_nm(self):
        """Space before nm is removed."""
        assert normalize_feature_name("452.25 nm") == "452.25nm"
    
    def test_normalize_without_nm_suffix(self):
        """Numeric string gets nm suffix added."""
        assert normalize_feature_name("452.25") == "452.25nm"
    
    def test_normalize_float_value(self):
        """Float value is formatted correctly."""
        # When passed as string
        result = normalize_feature_name("536.82")
        assert result == "536.82nm"
    
    def test_normalize_trailing_zeros(self):
        """Trailing zeros are handled correctly."""
        # Keep at least 2 decimal places
        assert normalize_feature_name("452.20nm") == "452.20nm"
        assert normalize_feature_name("452.2nm") == "452.20nm"


class TestWavelengthsToFeatureNames:
    """Test wavelength array to feature names conversion."""
    
    def test_basic_conversion(self):
        """Convert wavelength array to feature names."""
        wavelengths = np.array([452.25, 536.82, 892.95])
        names = wavelengths_to_feature_names(wavelengths)
        assert names == ["452.25nm", "536.82nm", "892.95nm"]
    
    def test_empty_array(self):
        """Empty array returns empty list."""
        assert wavelengths_to_feature_names(np.array([])) == []


# ============================================================================
# Feature Alignment Tests
# ============================================================================

class TestFeatureAlignment:
    """Test feature alignment for reduced models."""
    
    def test_full_feature_model_returns_unchanged(self):
        """With required_feature_names=None, X is returned unchanged."""
        X = np.random.rand(100, 159)
        full_names = [f"{400 + i * 2}.00nm" for i in range(159)]
        
        X_aligned = align_features_for_model(X, full_names, None)
        
        assert X_aligned is X  # Same object
        assert X_aligned.shape == (100, 159)
    
    def test_reduced_model_slices_features(self):
        """Reduced model gets correct feature subset."""
        X_full = np.arange(100 * 10).reshape(100, 10).astype(float)
        full_names = [f"{400 + i * 10}.00nm" for i in range(10)]
        # Select features at indices 2, 5, 7
        required_names = ["420.00nm", "450.00nm", "470.00nm"]
        
        X_aligned = align_features_for_model(X_full, full_names, required_names)
        
        assert X_aligned.shape == (100, 3)
        # Check that correct columns were selected
        np.testing.assert_array_equal(X_aligned[:, 0], X_full[:, 2])  # 420nm is index 2
        np.testing.assert_array_equal(X_aligned[:, 1], X_full[:, 5])  # 450nm is index 5
        np.testing.assert_array_equal(X_aligned[:, 2], X_full[:, 7])  # 470nm is index 7
    
    def test_order_is_preserved(self):
        """Features are aligned in the exact order specified."""
        X_full = np.arange(100 * 5).reshape(100, 5).astype(float)
        full_names = ["400.00nm", "410.00nm", "420.00nm", "430.00nm", "440.00nm"]
        # Reverse order
        required_names = ["440.00nm", "420.00nm", "400.00nm"]
        
        X_aligned = align_features_for_model(X_full, full_names, required_names)
        
        # Check order matches required_names, not sorted
        np.testing.assert_array_equal(X_aligned[:, 0], X_full[:, 4])  # 440nm
        np.testing.assert_array_equal(X_aligned[:, 1], X_full[:, 2])  # 420nm
        np.testing.assert_array_equal(X_aligned[:, 2], X_full[:, 0])  # 400nm
    
    def test_missing_feature_raises_error(self):
        """Missing required feature raises ValueError."""
        X_full = np.random.rand(100, 5)
        full_names = ["400.00nm", "410.00nm", "420.00nm", "430.00nm", "440.00nm"]
        required_names = ["400.00nm", "999.99nm"]  # 999.99nm doesn't exist
        
        with pytest.raises(ValueError) as exc_info:
            align_features_for_model(X_full, full_names, required_names)
        
        assert "999.99nm" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
    
    def test_normalization_allows_matching(self):
        """Feature names are normalized for matching."""
        X_full = np.random.rand(10, 3)
        full_names = ["452.25 nm", "536.82nm", "892.95nm"]  # Note space in first
        required_names = ["452.25nm"]  # No space
        
        # Should succeed due to normalization
        X_aligned = align_features_for_model(X_full, full_names, required_names)
        assert X_aligned.shape == (10, 1)


# ============================================================================
# Reduced Package Resolution Tests
# ============================================================================

class TestReducedPackageResolution:
    """Test reduced model package folder resolution."""
    
    def test_resolve_package_finds_nested_files(self, tmp_path):
        """Resolves model and feature_names.json in nested structure."""
        # Create nested structure: package/timestamp/multiclass/Balanced/
        balanced_dir = tmp_path / "2026-02-02_12-00-00" / "multiclass" / "Balanced"
        balanced_dir.mkdir(parents=True)
        
        # Create model file
        model_path = balanced_dir / "xgboost_model.pkl"
        model_path.write_bytes(b"fake model")
        
        # Create feature_names.json
        feature_names = ["452.25nm", "536.82nm"]
        (balanced_dir / "feature_names.json").write_text(json.dumps(feature_names))
        
        # Resolve
        pkg_info = resolve_reduced_package(tmp_path)
        
        assert pkg_info.model_path == model_path
        assert pkg_info.feature_names_path == balanced_dir / "feature_names.json"
        assert pkg_info.feature_names == feature_names
        assert pkg_info.balance_type == "Balanced"
    
    def test_invalid_package_raises_error(self, tmp_path):
        """Package without required files raises ValueError."""
        # Create empty directory
        (tmp_path / "empty").mkdir()
        
        with pytest.raises(ValueError) as exc_info:
            resolve_reduced_package(tmp_path / "empty")
        
        assert "Invalid reduced model package" in str(exc_info.value)
    
    def test_is_reduced_model_package_true(self, tmp_path):
        """is_reduced_model_package returns True for valid package."""
        # Create simple structure
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"fake model")
        (tmp_path / "feature_names.json").write_text(json.dumps(["452.25nm"]))
        
        assert is_reduced_model_package(tmp_path) is True
    
    def test_is_reduced_model_package_false_for_file(self, tmp_path):
        """is_reduced_model_package returns False for file."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"fake model")
        
        assert is_reduced_model_package(model_file) is False


# ============================================================================
# ModelManager Integration Tests
# ============================================================================

class TestModelManagerIntegration:
    """Integration tests for ModelManager with reduced models."""
    
    def test_load_full_feature_model(self, tmp_path):
        """Loading .pkl file sets model_category to 'full'."""
        # Create a mock sklearn model file
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        
        model_path = tmp_path / "full_model.pkl"
        joblib.dump(model, model_path)
        
        # Load with ModelManager
        manager = ModelManager()
        model_info = manager.load_model(str(model_path))
        
        assert model_info.model_category == MODEL_CATEGORY_FULL
        assert model_info.required_feature_names is None
        assert manager.required_feature_names is None
    
    def test_load_reduced_model_package(self, tmp_path):
        """Loading package folder sets model_category to 'reduced'."""
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # Create reduced package structure
        pkg_dir = tmp_path / "reduced_pkg"
        pkg_dir.mkdir()
        
        # Create model
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        joblib.dump(model, pkg_dir / "model.pkl")
        
        # Create feature_names.json
        feature_names = ["452.25nm", "536.82nm", "892.95nm"]
        (pkg_dir / "feature_names.json").write_text(json.dumps(feature_names))
        
        # Load with ModelManager
        manager = ModelManager()
        model_info = manager.load_model(str(pkg_dir))
        
        assert model_info.model_category == MODEL_CATEGORY_REDUCED
        assert model_info.required_feature_names == feature_names
        assert manager.required_feature_names == feature_names
    
    def test_unload_clears_required_feature_names(self, tmp_path):
        """Unloading model clears required_feature_names."""
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # Create reduced package
        pkg_dir = tmp_path / "reduced_pkg"
        pkg_dir.mkdir()
        
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        joblib.dump(model, pkg_dir / "model.pkl")
        (pkg_dir / "feature_names.json").write_text(json.dumps(["452.25nm"]))
        
        # Load and unload
        manager = ModelManager()
        manager.load_model(str(pkg_dir))
        assert manager.required_feature_names is not None
        
        manager.unload()
        assert manager.required_feature_names is None


# ============================================================================
# CLI Entry Point for Quick Validation
# ============================================================================

def run_smoke_test():
    """Run quick smoke test without pytest."""
    print("=" * 60)
    print("SMOKE TEST: Feature Alignment System")
    print("=" * 60)
    
    # Test 1: Feature name normalization
    print("\n[1] Testing feature name normalization...")
    assert normalize_feature_name("452.25nm") == "452.25nm"
    assert normalize_feature_name("452.25 nm") == "452.25nm"
    assert normalize_feature_name("452.25") == "452.25nm"
    print("    ✓ Feature name normalization works")
    
    # Test 2: Wavelengths to feature names
    print("\n[2] Testing wavelength conversion...")
    wl = np.array([452.25, 536.82, 892.95])
    names = wavelengths_to_feature_names(wl)
    assert names == ["452.25nm", "536.82nm", "892.95nm"]
    print("    ✓ Wavelength conversion works")
    
    # Test 3: Full-feature model (no alignment)
    print("\n[3] Testing full-feature model (no alignment)...")
    X = np.random.rand(100, 159)
    full_names = [f"{400 + i * 2}.00nm" for i in range(159)]
    X_out = align_features_for_model(X, full_names, None)
    assert X_out is X
    print("    ✓ Full-feature model returns unchanged data")
    
    # Test 4: Reduced model alignment
    print("\n[4] Testing reduced model feature alignment...")
    X_full = np.arange(50).reshape(5, 10).astype(float)
    full_names = [f"{400 + i * 10}.00nm" for i in range(10)]
    required = ["420.00nm", "450.00nm", "470.00nm"]  # indices 2, 5, 7
    X_aligned = align_features_for_model(X_full, full_names, required)
    assert X_aligned.shape == (5, 3)
    np.testing.assert_array_equal(X_aligned[:, 0], X_full[:, 2])
    np.testing.assert_array_equal(X_aligned[:, 1], X_full[:, 5])
    np.testing.assert_array_equal(X_aligned[:, 2], X_full[:, 7])
    print("    ✓ Reduced model alignment works correctly")
    
    # Test 5: Order preservation
    print("\n[5] Testing order preservation...")
    required_rev = ["470.00nm", "420.00nm"]  # Reversed subset
    X_rev = align_features_for_model(X_full, full_names, required_rev)
    np.testing.assert_array_equal(X_rev[:, 0], X_full[:, 7])  # 470nm first
    np.testing.assert_array_equal(X_rev[:, 1], X_full[:, 2])  # 420nm second
    print("    ✓ Feature order matches training order")
    
    # Test 6: Missing feature error
    print("\n[6] Testing missing feature detection...")
    try:
        align_features_for_model(X_full, full_names, ["999.99nm"])
        print("    ✗ Should have raised error!")
        return False
    except ValueError as e:
        assert "999.99nm" in str(e)
        print("    ✓ Missing feature raises error with feature name")
    
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = run_smoke_test()
    sys.exit(0 if success else 1)
