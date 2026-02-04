"""
Simple test script to verify ImageViewer and image_ops utilities.

Run this to test the implementation without UI wiring.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# Test imports
try:
    from app.ui import ImageViewer
    from app.utils import normalize_to_uint8, apply_colormap
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test image_ops functions
print("\n=== Testing image_ops ===")

# Create test image
test_img = np.random.randn(100, 100) * 50 + 128

# Test normalization methods
for method in ["percentile", "minmax", "std"]:
    result = normalize_to_uint8(test_img, method=method)
    assert result.dtype == np.uint8, f"Wrong dtype for {method}"
    assert result.shape == test_img.shape, f"Wrong shape for {method}"
    print(f"✓ normalize_to_uint8 with method='{method}' works")

# Test colormaps
test_normalized = test_img / test_img.max()
for cmap in ["viridis", "jet", "hot", "cool", "gray"]:
    result = apply_colormap(test_normalized, name=cmap)
    assert result.dtype == np.uint8, f"Wrong dtype for {cmap}"
    assert result.shape == (*test_img.shape, 3), f"Wrong shape for {cmap}"
    print(f"✓ apply_colormap with name='{cmap}' works")

# Test edge cases
empty_img = np.array([])
result = normalize_to_uint8(empty_img)
assert result.size == 0, "Empty image handling failed"
print("✓ Edge case: empty image")

constant_img = np.ones((50, 50)) * 42
result = normalize_to_uint8(constant_img)
assert result.dtype == np.uint8, "Constant image dtype wrong"
print("✓ Edge case: constant image")

print("\n=== Testing ImageViewer ===")

try:
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = ImageViewer()
    print("✓ ImageViewer instantiation successful")

    # Test set_image with grayscale
    gray_img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    viewer.set_image(gray_img)
    print("✓ set_image with grayscale uint8")

    # Test set_image with RGB
    rgb_img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    viewer.set_image(rgb_img)
    print("✓ set_image with RGB uint8")

    # Test set_image with float
    float_img = np.random.rand(200, 300)
    viewer.set_image(float_img)
    print("✓ set_image with float [0,1]")

    # Test overlay
    mask = np.random.rand(200, 300) > 0.8
    viewer.set_overlay(mask, alpha=0.5)
    print("✓ set_overlay with boolean mask")

    # Test clear overlay
    viewer.set_overlay(None)
    print("✓ set_overlay with None (clear)")

    # Test clear all
    viewer.clear()
    print("✓ clear method")

    # Test reset_zoom
    viewer.set_image(gray_img)
    viewer.reset_zoom()
    print("✓ reset_zoom method")

    print("\n✓ All tests passed!")
    print("\nNote: This test doesn't launch the UI. To verify interactivity:")
    print("  - Integrate ImageViewer into binary_class_inference_ui.py")
    print("  - Test mouse wheel zoom and click-drag panning manually")

except ImportError:
    print("⚠ PyQt5 not available - skipping UI tests")
    print("✓ Non-UI components tested successfully")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
