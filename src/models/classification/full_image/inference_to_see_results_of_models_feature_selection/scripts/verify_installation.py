"""
Quick verification script - checks all imports work correctly.
Run this to verify the implementation is properly installed.

Usage: python scripts/verify_installation.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("ImageViewer Widget - Installation Verification")
print("="*70)

# Test 1: Import UI module
print("\n[1/5] Testing UI module import...")
try:
    from app.ui import ImageViewer
    print("    ✓ app.ui.ImageViewer imported successfully")
except ImportError as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Import utils
print("\n[2/5] Testing utils import...")
try:
    from app.utils import normalize_to_uint8, apply_colormap
    print("    ✓ app.utils.normalize_to_uint8 imported successfully")
    print("    ✓ app.utils.apply_colormap imported successfully")
except ImportError as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Check PyQt5 availability
print("\n[3/5] Testing PyQt5 availability...")
try:
    from PyQt5.QtWidgets import QApplication
    print("    ✓ PyQt5 is available")
except ImportError:
    print("    ⚠ WARNING: PyQt5 not available (UI features will not work)")

# Test 4: Check NumPy availability
print("\n[4/5] Testing NumPy availability...")
try:
    import numpy as np
    print(f"    ✓ NumPy {np.__version__} is available")
except ImportError:
    print("    ✗ FAILED: NumPy is required")
    sys.exit(1)

# Test 5: Verify class structure
print("\n[5/5] Testing class structure...")
try:
    # Check ImageViewer has required methods
    required_methods = ['set_image', 'set_overlay', 'clear', 'reset_zoom']
    for method in required_methods:
        assert hasattr(ImageViewer, method), f"Missing method: {method}"
        print(f"    ✓ ImageViewer.{method}() exists")

    # Check function signatures
    import inspect
    sig = inspect.signature(normalize_to_uint8)
    assert 'img2d' in sig.parameters
    assert 'method' in sig.parameters
    print("    ✓ normalize_to_uint8() signature correct")

    sig = inspect.signature(apply_colormap)
    assert 'img2d_float01' in sig.parameters
    assert 'name' in sig.parameters
    print("    ✓ apply_colormap() signature correct")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✅ ALL CHECKS PASSED")
print("="*70)
print("\nInstallation verified successfully!")
print("\nNext steps:")
print("  • Run demo: python scripts/demo_image_viewer.py")
print("  • Run tests: python tests/test_image_viewer.py")
print("  • Integrate into UI: Import from app.ui and app.utils")
print("\nDocumentation:")
print("  • README_IMAGE_VIEWER.md - Main documentation")
print("  • IMAGEVIEWER_QUICK_REFERENCE.md - Quick reference")
print("  • IMPLEMENTATION_SUMMARY.md - Complete summary")
print("="*70)
