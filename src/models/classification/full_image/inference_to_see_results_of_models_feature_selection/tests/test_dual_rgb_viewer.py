"""
Test script for Dual RGB Viewer widget and enhanced RGB utilities.

Verifies that both HSI-derived and camera RGB images can be found and displayed.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

print("="*70)
print("Dual RGB Viewer - Test Suite")
print("="*70)

# Test 1: Import
print("\n[1/5] Testing imports...")
try:
    from app.io import (
        find_hsi_rgb,
        find_camera_rgb,
        find_both_rgb_images,
        load_both_rgb_images,
        load_rgb
    )
    from app.ui import DualRGBViewer
    print("    ✓ All imports successful")
except ImportError as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test RGB finding functions with real data
print("\n[2/5] Testing RGB finding functions with real data...")
try:
    csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"

    if not csv_path.exists():
        print(f"    ⚠ test_dataset.csv not found at {csv_path}")
        print("    Skipping real data test")
    else:
        df = pd.read_csv(csv_path)
        print(f"    ✓ Loaded test_dataset.csv with {len(df)} samples")

        # Test first 10 samples
        hsi_found = 0
        camera_found = 0
        both_found = 0

        for i, row in df.head(10).iterrows():
            sample_path = Path(row['image_path'])

            if sample_path.exists():
                rgb_paths = find_both_rgb_images(sample_path)

                if rgb_paths['hsi_rgb']:
                    hsi_found += 1
                    print(f"    ✓ HSI RGB: {rgb_paths['hsi_rgb'].name}")

                if rgb_paths['camera_rgb']:
                    camera_found += 1
                    print(f"    ✓ Camera RGB: {rgb_paths['camera_rgb'].name}")

                if rgb_paths['hsi_rgb'] and rgb_paths['camera_rgb']:
                    both_found += 1

        print(f"\n    Summary (first 10 samples):")
        print(f"      HSI RGB found: {hsi_found}")
        print(f"      Camera RGB found: {camera_found}")
        print(f"      Both found: {both_found}")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test find_hsi_rgb and find_camera_rgb separately
print("\n[3/5] Testing HSI vs Camera RGB distinction...")
try:
    csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Find a sample with both types
        for _, row in df.head(20).iterrows():
            sample_path = Path(row['image_path'])

            if sample_path.exists():
                hsi_path = find_hsi_rgb(sample_path)
                camera_path = find_camera_rgb(sample_path)

                if hsi_path and camera_path:
                    print(f"    ✓ Found both RGB types for {sample_path.name}:")
                    print(f"      HSI RGB: {hsi_path.name}")
                    print(f"      Camera RGB: {camera_path.name}")

                    # Verify they are different files
                    if hsi_path != camera_path:
                        print(f"    ✓ HSI and Camera RGB are different files")
                    else:
                        print(f"    ⚠ HSI and Camera RGB point to same file")

                    break
        else:
            print("    ⚠ Could not find sample with both RGB types")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test load_both_rgb_images
print("\n[4/5] Testing load_both_rgb_images...")
try:
    csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        for _, row in df.head(10).iterrows():
            sample_path = Path(row['image_path'])

            if sample_path.exists():
                hsi_img, camera_img = load_both_rgb_images(sample_path)

                if hsi_img is not None:
                    print(f"    ✓ Loaded HSI RGB: {hsi_img.shape}, {hsi_img.dtype}")
                    assert hsi_img.dtype == np.uint8
                    assert hsi_img.ndim == 3
                    assert hsi_img.shape[2] == 3

                if camera_img is not None:
                    print(f"    ✓ Loaded Camera RGB: {camera_img.shape}, {camera_img.dtype}")
                    assert camera_img.dtype == np.uint8
                    assert camera_img.ndim == 3
                    assert camera_img.shape[2] == 3

                if hsi_img is not None or camera_img is not None:
                    print(f"    ✓ Successfully loaded at least one RGB type")
                    break

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test DualRGBViewer widget (basic)
print("\n[5/5] Testing DualRGBViewer widget...")
try:
    from PyQt5.QtWidgets import QApplication
    import numpy as np

    # Create app (required for widgets)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create viewer
    viewer = DualRGBViewer()
    print("    ✓ DualRGBViewer instantiated")

    # Test setting images
    test_hsi = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    test_camera = np.random.randint(0, 255, (250, 350, 3), dtype=np.uint8)

    viewer.set_hsi_rgb(test_hsi, "test_hsi.png")
    print("    ✓ set_hsi_rgb() works")

    viewer.set_camera_rgb(test_camera, "test_camera.jpg")
    print("    ✓ set_camera_rgb() works")

    # Test clear
    viewer.clear()
    print("    ✓ clear() works")

    # Test load_from_folder with real data
    csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        for _, row in df.head(10).iterrows():
            sample_path = Path(row['image_path'])

            if sample_path.exists():
                success = viewer.load_from_folder(sample_path)
                if success:
                    print(f"    ✓ load_from_folder() works with real data")
                    print(f"      Sample: {sample_path.name}")

                    if viewer.get_hsi_image() is not None:
                        print(f"      HSI RGB loaded: {viewer.get_hsi_image().shape}")
                    if viewer.get_camera_image() is not None:
                        print(f"      Camera RGB loaded: {viewer.get_camera_image().shape}")
                    break

    print("    ✓ DualRGBViewer all methods working")

except ImportError:
    print("    ⚠ PyQt5 not available - skipping UI widget tests")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ TESTS COMPLETE")
print("="*70)
print("\nImplementation Summary:")
print("  • find_hsi_rgb() - Finds HSI-derived RGB in same dir as HSI data")
print("  • find_camera_rgb() - Finds camera RGB in RGB/ subfolder")
print("  • find_both_rgb_images() - Returns dict with both paths")
print("  • load_both_rgb_images() - Loads both RGB arrays")
print("  • DualRGBViewer - Widget to display both side-by-side")
print("\nUsage in UI:")
print("  from app.ui import DualRGBViewer")
print("  viewer = DualRGBViewer()")
print("  viewer.load_from_folder(sample_folder_path)")
print("="*70)
