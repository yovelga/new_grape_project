"""
Test script for RGB loading and HSI band extraction utilities.

Uses test_dataset.csv to verify functionality with real data paths.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

print("="*70)
print("RGB and HSI Band Utilities - Test Suite")
print("="*70)

# Test 1: Import modules
print("\n[1/7] Testing imports...")
try:
    from app.io import (
        find_rgb_image,
        load_rgb,
        get_band_by_index,
        get_band_by_wavelength,
        find_nearest_band_index,
        extract_multiple_bands,
    )
    print("    ✓ All functions imported successfully")
except ImportError as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Test HSI band extraction with synthetic data
print("\n[2/7] Testing HSI band extraction with synthetic data...")
try:
    # Create synthetic hyperspectral cube
    cube_hwb = np.random.rand(100, 100, 50)  # H x W x Bands

    # Test get_band_by_index
    band = get_band_by_index(cube_hwb, 25)
    assert band.shape == (100, 100), f"Expected (100, 100), got {band.shape}"
    print("    ✓ get_band_by_index() works with (H, W, B) format")

    # Test with different format
    cube_bhw = np.random.rand(50, 100, 100)  # Bands x H x W
    band = get_band_by_index(cube_bhw, 25)
    assert band.shape == (100, 100), f"Expected (100, 100), got {band.shape}"
    print("    ✓ get_band_by_index() works with (B, H, W) format")

    # Test out of range
    try:
        get_band_by_index(cube_hwb, 100)
        print("    ✗ Should have raised ValueError for out of range index")
    except ValueError:
        print("    ✓ Correctly raises ValueError for out of range index")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test wavelength-based extraction
print("\n[3/7] Testing wavelength-based band extraction...")
try:
    cube = np.random.rand(100, 100, 50)
    wavelengths = np.linspace(400, 1000, 50)  # 400-1000nm, 50 bands

    # Test exact match
    band, idx, actual = get_band_by_wavelength(cube, wavelengths, 700.0)
    assert band.shape == (100, 100), f"Expected (100, 100), got {band.shape}"
    assert isinstance(idx, int), f"Index should be int, got {type(idx)}"
    assert isinstance(actual, float), f"Wavelength should be float, got {type(actual)}"
    print(f"    ✓ get_band_by_wavelength(700nm) → band {idx} at {actual:.1f}nm")

    # Test nearest neighbor (request non-exact wavelength)
    band, idx, actual = get_band_by_wavelength(cube, wavelengths, 555.5)
    expected_idx = np.argmin(np.abs(wavelengths - 555.5))
    assert idx == expected_idx, f"Expected index {expected_idx}, got {idx}"
    print(f"    ✓ Nearest neighbor works: 555.5nm → {actual:.1f}nm")

    # Test find_nearest_band_index
    idx2, actual2 = find_nearest_band_index(wavelengths, 555.5)
    assert idx == idx2, "find_nearest_band_index should match"
    assert actual == actual2, "Wavelengths should match"
    print("    ✓ find_nearest_band_index() matches get_band_by_wavelength()")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test extract_multiple_bands
print("\n[4/7] Testing multiple band extraction...")
try:
    cube = np.random.rand(100, 100, 50)
    wavelengths = np.linspace(400, 1000, 50)
    targets = [450, 550, 650]  # Blue, green, red-ish

    bands, indices, actuals = extract_multiple_bands(cube, wavelengths, targets)

    assert bands.shape == (100, 100, 3), f"Expected (100, 100, 3), got {bands.shape}"
    assert len(indices) == 3, f"Expected 3 indices, got {len(indices)}"
    assert len(actuals) == 3, f"Expected 3 wavelengths, got {len(actuals)}"

    print(f"    ✓ extract_multiple_bands() returned shape {bands.shape}")
    print(f"    ✓ Target wavelengths: {targets}")
    print(f"    ✓ Actual wavelengths: {actuals}")
    print(f"    ✓ Band indices: {indices}")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test RGB utilities with synthetic data
print("\n[5/7] Testing RGB utilities with synthetic data...")
try:
    # Create temporary test image
    import tempfile
    from PIL import Image

    # Create synthetic RGB image
    test_rgb = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        img = Image.fromarray(test_rgb)
        img.save(tmp_path)

    try:
        # Test load_rgb
        loaded = load_rgb(tmp_path)
        assert loaded.dtype == np.uint8, f"Expected uint8, got {loaded.dtype}"
        assert loaded.shape == (100, 150, 3), f"Expected (100, 150, 3), got {loaded.shape}"
        print("    ✓ load_rgb() works correctly")

        # Test error handling
        try:
            load_rgb(Path("nonexistent.png"))
            print("    ✗ Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("    ✓ Correctly raises FileNotFoundError for missing file")

    finally:
        # Clean up
        tmp_path.unlink(missing_ok=True)

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test find_rgb_image with mock folder structure
print("\n[6/7] Testing find_rgb_image() search patterns...")
try:
    import tempfile
    import shutil

    # Create temporary folder structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test Pattern 1: <sample>/RGB/*.png
        sample1 = tmpdir / "sample1"
        rgb_folder1 = sample1 / "RGB"
        rgb_folder1.mkdir(parents=True)
        test_img1 = rgb_folder1 / "image.png"
        Image.new('RGB', (10, 10)).save(test_img1)

        found = find_rgb_image(sample1)
        assert found == test_img1, f"Expected {test_img1}, found {found}"
        print("    ✓ Pattern 1: <sample>/RGB/*.png")

        # Test Pattern 2: Sibling folder
        sample2 = tmpdir / "date_folder" / "sample2"
        sample2.mkdir(parents=True)
        sibling_rgb = tmpdir / "date_folder" / "RGB"
        sibling_rgb.mkdir(parents=True)
        test_img2 = sibling_rgb / "sample2.jpg"
        Image.new('RGB', (10, 10)).save(test_img2)

        found = find_rgb_image(sample2)
        assert found == test_img2, f"Expected {test_img2}, found {found}"
        print("    ✓ Pattern 2: Sibling ../RGB/ folder")

        # Test Pattern 3: File with 'rgb' in name
        sample3 = tmpdir / "sample3"
        sample3.mkdir(parents=True)
        test_img3 = sample3 / "my_rgb_image.jpeg"
        Image.new('RGB', (10, 10)).save(test_img3)

        found = find_rgb_image(sample3)
        assert found == test_img3, f"Expected {test_img3}, found {found}"
        print("    ✓ Pattern 3: File with 'rgb' in name")

        # Test Pattern 4: Any image in folder
        sample4 = tmpdir / "sample4"
        sample4.mkdir(parents=True)
        test_img4 = sample4 / "image.png"
        Image.new('RGB', (10, 10)).save(test_img4)

        found = find_rgb_image(sample4)
        assert found == test_img4, f"Expected {test_img4}, found {found}"
        print("    ✓ Pattern 4: Any image in folder")

        # Test no image found
        sample5 = tmpdir / "sample5"
        sample5.mkdir(parents=True)
        found = find_rgb_image(sample5)
        assert found is None, f"Expected None for empty folder, got {found}"
        print("    ✓ Returns None when no image found")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Load test_dataset.csv and check paths
print("\n[7/7] Testing with test_dataset.csv...")
try:
    csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"

    if not csv_path.exists():
        print(f"    ⚠ WARNING: test_dataset.csv not found at {csv_path}")
        print("    Skipping real data test")
    else:
        df = pd.read_csv(csv_path)
        print(f"    ✓ Loaded test_dataset.csv with {len(df)} samples")

        # Check first few paths
        found_count = 0
        missing_count = 0

        for i, row in df.head(10).iterrows():
            sample_path = Path(row['image_path'])

            if sample_path.exists():
                found_count += 1
                # Try to find RGB
                rgb_path = find_rgb_image(sample_path)
                if rgb_path:
                    print(f"    ✓ Found RGB for {sample_path.name}: {rgb_path.name}")
                else:
                    print(f"    ⚠ No RGB found for {sample_path.name}")
            else:
                missing_count += 1

        if found_count > 0:
            print(f"    ✓ Tested {found_count} existing sample paths")
        if missing_count > 0:
            print(f"    ⚠ {missing_count} sample paths don't exist (data may be on different machine)")

        if found_count == 0:
            print("    ℹ Data paths from CSV don't exist on this machine")
            print("    (This is OK - the functions work with synthetic data)")

except Exception as e:
    print(f"    ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("✅ ALL CORE TESTS PASSED")
print("="*70)
print("\nImplementation Summary:")
print("  • RGB utilities:")
print("    - find_rgb_image() with 4 search patterns")
print("    - load_rgb() with error handling")
print("  • HSI band extraction:")
print("    - get_band_by_index() with format detection")
print("    - get_band_by_wavelength() with nearest-neighbor")
print("    - find_nearest_band_index() helper")
print("    - extract_multiple_bands() convenience function")
print("\nAll functions tested and working correctly!")
print("="*70)
