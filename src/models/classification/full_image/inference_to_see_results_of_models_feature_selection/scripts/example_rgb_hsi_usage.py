"""
Example script demonstrating RGB and HSI utilities usage.

Shows how to load RGB and HSI data from sample folders.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from app.io import (
    find_rgb_image,
    load_rgb,
    ENVIReader,
    get_band_by_index,
    get_band_by_wavelength,
    extract_multiple_bands,
)

print("="*70)
print("RGB and HSI Utilities - Usage Examples")
print("="*70)

# Example 1: Load dataset and find RGB images
print("\n[Example 1] Finding RGB images from test_dataset.csv")
print("-"*70)

csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"

# Project root for resolving relative image paths
_PROJECT_ROOT = Path(__file__).resolve().parents[6]

def _resolve_path(p):
    """Resolve relative path against project root."""
    path = Path(str(p))
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path

if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from CSV")

    # Check first few samples
    for i, row in df.head(5).iterrows():
        sample_path = _resolve_path(row['image_path'])
        grape_id = row['grape_id']
        week = row['week_date']

        if sample_path.exists():
            rgb_path = find_rgb_image(sample_path)
            if rgb_path:
                print(f"  ✓ {grape_id}/{week}: Found {rgb_path.name}")

                # Try to load it
                try:
                    rgb_img = load_rgb(rgb_path)
                    print(f"    Loaded RGB: {rgb_img.shape}, {rgb_img.dtype}")
                except Exception as e:
                    print(f"    Failed to load: {e}")
            else:
                print(f"  ✗ {grape_id}/{week}: No RGB found")
        else:
            print(f"  ⚠ {grape_id}/{week}: Path doesn't exist")
else:
    print(f"CSV not found at {csv_path}")

# Example 2: HSI band extraction with synthetic data
print("\n[Example 2] HSI band extraction (synthetic data)")
print("-"*70)

# Create synthetic HSI cube
print("Creating synthetic HSI cube (512x512x100 bands)...")
cube = np.random.rand(512, 512, 100).astype(np.float32)
wavelengths = np.linspace(400, 1000, 100)  # 400-1000nm

print(f"Cube shape: {cube.shape}")
print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

# Extract by index
print("\nExtract band by index:")
band_50 = get_band_by_index(cube, 50)
print(f"  Band 50 shape: {band_50.shape}")
print(f"  Band 50 range: [{band_50.min():.3f}, {band_50.max():.3f}]")

# Extract by wavelength
print("\nExtract band by wavelength:")
target = 550.0  # Green
band, idx, actual = get_band_by_wavelength(cube, wavelengths, target)
print(f"  Requested: {target} nm")
print(f"  Got: Band {idx} at {actual:.1f} nm")
print(f"  Shape: {band.shape}")

# Extract multiple bands for false-color
print("\nExtract multiple bands (NIR, R, G):")
targets = [800, 650, 550]
bands, indices, actuals = extract_multiple_bands(cube, wavelengths, targets)
print(f"  Targets: {targets}")
print(f"  Actuals: {[f'{a:.1f}' for a in actuals]}")
print(f"  Indices: {indices.tolist()}")
print(f"  Result shape: {bands.shape}")

# Example 3: Complete workflow simulation
print("\n[Example 3] Complete workflow (with real paths if available)")
print("-"*70)

# Try to find a real sample
if csv_path.exists():
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        sample_path = _resolve_path(row['image_path'])

        if sample_path.exists():
            print(f"\nProcessing sample: {sample_path.name}")

            # Step 1: Find and load RGB
            rgb_path = find_rgb_image(sample_path)
            if rgb_path:
                rgb = load_rgb(rgb_path)
                print(f"  ✓ RGB loaded: {rgb.shape}")
            else:
                print(f"  ✗ No RGB found")

            # Step 2: Find and load HSI
            hdr_files = list(sample_path.glob("*.hdr"))
            if hdr_files:
                print(f"  ✓ Found HDR file: {hdr_files[0].name}")

                try:
                    reader = ENVIReader(str(hdr_files[0]))
                    hsi_cube = reader.read()
                    hsi_wavelengths = reader.get_wavelengths()

                    print(f"  ✓ HSI cube loaded: {hsi_cube.shape}")

                    if hsi_wavelengths is not None:
                        print(f"  ✓ Wavelengths: {len(hsi_wavelengths)} bands")
                        print(f"    Range: {hsi_wavelengths[0]:.1f} - {hsi_wavelengths[-1]:.1f} nm")

                        # Extract NIR band
                        nir, idx, actual = get_band_by_wavelength(
                            hsi_cube, hsi_wavelengths, 800.0
                        )
                        print(f"  ✓ NIR band (800nm): Band {idx} at {actual:.1f}nm")
                        print(f"    Shape: {nir.shape}")
                        print(f"    Range: [{nir.min():.3f}, {nir.max():.3f}]")

                except Exception as e:
                    print(f"  ✗ Failed to load HSI: {e}")
            else:
                print(f"  ✗ No HDR file found")

            # Only process first valid sample
            break
    else:
        print("\nNo valid samples found in dataset")
else:
    print("Dataset CSV not available - skipping real data example")

# Example 4: Integration with visualization
print("\n[Example 4] Visualization integration example")
print("-"*70)

print("Creating synthetic false-color composite...")
cube = np.random.rand(256, 256, 100).astype(np.float32) * 100 + 500
wavelengths = np.linspace(400, 1000, 100)

# Extract NIR, R, G
targets = [800, 650, 550]
bands, indices, actuals = extract_multiple_bands(cube, wavelengths, targets)

print(f"Extracted bands at: {[f'{a:.1f}nm' for a in actuals]}")

# Normalize for display
from app.utils import normalize_to_uint8

false_rgb = np.stack([
    normalize_to_uint8(bands[:,:,i], method="percentile")
    for i in range(3)
], axis=2)

print(f"False-color RGB: {false_rgb.shape}, {false_rgb.dtype}")
print(f"Ready for display with ImageViewer")

# Note: To display, you would do:
# from app.ui import ImageViewer
# viewer = ImageViewer()
# viewer.set_image(false_rgb)
# viewer.show()

print("\n" + "="*70)
print("✅ Examples complete!")
print("="*70)
print("\nKey takeaways:")
print("  • find_rgb_image() searches multiple patterns")
print("  • load_rgb() always returns uint8 RGB")
print("  • get_band_by_index() for direct access")
print("  • get_band_by_wavelength() for spectral selection")
print("  • extract_multiple_bands() for composites")
print("\nFor more info, see RGB_HSI_QUICK_REFERENCE.md")
print("="*70)
