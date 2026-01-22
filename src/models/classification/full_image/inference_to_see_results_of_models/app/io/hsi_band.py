"""
HSI (Hyperspectral Image) band extraction utilities.

Provides functions to extract specific bands from hyperspectral cubes
by index or wavelength.
"""

import numpy as np
from typing import Tuple, Optional


def get_band_by_index(cube: np.ndarray, band_idx: int) -> np.ndarray:
    """
    Extract a single band from hyperspectral cube by index.

    Args:
        cube: Hyperspectral cube with shape (H, W, Bands) or (Bands, H, W)
        band_idx: Band index (0-based)

    Returns:
        2D array with shape (H, W) containing the requested band

    Raises:
        ValueError: If band_idx is out of range
        IndexError: If cube dimensions are invalid

    Example:
        >>> cube = np.random.rand(512, 512, 224)  # H x W x Bands
        >>> band = get_band_by_index(cube, 100)
        >>> print(band.shape)
        (512, 512)
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    # Detect format: (H, W, Bands) or (Bands, H, W)
    # Assume smallest dimension is bands
    band_axis = np.argmin(cube.shape)

    num_bands = cube.shape[band_axis]

    if band_idx < 0 or band_idx >= num_bands:
        raise ValueError(
            f"Band index {band_idx} out of range [0, {num_bands-1}]"
        )

    # Extract band based on axis
    if band_axis == 0:
        # Shape: (Bands, H, W)
        return cube[band_idx, :, :]
    elif band_axis == 1:
        # Shape: (H, Bands, W) - unusual but handle it
        return cube[:, band_idx, :]
    else:
        # Shape: (H, W, Bands) - most common
        return cube[:, :, band_idx]


def get_band_by_wavelength(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    target_nm: float
) -> Tuple[np.ndarray, int, float]:
    """
    Extract band closest to target wavelength from hyperspectral cube.

    Uses robust nearest-neighbor selection to find the band with wavelength
    closest to the target.

    Args:
        cube: Hyperspectral cube with shape (H, W, Bands) or (Bands, H, W)
        wavelengths: Array of wavelengths in nanometers for each band
        target_nm: Target wavelength in nanometers

    Returns:
        Tuple of:
            - band_2d: 2D array (H, W) with the selected band
            - idx: Index of the selected band (0-based)
            - actual_nm: Actual wavelength of the selected band

    Raises:
        ValueError: If wavelengths array length doesn't match cube bands
        ValueError: If inputs are invalid

    Example:
        >>> cube = np.random.rand(512, 512, 224)
        >>> wavelengths = np.linspace(400, 1000, 224)
        >>> band, idx, actual = get_band_by_wavelength(cube, wavelengths, 550.0)
        >>> print(f"Requested 550nm, got {actual:.1f}nm at index {idx}")
        >>> print(band.shape)
        (512, 512)
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    wavelengths = np.asarray(wavelengths)

    if wavelengths.ndim != 1:
        raise ValueError(
            f"Wavelengths must be 1D array, got shape {wavelengths.shape}"
        )

    # Detect band axis (smallest dimension)
    band_axis = np.argmin(cube.shape)
    num_bands = cube.shape[band_axis]

    if len(wavelengths) != num_bands:
        raise ValueError(
            f"Wavelengths length ({len(wavelengths)}) doesn't match "
            f"number of bands ({num_bands})"
        )

    # Find nearest wavelength using robust method
    # Handle edge cases: NaN, inf, empty arrays
    if len(wavelengths) == 0:
        raise ValueError("Wavelengths array is empty")

    if not np.isfinite(target_nm):
        raise ValueError(f"Target wavelength must be finite, got {target_nm}")

    # Calculate absolute differences
    differences = np.abs(wavelengths - target_nm)

    # Find index of minimum difference
    idx = int(np.argmin(differences))
    actual_nm = float(wavelengths[idx])

    # Extract the band
    band_2d = get_band_by_index(cube, idx)

    return band_2d, idx, actual_nm


def find_nearest_band_index(
    wavelengths: np.ndarray,
    target_nm: float
) -> Tuple[int, float]:
    """
    Find index of band with wavelength nearest to target.

    Helper function for wavelength-based band selection.

    Args:
        wavelengths: Array of wavelengths in nanometers
        target_nm: Target wavelength in nanometers

    Returns:
        Tuple of (index, actual_wavelength)

    Example:
        >>> wavelengths = np.array([450, 500, 550, 600, 650])
        >>> idx, actual = find_nearest_band_index(wavelengths, 525)
        >>> print(f"Index: {idx}, Wavelength: {actual}nm")
        Index: 2, Wavelength: 550.0nm
    """
    wavelengths = np.asarray(wavelengths)

    if wavelengths.ndim != 1 or len(wavelengths) == 0:
        raise ValueError("Wavelengths must be non-empty 1D array")

    if not np.isfinite(target_nm):
        raise ValueError(f"Target wavelength must be finite, got {target_nm}")

    differences = np.abs(wavelengths - target_nm)
    idx = int(np.argmin(differences))
    actual_nm = float(wavelengths[idx])

    return idx, actual_nm


def extract_multiple_bands(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    target_wavelengths: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract multiple bands by wavelength.

    Convenience function to extract several bands at once.

    Args:
        cube: Hyperspectral cube
        wavelengths: Wavelength array
        target_wavelengths: List of target wavelengths in nm

    Returns:
        Tuple of:
            - bands: Array with shape (H, W, N) where N = len(target_wavelengths)
            - indices: Array of selected band indices
            - actual_wavelengths: Array of actual wavelengths selected

    Example:
        >>> targets = [450, 550, 650]  # Blue, Green, Red-ish
        >>> bands, indices, actuals = extract_multiple_bands(cube, wl, targets)
        >>> print(bands.shape)
        (512, 512, 3)
    """
    if not target_wavelengths:
        raise ValueError("target_wavelengths cannot be empty")

    bands_list = []
    indices_list = []
    actuals_list = []

    for target_nm in target_wavelengths:
        band, idx, actual = get_band_by_wavelength(cube, wavelengths, target_nm)
        bands_list.append(band)
        indices_list.append(idx)
        actuals_list.append(actual)

    # Stack bands along third dimension
    bands = np.stack(bands_list, axis=2)
    indices = np.array(indices_list, dtype=np.int32)
    actuals = np.array(actuals_list, dtype=np.float32)

    return bands, indices, actuals
