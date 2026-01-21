"""
Spectral preprocessing utilities for hyperspectral data.

Provides clean, vectorized implementations of:
- SNV (Standard Normal Variate) normalization
- Wavelength range selection
- L2 normalization
- Full preprocessing pipeline

All functions are designed to be numerically stable and memory efficient.
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.types import PreprocessConfig


# ============================================================================
# Core Preprocessing Functions
# ============================================================================

def snv_normalize(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Apply Standard Normal Variate (SNV) normalization to spectral data.

    SNV centers each spectrum by its mean and scales by its standard deviation.
    This removes multiplicative scatter effects common in spectroscopy.

    Formula: X_snv[i] = (X[i] - mean(X[i])) / std(X[i])

    Args:
        X: Input spectra with shape:
            - (N, C): N samples, C spectral bands
            - (H, W, C): Spatial image with C spectral bands
            - (..., C): Any shape where last dimension is spectral
        eps: Small value added to std to prevent division by zero

    Returns:
        SNV-normalized spectra with same shape and dtype as input

    Note:
        - Each spectrum (row/pixel) is normalized independently
        - Spectra with near-zero std will be scaled by eps, not set to zero
        - Vectorized implementation for efficiency

    Example:
        >>> X = np.random.rand(100, 224)  # 100 pixels, 224 bands
        >>> X_snv = snv_normalize(X)
        >>> assert X_snv.shape == X.shape
    """
    # Ensure float type for numerical stability
    X = np.asarray(X, dtype=np.float64)
    original_shape = X.shape
    original_dtype = X.dtype

    # Flatten to 2D: (N_pixels, C_bands)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])

    # Compute mean and std along spectral axis (axis=1)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)

    # Prevent division by zero with eps
    std = np.maximum(std, eps)

    # Apply SNV normalization
    X_snv = (X - mean) / std

    # Restore original shape
    X_snv = X_snv.reshape(original_shape)

    return X_snv.astype(np.float32)


def select_wavelength_range(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select wavelength range from hyperspectral cube.

    Filters the spectral dimension to include only bands within the
    specified wavelength range [wl_min, wl_max].

    Args:
        cube: Hyperspectral data with shape:
            - (H, W, C): Spatial cube with C spectral bands
            - (N, C): Flattened pixels with C spectral bands
        wavelengths: 1D array of wavelength values (nm) corresponding to bands.
            Must have length equal to cube's last dimension.
        wl_min: Minimum wavelength (nm), inclusive. None = no lower bound.
        wl_max: Maximum wavelength (nm), inclusive. None = no upper bound.

    Returns:
        Tuple of:
            - cube_filtered: Filtered cube with selected bands
            - wavelengths_filtered: Corresponding wavelength values

    Raises:
        ValueError: If wavelengths length doesn't match cube bands
        ValueError: If no wavelengths fall within the specified range
        ValueError: If wl_min >= wl_max

    Example:
        >>> cube = np.random.rand(512, 512, 224)
        >>> wavelengths = np.linspace(400, 1000, 224)
        >>> cube_filt, wl_filt = select_wavelength_range(cube, wavelengths, 450, 925)
    """
    # Input validation
    wavelengths = np.asarray(wavelengths).flatten()

    if len(wavelengths) != cube.shape[-1]:
        raise ValueError(
            f"wavelengths length ({len(wavelengths)}) must match cube's "
            f"last dimension ({cube.shape[-1]})"
        )

    if wl_min is not None and wl_max is not None and wl_min >= wl_max:
        raise ValueError(f"wl_min ({wl_min}) must be less than wl_max ({wl_max})")

    # Build mask for valid wavelengths
    mask = np.ones(len(wavelengths), dtype=bool)

    if wl_min is not None:
        mask &= (wavelengths >= wl_min)

    if wl_max is not None:
        mask &= (wavelengths <= wl_max)

    # Get valid indices
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No wavelengths found in range [{wl_min}, {wl_max}]. "
            f"Available range: [{wavelengths.min():.1f}, {wavelengths.max():.1f}]"
        )

    # Filter cube and wavelengths
    cube_filtered = cube[..., valid_indices]
    wavelengths_filtered = wavelengths[valid_indices]

    return cube_filtered, wavelengths_filtered


def l2_normalize(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Apply L2 (Euclidean) normalization to spectral data.

    Normalizes each spectrum to unit length.

    Formula: X_norm[i] = X[i] / ||X[i]||_2

    Args:
        X: Input spectra with shape (..., C) where C is spectral bands
        eps: Small value to prevent division by zero

    Returns:
        L2-normalized spectra with same shape

    Example:
        >>> X = np.random.rand(100, 224)
        >>> X_norm = l2_normalize(X)
        >>> norms = np.linalg.norm(X_norm, axis=-1)
        >>> assert np.allclose(norms, 1.0)
    """
    X = np.asarray(X, dtype=np.float64)

    # Compute L2 norm along spectral axis
    norms = np.linalg.norm(X, axis=-1, keepdims=True)

    # Prevent division by zero
    norms = np.maximum(norms, eps)

    return (X / norms).astype(np.float32)


def select_bands(cube: np.ndarray, band_indices: np.ndarray) -> np.ndarray:
    """
    Select specific bands from hyperspectral cube.

    Args:
        cube: Hyperspectral data with shape (..., C)
        band_indices: 1D array of band indices to select

    Returns:
        Filtered cube with only selected bands

    Raises:
        IndexError: If any band index is out of range
    """
    band_indices = np.asarray(band_indices).flatten()

    # Validate indices
    max_band = cube.shape[-1] - 1
    if np.any(band_indices < 0) or np.any(band_indices > max_band):
        raise IndexError(
            f"Band indices must be in range [0, {max_band}], "
            f"got min={band_indices.min()}, max={band_indices.max()}"
        )

    return cube[..., band_indices]


# ============================================================================
# Preprocessing Pipeline
# ============================================================================

def apply_preprocessing(
    cube: np.ndarray,
    config: "PreprocessConfig"
) -> np.ndarray:
    """
    Apply full preprocessing pipeline based on configuration.

    Applies in order:
    1. Band selection (if configured)
    2. SNV normalization (if configured)
    3. L2 normalization (if configured)

    Args:
        cube: Hyperspectral data with shape (H, W, C) or (N, C)
        config: PreprocessConfig specifying preprocessing steps

    Returns:
        Preprocessed cube with same spatial shape, possibly fewer bands

    Note:
        Wavelength filtering must be done separately if wavelengths array
        is not stored in the config.
    """
    # Band selection
    if config.needs_band_selection:
        cube = select_bands(cube, config.band_indices)
    elif config.needs_wavelength_filter and config.wavelengths is not None:
        cube, _ = select_wavelength_range(
            cube,
            config.wavelengths,
            config.wl_min,
            config.wl_max
        )

    # SNV normalization
    if config.use_snv:
        cube = snv_normalize(cube)

    # L2 normalization
    if config.use_l2_norm:
        cube = l2_normalize(cube)

    return cube


# ============================================================================
# Legacy Functions (Backward Compatibility)
# ============================================================================

def filter_wavelengths(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    min_wl: float = 450.0,
    max_wl: float = 925.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter hyperspectral cube to specific wavelength range.

    DEPRECATED: Use select_wavelength_range() instead.

    Args:
        cube: Hyperspectral cube (H, W, C)
        wavelengths: Array of wavelength values in nm
        min_wl: Minimum wavelength (nm)
        max_wl: Maximum wavelength (nm)

    Returns:
        Tuple of (filtered_cube, filtered_wavelengths)
    """
    return select_wavelength_range(cube, wavelengths, min_wl, max_wl)


class SpectralPreprocessor:
    """
    Preprocessing pipeline for spectral images.

    DEPRECATED: Use apply_preprocessing() with PreprocessConfig instead.
    """

    def __init__(self,
                 normalize: bool = True,
                 remove_bands: Optional[list] = None,
                 apply_savgol: bool = False):
        """
        Initialize preprocessor.

        Args:
            normalize: Whether to normalize data
            remove_bands: Indices of bands to remove
            apply_savgol: Whether to apply Savitzky-Golay filter
        """
        self.normalize = normalize
        self.remove_bands = remove_bands or []
        self.apply_savgol = apply_savgol

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline."""
        # Remove specified bands
        if self.remove_bands:
            valid_bands = [i for i in range(image.shape[2]) if i not in self.remove_bands]
            image = image[:, :, valid_bands]

        # Normalize
        if self.normalize:
            image = l2_normalize(image)

        # Apply Savitzky-Golay smoothing
        if self.apply_savgol:
            image = self._savgol_filter(image)

        return image

    def _savgol_filter(self, image: np.ndarray,
                       window_length: int = 11,
                       polyorder: int = 2) -> np.ndarray:
        """Apply Savitzky-Golay filter along spectral dimension."""
        try:
            from scipy.signal import savgol_filter

            h, w, c = image.shape
            image_flat = image.reshape(-1, c)

            filtered = np.apply_along_axis(
                lambda x: savgol_filter(x, window_length, polyorder),
                axis=1,
                arr=image_flat
            )

            return filtered.reshape(h, w, c)
        except ImportError:
            return image


def extract_rgb_from_spectral(
    image: np.ndarray,
    rgb_bands: Tuple[int, int, int] = (0, 1, 2)
) -> np.ndarray:
    """
    Extract RGB representation from spectral image.

    Args:
        image: Spectral image (H, W, C)
        rgb_bands: Indices for R, G, B bands

    Returns:
        RGB image (H, W, 3) as uint8
    """
    r_idx, g_idx, b_idx = rgb_bands
    rgb = np.stack([
        image[:, :, r_idx],
        image[:, :, g_idx],
        image[:, :, b_idx]
    ], axis=2)

    # Normalize to 0-255
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max > rgb_min:
        rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
    else:
        rgb = np.zeros_like(rgb, dtype=np.uint8)

    return rgb


# ============================================================================
# Debug / Sanity Check Utilities
# ============================================================================

def _run_sanity_checks() -> bool:
    """
    Run sanity checks on preprocessing functions.

    Returns:
        True if all checks pass

    Raises:
        AssertionError: If any check fails
    """
    print("Running spectral preprocessing sanity checks...")

    # Test 1: SNV normalization on 2D array
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                  [10.0, 20.0, 30.0, 40.0, 50.0]])
    X_snv = snv_normalize(X)

    # After SNV, each row should have mean ≈ 0 and std ≈ 1
    row_means = np.mean(X_snv, axis=1)
    row_stds = np.std(X_snv, axis=1)

    assert np.allclose(row_means, 0, atol=1e-6), f"SNV mean not 0: {row_means}"
    assert np.allclose(row_stds, 1, atol=1e-6), f"SNV std not 1: {row_stds}"
    print("  ✓ SNV normalization produces mean=0, std=1")

    # Test 2: SNV on 3D cube
    cube = np.random.rand(10, 10, 50).astype(np.float32)
    cube_snv = snv_normalize(cube)
    assert cube_snv.shape == cube.shape
    assert cube_snv.dtype == np.float32
    print("  ✓ SNV works on 3D cube, preserves shape and dtype")

    # Test 3: Wavelength selection
    wavelengths = np.linspace(400, 1000, 224)
    cube = np.random.rand(10, 10, 224)

    cube_filt, wl_filt = select_wavelength_range(cube, wavelengths, 450, 925)

    assert wl_filt.min() >= 450
    assert wl_filt.max() <= 925
    assert cube_filt.shape[-1] == len(wl_filt)
    print(f"  ✓ Wavelength selection: 224 bands -> {len(wl_filt)} bands")

    # Test 4: Wavelength selection edge case - no lower bound
    cube_filt2, wl_filt2 = select_wavelength_range(cube, wavelengths, None, 600)
    assert wl_filt2.max() <= 600
    print("  ✓ Wavelength selection with no lower bound works")

    # Test 5: L2 normalization
    X = np.random.rand(100, 50)
    X_l2 = l2_normalize(X)
    norms = np.linalg.norm(X_l2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    print("  ✓ L2 normalization produces unit vectors")

    # Test 6: Band selection
    cube = np.random.rand(10, 10, 100)
    indices = np.array([0, 10, 20, 50, 99])
    cube_sel = select_bands(cube, indices)
    assert cube_sel.shape == (10, 10, 5)
    print("  ✓ Band selection works correctly")

    # Test 7: Error handling - wavelength mismatch
    try:
        select_wavelength_range(cube, np.linspace(400, 1000, 50), 450, 925)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must match" in str(e)
        print("  ✓ Wavelength mismatch error is caught")

    # Test 8: Error handling - no valid wavelengths
    try:
        select_wavelength_range(
            np.random.rand(10, 10, 50),
            np.linspace(400, 500, 50),  # wavelengths 400-500
            600, 700  # filter for 600-700
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "No wavelengths found" in str(e)
        print("  ✓ No valid wavelengths error is caught")

    # Test 9: Numerical stability - zero std spectrum
    X_zero_var = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])  # constant spectrum
    X_snv_zero = snv_normalize(X_zero_var)
    assert not np.any(np.isnan(X_snv_zero))
    assert not np.any(np.isinf(X_snv_zero))
    print("  ✓ SNV handles zero-variance spectra without NaN/Inf")

    print("\n✅ All spectral preprocessing sanity checks passed!")
    return True


if __name__ == "__main__":
    # Run sanity checks when module is executed directly
    _run_sanity_checks()


