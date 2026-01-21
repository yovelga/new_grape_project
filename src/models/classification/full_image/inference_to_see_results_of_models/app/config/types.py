"""
Configuration types for the inference application.

Centralized dataclass definitions for preprocessing, model, and inference configuration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:
    """
    Configuration for spectral preprocessing.

    Controls wavelength filtering and normalization applied to hyperspectral data
    before model inference.

    Attributes:
        use_snv: Whether to apply Standard Normal Variate (SNV) normalization.
            SNV centers each spectrum and scales by its standard deviation,
            removing multiplicative scatter effects.
        wl_min: Minimum wavelength in nm for filtering. If None, no lower bound.
        wl_max: Maximum wavelength in nm for filtering. If None, no upper bound.
        wavelengths: Array of wavelength values corresponding to spectral bands.
            Required if wl_min or wl_max is set and filtering by wavelength.
        band_indices: Explicit band indices to select (alternative to wavelength filtering).
            If provided, wavelength filtering is ignored.
        use_l2_norm: Whether to apply per-pixel L2 normalization after other preprocessing.

    Example:
        >>> config = PreprocessConfig(use_snv=True, wl_min=450.0, wl_max=925.0)
        >>> config.validate()  # Raises if invalid
    """
    use_snv: bool = True
    wl_min: Optional[float] = None
    wl_max: Optional[float] = None
    wavelengths: Optional[np.ndarray] = None
    band_indices: Optional[np.ndarray] = None
    use_l2_norm: bool = False

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check wavelength range consistency
        if self.wl_min is not None and self.wl_max is not None:
            if self.wl_min >= self.wl_max:
                raise ValueError(
                    f"wl_min ({self.wl_min}) must be less than wl_max ({self.wl_max})"
                )

        # Check band_indices
        if self.band_indices is not None:
            if not isinstance(self.band_indices, np.ndarray):
                raise TypeError(
                    f"band_indices must be numpy array, got {type(self.band_indices).__name__}"
                )
            if len(self.band_indices) == 0:
                raise ValueError("band_indices cannot be empty")

        # Check wavelengths array if wavelength filtering is requested
        if (self.wl_min is not None or self.wl_max is not None) and self.band_indices is None:
            if self.wavelengths is None:
                raise ValueError(
                    "wavelengths array required when wl_min or wl_max is set "
                    "(unless band_indices is provided)"
                )

        # Check wavelengths array type
        if self.wavelengths is not None:
            if not isinstance(self.wavelengths, np.ndarray):
                raise TypeError(
                    f"wavelengths must be numpy array, got {type(self.wavelengths).__name__}"
                )
            if len(self.wavelengths) == 0:
                raise ValueError("wavelengths cannot be empty")

    @property
    def needs_wavelength_filter(self) -> bool:
        """Check if wavelength filtering is configured."""
        return (
            self.band_indices is None and
            (self.wl_min is not None or self.wl_max is not None)
        )

    @property
    def needs_band_selection(self) -> bool:
        """Check if explicit band selection is configured."""
        return self.band_indices is not None

    def __repr__(self) -> str:
        parts = [f"use_snv={self.use_snv}"]
        if self.wl_min is not None:
            parts.append(f"wl_min={self.wl_min}")
        if self.wl_max is not None:
            parts.append(f"wl_max={self.wl_max}")
        if self.band_indices is not None:
            parts.append(f"band_indices=[{len(self.band_indices)} bands]")
        if self.use_l2_norm:
            parts.append("use_l2_norm=True")
        return f"PreprocessConfig({', '.join(parts)})"


@dataclass
class InferenceConfig:
    """
    Configuration for model inference.

    Attributes:
        chunk_size: Number of pixels to process per batch during chunked inference.
        target_class_index: Class index for probability output.
            For binary models, None defaults to positive class (index 1).
        device: Device for PyTorch models ('cpu', 'cuda', 'mps').
    """
    chunk_size: int = 200_000
    target_class_index: Optional[int] = None
    device: str = "cpu"

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.target_class_index is not None and self.target_class_index < 0:
            raise ValueError(
                f"target_class_index must be >= 0, got {self.target_class_index}"
            )
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(
                f"device must be 'cpu', 'cuda', or 'mps', got '{self.device}'"
            )


# ============================================================================
# Debug / Sanity Check Utilities
# ============================================================================

def _run_config_sanity_checks() -> bool:
    """
    Run sanity checks on configuration types.

    Returns:
        True if all checks pass

    Raises:
        AssertionError: If any check fails
    """
    print("Running PreprocessConfig sanity checks...")

    # Test 1: Default config should be valid (no wavelength filtering without wavelengths)
    cfg = PreprocessConfig(use_snv=True)
    cfg.validate()
    print("  ✓ Default config (SNV only) is valid")

    # Test 2: Wavelength filtering requires wavelengths array
    cfg_wl = PreprocessConfig(use_snv=True, wl_min=450.0, wl_max=925.0)
    try:
        cfg_wl.validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "wavelengths array required" in str(e)
        print("  ✓ Wavelength filtering requires wavelengths array")

    # Test 3: With wavelengths provided, should be valid
    wavelengths = np.linspace(400, 1000, 224)
    cfg_wl_valid = PreprocessConfig(
        use_snv=True,
        wl_min=450.0,
        wl_max=925.0,
        wavelengths=wavelengths
    )
    cfg_wl_valid.validate()
    assert cfg_wl_valid.needs_wavelength_filter
    print("  ✓ Wavelength filtering with wavelengths array is valid")

    # Test 4: Band indices should work
    cfg_bands = PreprocessConfig(
        use_snv=True,
        band_indices=np.array([0, 10, 20, 30])
    )
    cfg_bands.validate()
    assert cfg_bands.needs_band_selection
    assert not cfg_bands.needs_wavelength_filter
    print("  ✓ Band indices config is valid")

    # Test 5: Invalid wl_min > wl_max
    cfg_invalid = PreprocessConfig(wl_min=900.0, wl_max=450.0, wavelengths=wavelengths)
    try:
        cfg_invalid.validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "wl_min" in str(e) and "wl_max" in str(e)
        print("  ✓ Invalid wl_min > wl_max is caught")

    # Test 6: InferenceConfig validation
    print("\nRunning InferenceConfig sanity checks...")
    inf_cfg = InferenceConfig(chunk_size=100_000, device="cpu")
    inf_cfg.validate()
    print("  ✓ Default InferenceConfig is valid")

    # Test 7: Invalid chunk_size
    try:
        InferenceConfig(chunk_size=0).validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Invalid chunk_size is caught")

    print("\n✅ All sanity checks passed!")
    return True


if __name__ == "__main__":
    # Run sanity checks when module is executed directly
    _run_config_sanity_checks()
