"""Preprocessing module

Provides spectral preprocessing utilities for hyperspectral data.
"""

from .spectral import (
    snv_normalize,
    select_wavelength_range,
    l2_normalize,
    select_bands,
    apply_preprocessing,
    filter_wavelengths,  # Legacy alias
    SpectralPreprocessor,  # Legacy class
    extract_rgb_from_spectral,
)

__all__ = [
    # Core functions
    "snv_normalize",
    "select_wavelength_range",
    "l2_normalize",
    "select_bands",
    "apply_preprocessing",
    # Legacy
    "filter_wavelengths",
    "SpectralPreprocessor",
    "extract_rgb_from_spectral",
]
