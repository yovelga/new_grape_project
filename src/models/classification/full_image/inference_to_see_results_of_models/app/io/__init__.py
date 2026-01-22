"""Input/Output module for various file formats"""

from .envi import ENVIReader
from .rgb import (
    RGBReader,
    find_rgb_image,
    load_rgb,
    find_hsi_rgb,
    find_camera_rgb,
    find_both_rgb_images,
    load_both_rgb_images,
    find_canon_rgb,
)
from .hsi_band import (
    get_band_by_index,
    get_band_by_wavelength,
    find_nearest_band_index,
    extract_multiple_bands,
)

__all__ = [
    # ENVI reader
    "ENVIReader",
    # RGB utilities
    "RGBReader",
    "find_rgb_image",
    "load_rgb",
    "find_hsi_rgb",
    "find_camera_rgb",
    "find_both_rgb_images",
    "load_both_rgb_images",
    "find_canon_rgb",
    # HSI band extraction
    "get_band_by_index",
    "get_band_by_wavelength",
    "find_nearest_band_index",
    "extract_multiple_bands",
]


