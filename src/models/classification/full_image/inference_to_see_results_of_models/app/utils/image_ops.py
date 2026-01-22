"""
Image processing utilities for normalization and colormap application.

Lightweight helpers for preparing images for display.
"""

import numpy as np
from typing import Literal


def normalize_to_uint8(
    img2d: np.ndarray,
    method: Literal["percentile", "minmax", "std"] = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0
) -> np.ndarray:
    """
    Normalize a 2D image array to uint8 range [0, 255].

    Args:
        img2d: Input 2D array (grayscale image)
        method: Normalization method:
            - "percentile": Clip to percentile range (robust to outliers)
            - "minmax": Linear min-max scaling
            - "std": Center on mean, clip at ±3 standard deviations
        p_low: Lower percentile for "percentile" method (default: 2)
        p_high: Upper percentile for "percentile" method (default: 98)

    Returns:
        uint8 array with values in [0, 255]

    Example:
        >>> img_norm = normalize_to_uint8(raw_image, method="percentile")
    """
    if img2d.size == 0:
        return np.zeros_like(img2d, dtype=np.uint8)

    img_float = img2d.astype(np.float32)

    if method == "percentile":
        vmin = np.percentile(img_float, p_low)
        vmax = np.percentile(img_float, p_high)
    elif method == "minmax":
        vmin = img_float.min()
        vmax = img_float.max()
    elif method == "std":
        mean = img_float.mean()
        std = img_float.std()
        vmin = mean - 3 * std
        vmax = mean + 3 * std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Avoid division by zero
    if vmax - vmin < 1e-8:
        return np.full_like(img2d, 127, dtype=np.uint8)

    # Normalize to [0, 1]
    img_norm = np.clip((img_float - vmin) / (vmax - vmin), 0, 1)

    # Scale to [0, 255]
    return (img_norm * 255).astype(np.uint8)


def apply_colormap(
    img2d_float01: np.ndarray,
    name: Literal["viridis", "jet", "hot", "cool", "gray"] = "viridis"
) -> np.ndarray:
    """
    Apply a colormap to a normalized 2D float array.

    Args:
        img2d_float01: Input 2D array with values in [0, 1]
        name: Colormap name:
            - "viridis": Perceptually uniform, good default
            - "jet": Classic rainbow colormap
            - "hot": Black-red-yellow-white
            - "cool": Cyan-magenta
            - "gray": Grayscale

    Returns:
        RGB uint8 array with shape (H, W, 3)

    Example:
        >>> img_norm = img / img.max()  # Normalize to [0, 1]
        >>> img_rgb = apply_colormap(img_norm, name="viridis")

    Note:
        Uses simple built-in colormaps to avoid matplotlib dependency.
        For more advanced colormaps, consider matplotlib.cm.
    """
    if img2d_float01.size == 0:
        return np.zeros((*img2d_float01.shape, 3), dtype=np.uint8)

    # Clip to [0, 1] range
    img = np.clip(img2d_float01, 0, 1)

    h, w = img.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if name == "gray":
        # Simple grayscale
        gray_val = (img * 255).astype(np.uint8)
        rgb[..., 0] = gray_val
        rgb[..., 1] = gray_val
        rgb[..., 2] = gray_val

    elif name == "jet":
        # Classic jet colormap approximation
        # Blue -> Cyan -> Green -> Yellow -> Red
        scaled = img * 255

        # Red channel
        rgb[..., 0] = np.clip(
            np.where(scaled < 96, 0,
            np.where(scaled < 160, (scaled - 96) * 4,
            np.where(scaled < 224, 255, 255 - (scaled - 224) * 4))),
            0, 255
        ).astype(np.uint8)

        # Green channel
        rgb[..., 1] = np.clip(
            np.where(scaled < 32, 0,
            np.where(scaled < 96, (scaled - 32) * 4,
            np.where(scaled < 160, 255,
            np.where(scaled < 224, 255 - (scaled - 160) * 4, 0)))),
            0, 255
        ).astype(np.uint8)

        # Blue channel
        rgb[..., 2] = np.clip(
            np.where(scaled < 32, 127 + scaled * 2,
            np.where(scaled < 96, 255 - (scaled - 32) * 4, 0)),
            0, 255
        ).astype(np.uint8)

    elif name == "hot":
        # Black -> Red -> Yellow -> White
        scaled = img * 255

        rgb[..., 0] = np.clip(scaled * 1.5, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip((scaled - 85) * 1.5, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip((scaled - 170) * 1.5, 0, 255).astype(np.uint8)

    elif name == "cool":
        # Cyan -> Magenta
        val = (img * 255).astype(np.uint8)
        rgb[..., 0] = val  # Red increases
        rgb[..., 1] = 255 - val  # Green decreases
        rgb[..., 2] = 255  # Blue constant

    elif name == "viridis":
        # Viridis approximation (purple -> green -> yellow)
        # Simplified version of matplotlib's viridis
        scaled = img * 255

        # Red channel
        rgb[..., 0] = np.clip(
            np.where(scaled < 128, scaled * 0.3,
                     (scaled - 128) * 1.7 + 38),
            0, 255
        ).astype(np.uint8)

        # Green channel
        rgb[..., 1] = np.clip(
            np.where(scaled < 64, scaled * 0.5,
            np.where(scaled < 192, 32 + (scaled - 64) * 1.5,
                     224 - (scaled - 192) * 0.5)),
            0, 255
        ).astype(np.uint8)

        # Blue channel
        rgb[..., 2] = np.clip(
            np.where(scaled < 128, 85 + scaled * 0.8,
                     187 - (scaled - 128) * 1.3),
            0, 255
        ).astype(np.uint8)

    else:
        raise ValueError(f"Unknown colormap: {name}")

    return rgb
