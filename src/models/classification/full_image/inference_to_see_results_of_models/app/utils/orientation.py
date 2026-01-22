"""
Image orientation utilities for HSI cube and display alignment.

Provides functions to:
- Ensure HSI cube is in (H, W, C) format
- Apply rotation and flip transformations for display
- Keep masks/overlays aligned with transformed images
"""

import numpy as np
from typing import Literal, Tuple


def ensure_hwc(cube: np.ndarray) -> np.ndarray:
    """
    Ensure HSI cube is in (H, W, C) format.

    Detects common formats and converts to Height x Width x Channels.

    Args:
        cube: Input array, expected 3D

    Returns:
        Array in (H, W, C) format

    Raises:
        ValueError: If array is not 3D

    Detection heuristics:
    - (H, W, C): W and H are typically larger than C (spectral bands ~100-300)
    - (C, H, W): First dim is small (channels), common in some ENVI readers
    - (W, H, C): Uncommon but possible if width > height
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D array, got {cube.ndim}D")

    d0, d1, d2 = cube.shape

    # Heuristic: spectral bands are usually in range [50, 500]
    # Image dimensions are usually > 100
    # The smallest dimension is likely the channel/band dimension

    dims = [(d0, 0), (d1, 1), (d2, 2)]
    dims_sorted = sorted(dims, key=lambda x: x[0])

    smallest_dim, smallest_idx = dims_sorted[0]

    # If smallest dimension is already last (axis=2), assume (H, W, C)
    if smallest_idx == 2:
        # Already (H, W, C) - most common case
        return cube

    # If smallest dimension is first (axis=0), assume (C, H, W)
    if smallest_idx == 0:
        # Transpose from (C, H, W) to (H, W, C)
        return np.transpose(cube, (1, 2, 0))

    # If smallest dimension is middle (axis=1), unusual case
    # Could be (H, C, W) - transpose to (H, W, C)
    if smallest_idx == 1:
        return np.transpose(cube, (0, 2, 1))

    # Fallback - return as-is
    return cube


def apply_display_transform(
    image: np.ndarray,
    rotate_deg: Literal[0, 90, 180, 270] = 0,
    flip_h: bool = False,
    flip_v: bool = False
) -> np.ndarray:
    """
    Apply rotation and flip transformations for display.

    Args:
        image: 2D (H, W) or 3D (H, W, C) array
        rotate_deg: Rotation in degrees (clockwise), must be 0, 90, 180, or 270
        flip_h: Flip horizontally (left-right)
        flip_v: Flip vertically (up-down)

    Returns:
        Transformed image array
    """
    if image is None or image.size == 0:
        return image

    result = image.copy()

    # Apply rotation (clockwise)
    if rotate_deg == 90:
        result = np.rot90(result, k=-1)  # k=-1 for clockwise 90
    elif rotate_deg == 180:
        result = np.rot90(result, k=2)
    elif rotate_deg == 270:
        result = np.rot90(result, k=1)  # k=1 for counter-clockwise 90 = clockwise 270

    # Apply flips
    if flip_h:
        result = np.fliplr(result)
    if flip_v:
        result = np.flipud(result)

    return result


def get_transformed_shape(
    original_shape: Tuple[int, int],
    rotate_deg: Literal[0, 90, 180, 270] = 0
) -> Tuple[int, int]:
    """
    Get the shape after rotation.

    Args:
        original_shape: (H, W) tuple
        rotate_deg: Rotation in degrees

    Returns:
        (new_H, new_W) after rotation
    """
    h, w = original_shape
    if rotate_deg in [90, 270]:
        return (w, h)
    return (h, w)


class DisplayTransform:
    """
    Encapsulates display transformation settings.

    Usage:
        transform = DisplayTransform(rotate_deg=90, flip_h=True)
        transformed = transform.apply(image)
    """

    def __init__(
        self,
        rotate_deg: Literal[0, 90, 180, 270] = 0,
        flip_h: bool = False,
        flip_v: bool = False
    ):
        """Initialize transformation settings."""
        self.rotate_deg = rotate_deg
        self.flip_h = flip_h
        self.flip_v = flip_v

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply transformation to image."""
        return apply_display_transform(
            image,
            rotate_deg=self.rotate_deg,
            flip_h=self.flip_h,
            flip_v=self.flip_v
        )

    def get_transformed_shape(self, original_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get shape after transformation."""
        return get_transformed_shape(original_shape, self.rotate_deg)

    def __repr__(self) -> str:
        return f"DisplayTransform(rotate={self.rotate_deg}°, flip_h={self.flip_h}, flip_v={self.flip_v})"


def validate_2d_for_display(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Validate and ensure array is 2D for display.

    Args:
        arr: Input array
        name: Name for error messages

    Returns:
        2D array (H, W)

    Raises:
        ValueError: If array cannot be converted to 2D
    """
    if arr is None:
        raise ValueError(f"{name} is None")

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[2] == 1:
        # Single channel 3D array - squeeze to 2D
        return arr[:, :, 0]

    raise ValueError(f"{name} must be 2D, got shape {arr.shape}")


def contiguous_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is C-contiguous for QImage compatibility.

    Args:
        arr: Input array

    Returns:
        C-contiguous copy if needed
    """
    if arr is None:
        return arr
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr)
    return arr
