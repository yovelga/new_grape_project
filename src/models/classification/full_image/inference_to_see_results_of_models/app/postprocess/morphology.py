"""
Morphological operations for binary masks.

Provides morphological closing and other operations with graceful fallback:
- Prefers OpenCV (cv2) for performance
- Falls back to scipy.ndimage if cv2 not available
- Raises friendly error if neither is available
"""

import numpy as np

# Try to import image processing libraries
_CV2_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore

try:
    from scipy import ndimage
    _SCIPY_AVAILABLE = True
except ImportError:
    ndimage = None  # type: ignore


def _check_backend_available() -> str:
    """
    Check which backend is available.

    Returns:
        Name of available backend ('cv2', 'scipy', or raises error)
    """
    if _CV2_AVAILABLE:
        return "cv2"
    elif _SCIPY_AVAILABLE:
        return "scipy"
    else:
        raise ImportError(
            "Morphological operations require either OpenCV (cv2) or scipy.\n"
            "Install one with:\n"
            "  pip install opencv-python\n"
            "  or\n"
            "  pip install scipy"
        )


def morphological_close(mask: np.ndarray, ksize: int) -> np.ndarray:
    """
    Apply morphological closing to a binary mask.

    Closing = dilation followed by erosion. Fills small holes while
    preserving the overall shape.

    Args:
        mask: Binary mask (H, W) with dtype bool or uint8
        ksize: Kernel size for structuring element.
               Must be odd positive integer. 0 or negative disables operation.

    Returns:
        Closed binary mask with same shape and dtype as input

    Raises:
        ImportError: If neither cv2 nor scipy is available
        ValueError: If ksize is even
    """
    # Validate input
    if ksize <= 0:
        return mask  # Disabled

    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D")

    original_dtype = mask.dtype
    backend = _check_backend_available()

    if backend == "cv2":
        # Convert to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)

        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Apply closing
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # Convert back
        if original_dtype == bool:
            return closed > 0
        return closed

    else:  # scipy
        # Convert to bool for scipy
        mask_bool = mask.astype(bool)

        # Create structuring element (disk approximation)
        struct = _create_disk_structuring_element(ksize)

        # Closing = dilation then erosion
        dilated = ndimage.binary_dilation(mask_bool, structure=struct)
        closed = ndimage.binary_erosion(dilated, structure=struct)

        if original_dtype == bool:
            return closed
        return closed.astype(np.uint8) * 255


def morphological_open(mask: np.ndarray, ksize: int) -> np.ndarray:
    """
    Apply morphological opening to a binary mask.

    Opening = erosion followed by dilation. Removes small bright spots
    (noise) while preserving overall shape.

    Args:
        mask: Binary mask (H, W)
        ksize: Kernel size (odd positive integer). 0 or negative disables.

    Returns:
        Opened binary mask
    """
    if ksize <= 0:
        return mask

    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D")

    original_dtype = mask.dtype
    backend = _check_backend_available()

    if backend == "cv2":
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        if original_dtype == bool:
            return opened > 0
        return opened

    else:  # scipy
        mask_bool = mask.astype(bool)
        struct = _create_disk_structuring_element(ksize)

        eroded = ndimage.binary_erosion(mask_bool, structure=struct)
        opened = ndimage.binary_dilation(eroded, structure=struct)

        if original_dtype == bool:
            return opened
        return opened.astype(np.uint8) * 255


def dilate(mask: np.ndarray, ksize: int) -> np.ndarray:
    """
    Apply morphological dilation to a binary mask.

    Args:
        mask: Binary mask (H, W)
        ksize: Kernel size (odd positive integer)

    Returns:
        Dilated mask
    """
    if ksize <= 0:
        return mask

    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")

    original_dtype = mask.dtype
    backend = _check_backend_available()

    if backend == "cv2":
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(mask_uint8, kernel)

        if original_dtype == bool:
            return dilated > 0
        return dilated

    else:  # scipy
        mask_bool = mask.astype(bool)
        struct = _create_disk_structuring_element(ksize)
        dilated = ndimage.binary_dilation(mask_bool, structure=struct)

        if original_dtype == bool:
            return dilated
        return dilated.astype(np.uint8) * 255


def erode(mask: np.ndarray, ksize: int) -> np.ndarray:
    """
    Apply morphological erosion to a binary mask.

    Args:
        mask: Binary mask (H, W)
        ksize: Kernel size (odd positive integer)

    Returns:
        Eroded mask
    """
    if ksize <= 0:
        return mask

    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")

    original_dtype = mask.dtype
    backend = _check_backend_available()

    if backend == "cv2":
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        eroded = cv2.erode(mask_uint8, kernel)

        if original_dtype == bool:
            return eroded > 0
        return eroded

    else:  # scipy
        mask_bool = mask.astype(bool)
        struct = _create_disk_structuring_element(ksize)
        eroded = ndimage.binary_erosion(mask_bool, structure=struct)

        if original_dtype == bool:
            return eroded
        return eroded.astype(np.uint8) * 255


def _create_disk_structuring_element(size: int) -> np.ndarray:
    """
    Create a disk-shaped structuring element.

    Args:
        size: Diameter of the disk (must be odd)

    Returns:
        Boolean array representing the disk
    """
    radius = size // 2
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (x**2 + y**2) <= radius**2


def get_backend_info() -> dict:
    """
    Get information about available backends.

    Returns:
        Dict with backend availability info
    """
    return {
        "cv2_available": _CV2_AVAILABLE,
        "scipy_available": _SCIPY_AVAILABLE,
        "active_backend": "cv2" if _CV2_AVAILABLE else ("scipy" if _SCIPY_AVAILABLE else None),
    }


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on morphological operations."""
    print("Running morphology sanity checks...")

    backend_info = get_backend_info()
    print(f"  Backend info: {backend_info}")

    if backend_info["active_backend"] is None:
        print("  ⚠ No backend available, skipping tests")
        return True

    # Test 1: Basic closing fills small holes
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[9:11, 9:11] = False  # Small hole in center

    closed = morphological_close(mask, ksize=5)

    # Hole should be filled
    assert closed[9, 9] == True, "Closing should fill small holes"
    assert closed[10, 10] == True, "Closing should fill small holes"
    print("  ✓ Closing fills small holes")

    # Test 2: Opening removes small noise
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True  # Main blob
    mask[2, 2] = True  # Small noise pixel

    opened = morphological_open(mask, ksize=3)

    assert opened[2, 2] == False, "Opening should remove isolated pixels"
    assert opened[10, 10] == True, "Opening should preserve main blob"
    print("  ✓ Opening removes small noise")

    # Test 3: ksize=0 disables operation
    mask = np.ones((10, 10), dtype=bool)
    result = morphological_close(mask, ksize=0)
    assert np.array_equal(mask, result), "ksize=0 should return input unchanged"
    print("  ✓ ksize=0 disables operation")

    # Test 4: Even ksize raises error
    try:
        morphological_close(mask, ksize=4)
        raise AssertionError("Should have raised ValueError for even ksize")
    except ValueError as e:
        assert "odd" in str(e).lower()
        print("  ✓ Even ksize raises ValueError")

    # Test 5: Dtype preservation
    mask_uint8 = np.zeros((10, 10), dtype=np.uint8)
    mask_uint8[3:7, 3:7] = 255

    closed_uint8 = morphological_close(mask_uint8, ksize=3)
    assert closed_uint8.dtype == np.uint8, "Should preserve uint8 dtype"
    print("  ✓ Dtype preservation works")

    print("\n✅ All morphology sanity checks passed!")
    return True


if __name__ == "__main__":
    _run_sanity_checks()
