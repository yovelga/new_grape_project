"""
RGB image I/O utilities.

Provides functions to find and load RGB images from sample folders.
Supports both HSI-derived RGB (from hyperspectral data) and camera RGB.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict, Tuple


def find_rgb_image(sample_folder: Path) -> Optional[Path]:
    """
    Find RGB image in sample folder using multiple search patterns.

    Searches in the following order:
    1. <sample_folder>/RGB/*.{png,jpg,jpeg}
    2. Sibling folders: <sample_folder>/../RGB/
    3. Files in sample_folder containing 'rgb' in name

    Args:
        sample_folder: Path to sample folder

    Returns:
        Path to RGB image if found, None otherwise

    Example:
        >>> folder = Path("data/raw/1_01/16.08.24")
        >>> rgb_path = find_rgb_image(folder)
        >>> if rgb_path:
        ...     img = load_rgb(rgb_path)
    """
    sample_folder = Path(sample_folder)

    if not sample_folder.exists():
        return None

    # Pattern 1: <sample_folder>/RGB/*.{png,jpg,jpeg}
    rgb_subfolder = sample_folder / "RGB"
    if rgb_subfolder.exists() and rgb_subfolder.is_dir():
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            matches = list(rgb_subfolder.glob(f"*.{ext}"))
            if matches:
                return matches[0]  # Return first match

    # Pattern 2: Sibling folders like ../RGB/
    # Look for RGB folder at parent level
    parent = sample_folder.parent
    sibling_rgb = parent / "RGB"
    if sibling_rgb.exists() and sibling_rgb.is_dir():
        # Try to find image with matching sample name or date
        sample_name = sample_folder.name
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            # Try exact match
            candidate = sibling_rgb / f"{sample_name}.{ext}"
            if candidate.exists():
                return candidate
            # Try any image in folder
            matches = list(sibling_rgb.glob(f"*.{ext}"))
            if matches:
                return matches[0]

    # Pattern 3: Files containing 'rgb' in filename (case-insensitive)
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
        for candidate in sample_folder.glob(f"*.{ext}"):
            if 'rgb' in candidate.name.lower():
                return candidate

    # Pattern 4: Check direct RGB image in sample folder
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
        matches = list(sample_folder.glob(f"*.{ext}"))
        if matches:
            return matches[0]  # Return first image found

    return None


def load_rgb(path: Path) -> np.ndarray:
    """
    Load RGB image from path.

    Args:
        path: Path to RGB image file

    Returns:
        RGB image array with shape (H, W, 3) and dtype uint8 [0-255]

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or converted to RGB

    Example:
        >>> img = load_rgb(Path("sample/RGB/image.png"))
        >>> print(img.shape, img.dtype)
        (512, 512, 3) uint8
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"RGB image not found: {path}")

    try:
        # Load and convert to RGB (handles RGBA, grayscale, etc.)
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {arr.shape}")

        return arr

    except Exception as e:
        raise ValueError(f"Failed to load RGB image from {path}: {e}")


def find_hsi_rgb(sample_folder: Path) -> Optional[Path]:
    """
    Find HSI-derived RGB image in the same directory as HSI data.

    Args:
        sample_folder: Path to sample folder containing HSI data

    Returns:
        Path to HSI-derived RGB image if found, None otherwise
    """
    sample_folder = Path(sample_folder)

    if not sample_folder.exists():
        return None

    # Files containing 'rgb' in filename (HSI-derived typically named this way)
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
        for candidate in sample_folder.glob(f"*.{ext}"):
            if 'rgb' in candidate.name.lower():
                return candidate

    # Any image file directly in sample folder (fallback)
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
        matches = list(sample_folder.glob(f"*.{ext}"))
        if matches:
            return matches[0]

    return None


def find_camera_rgb(sample_folder: Path) -> Optional[Path]:
    """
    Find camera RGB image (typically in RGB/ subfolder).

    Args:
        sample_folder: Path to sample folder

    Returns:
        Path to camera RGB image if found, None otherwise
    """
    sample_folder = Path(sample_folder)

    if not sample_folder.exists():
        return None

    # <sample_folder>/RGB/*.{png,jpg,jpeg}
    rgb_subfolder = sample_folder / "RGB"
    if rgb_subfolder.exists() and rgb_subfolder.is_dir():
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            matches = list(rgb_subfolder.glob(f"*.{ext}"))
            if matches:
                return matches[0]

    # Sibling folders like ../RGB/
    parent = sample_folder.parent
    sibling_rgb = parent / "RGB"
    if sibling_rgb.exists() and sibling_rgb.is_dir():
        sample_name = sample_folder.name
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            candidate = sibling_rgb / f"{sample_name}.{ext}"
            if candidate.exists():
                return candidate
            matches = list(sibling_rgb.glob(f"*.{ext}"))
            if matches:
                return matches[0]

    return None


def find_both_rgb_images(sample_folder: Path) -> Dict[str, Optional[Path]]:
    """
    Find both HSI-derived and camera RGB images for a sample.

    Args:
        sample_folder: Path to sample folder

    Returns:
        Dictionary with keys 'hsi_rgb' and 'camera_rgb'
    """
    return {
        'hsi_rgb': find_hsi_rgb(sample_folder),
        'camera_rgb': find_camera_rgb(sample_folder),
    }


def load_both_rgb_images(sample_folder: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load both HSI-derived and camera RGB images for a sample.

    Args:
        sample_folder: Path to sample folder

    Returns:
        Tuple of (hsi_rgb_array, camera_rgb_array), either can be None
    """
    paths = find_both_rgb_images(sample_folder)

    hsi_rgb = None
    if paths['hsi_rgb'] is not None:
        try:
            hsi_rgb = load_rgb(paths['hsi_rgb'])
        except Exception:
            pass

    camera_rgb = None
    if paths['camera_rgb'] is not None:
        try:
            camera_rgb = load_rgb(paths['camera_rgb'])
        except Exception:
            pass

    return hsi_rgb, camera_rgb


class RGBReader:
    """Reader for standard RGB images."""

    @staticmethod
    def read(path: Union[str, Path]) -> np.ndarray:
        """
        Read RGB image.

        Args:
            path: Path to image file

        Returns:
            RGB image array with shape (H, W, 3)
        """
        img = Image.open(path).convert('RGB')
        return np.array(img)

    @staticmethod
    def save(image: np.ndarray, path: Union[str, Path]):
        """
        Save RGB image.

        Args:
            image: RGB image array
            path: Output path
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        img = Image.fromarray(image)
        img.save(path)
