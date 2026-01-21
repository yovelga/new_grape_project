"""
RGB image I/O utilities.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


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
