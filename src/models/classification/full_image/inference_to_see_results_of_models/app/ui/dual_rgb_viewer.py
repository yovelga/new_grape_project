"""
Dual RGB Image Viewer Widget

Displays both HSI-derived and camera RGB images side-by-side.
"""

import numpy as np
from pathlib import Path
from typing import Optional

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
except ImportError:
    raise ImportError("PyQt5 is required for DualRGBViewer widget")

from .image_viewer import ImageViewer


class DualRGBViewer(QWidget):
    """
    Widget for displaying both HSI-derived and camera RGB images side-by-side.

    Features:
    - Two synchronized image viewers
    - Labels indicating image source
    - Handles missing images gracefully
    - Automatic layout adjustment

    Usage:
        >>> viewer = DualRGBViewer()
        >>> viewer.set_hsi_rgb(hsi_image)
        >>> viewer.set_camera_rgb(camera_image)
        >>> # Or load both at once from folder
        >>> viewer.load_from_folder(sample_folder_path)
    """

    def __init__(self, parent=None):
        """Initialize the dual RGB viewer."""
        super().__init__(parent)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create horizontal layout for side-by-side viewers
        viewers_layout = QHBoxLayout()
        viewers_layout.setSpacing(10)

        # Create HSI RGB group
        self.hsi_group = QGroupBox("HSI-Derived RGB")
        hsi_layout = QVBoxLayout(self.hsi_group)
        self.hsi_viewer = ImageViewer()
        self.hsi_label = QLabel("No HSI RGB available")
        self.hsi_label.setAlignment(Qt.AlignCenter)
        self.hsi_label.setStyleSheet("color: gray; padding: 5px;")
        hsi_layout.addWidget(self.hsi_viewer)
        hsi_layout.addWidget(self.hsi_label)

        # Create Camera RGB group
        self.camera_group = QGroupBox("Camera RGB")
        camera_layout = QVBoxLayout(self.camera_group)
        self.camera_viewer = ImageViewer()
        self.camera_label = QLabel("No Camera RGB available")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("color: gray; padding: 5px;")
        camera_layout.addWidget(self.camera_viewer)
        camera_layout.addWidget(self.camera_label)

        # Add to horizontal layout
        viewers_layout.addWidget(self.hsi_group)
        viewers_layout.addWidget(self.camera_group)

        # Add to main layout
        main_layout.addLayout(viewers_layout)

        # State
        self._hsi_image: Optional[np.ndarray] = None
        self._camera_image: Optional[np.ndarray] = None

    def set_hsi_rgb(self, image: Optional[np.ndarray], filename: str = "") -> None:
        """
        Set the HSI-derived RGB image.

        Args:
            image: RGB image array (H, W, 3) uint8, or None to clear
            filename: Optional filename to display
        """
        self._hsi_image = image

        if image is not None:
            self.hsi_viewer.set_image(image)
            label_text = f"Shape: {image.shape}"
            if filename:
                label_text = f"{filename}\n{label_text}"
            self.hsi_label.setText(label_text)
            self.hsi_label.setStyleSheet("color: green; padding: 5px; font-weight: bold;")
        else:
            self.hsi_viewer.clear()
            self.hsi_label.setText("No HSI RGB available")
            self.hsi_label.setStyleSheet("color: gray; padding: 5px;")

    def set_camera_rgb(self, image: Optional[np.ndarray], filename: str = "") -> None:
        """
        Set the camera RGB image.

        Args:
            image: RGB image array (H, W, 3) uint8, or None to clear
            filename: Optional filename to display
        """
        self._camera_image = image

        if image is not None:
            self.camera_viewer.set_image(image)
            label_text = f"Shape: {image.shape}"
            if filename:
                label_text = f"{filename}\n{label_text}"
            self.camera_label.setText(label_text)
            self.camera_label.setStyleSheet("color: green; padding: 5px; font-weight: bold;")
        else:
            self.camera_viewer.clear()
            self.camera_label.setText("No Camera RGB available")
            self.camera_label.setStyleSheet("color: gray; padding: 5px;")

    def load_from_folder(self, sample_folder: Path) -> bool:
        """
        Load both RGB images from a sample folder.

        Args:
            sample_folder: Path to sample folder

        Returns:
            True if at least one image was loaded successfully
        """
        from ..io import find_both_rgb_images, load_rgb

        paths = find_both_rgb_images(sample_folder)

        success = False

        # Load HSI RGB
        if paths['hsi_rgb'] is not None:
            try:
                hsi_img = load_rgb(paths['hsi_rgb'])
                self.set_hsi_rgb(hsi_img, paths['hsi_rgb'].name)
                success = True
            except Exception as e:
                print(f"Failed to load HSI RGB: {e}")
                self.set_hsi_rgb(None)
        else:
            self.set_hsi_rgb(None)

        # Load Camera RGB
        if paths['camera_rgb'] is not None:
            try:
                camera_img = load_rgb(paths['camera_rgb'])
                self.set_camera_rgb(camera_img, paths['camera_rgb'].name)
                success = True
            except Exception as e:
                print(f"Failed to load Camera RGB: {e}")
                self.set_camera_rgb(None)
        else:
            self.set_camera_rgb(None)

        return success

    def clear(self) -> None:
        """Clear both viewers."""
        self.set_hsi_rgb(None)
        self.set_camera_rgb(None)

    def get_hsi_image(self) -> Optional[np.ndarray]:
        """Get the current HSI RGB image."""
        return self._hsi_image

    def get_camera_image(self) -> Optional[np.ndarray]:
        """Get the current camera RGB image."""
        return self._camera_image

    def reset_zoom(self) -> None:
        """Reset zoom on both viewers."""
        self.hsi_viewer.reset_zoom()
        self.camera_viewer.reset_zoom()
