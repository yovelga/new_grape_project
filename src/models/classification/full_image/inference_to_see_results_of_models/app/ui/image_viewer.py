"""
Simple Image Viewer Widget

QLabel-based widget for displaying images scaled to fit the widget.
Supports grayscale/RGB numpy arrays and optional overlay masks.
No zoom/pan - just simple fixed display that fills the available space.
"""

import numpy as np
from typing import Optional

try:
    from PyQt5.QtWidgets import QLabel, QSizePolicy
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QImage
except ImportError:
    raise ImportError("PyQt5 is required for ImageViewer widget")


class ImageViewer(QLabel):
    """
    Simple image viewer that scales images to fit the widget.

    Features:
    - Display grayscale or RGB numpy arrays
    - Auto-scales image to fill widget while maintaining aspect ratio
    - Optional overlay mask support
    - No zoom/pan - just simple display

    Usage:
        viewer = ImageViewer()
        viewer.set_image(image_array)
        viewer.set_overlay(mask_array, alpha=0.5)
    """

    def __init__(self, parent=None, placeholder_text: str = "No image loaded"):
        """Initialize the image viewer."""
        super().__init__(parent)

        # State tracking
        self._base_image: Optional[np.ndarray] = None
        self._overlay_mask: Optional[np.ndarray] = None
        self._overlay_alpha: float = 0.5
        self._overlay_color: tuple = (255, 0, 0)
        self._combined_pixmap: Optional[QPixmap] = None
        self._placeholder_text: str = placeholder_text

        # Configure widget
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #323232; color: #888; font-size: 12px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)
        
        # Show placeholder initially
        self.setText(self._placeholder_text)

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the base image to display.

        Args:
            image: 2D grayscale array (H, W) or 3D RGB array (H, W, 3)
                   Values should be uint8 [0, 255] or float [0, 1]
        """
        if image is None or image.size == 0:
            self._base_image = None
            self._combined_pixmap = None
            self.clear()
            return

        # Store the image
        self._base_image = image.copy()

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Create combined image with overlay if present
        self._update_display()

    def set_overlay(self, mask: Optional[np.ndarray], alpha: float = 0.5, color: tuple = (255, 0, 0)) -> None:
        """
        Set an overlay mask to display on top of the base image.

        Args:
            mask: Boolean array (H, W) where True indicates overlay pixels
                  If None, removes the overlay
            alpha: Transparency of overlay (0=transparent, 1=opaque)
            color: RGB tuple for overlay color, e.g. (255, 0, 0) for red
        """
        if mask is None:
            self._overlay_mask = None
        else:
            self._overlay_mask = mask.copy()
        
        self._overlay_alpha = np.clip(alpha, 0.0, 1.0)
        self._overlay_color = color
        
        # Update display with overlay
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed image with any overlay applied."""
        if self._base_image is None:
            return

        image = self._base_image.copy()
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Ensure 3-channel for overlay blending
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        # Apply overlay if present
        if self._overlay_mask is not None and self._overlay_mask.shape[:2] == image.shape[:2]:
            mask = self._overlay_mask.astype(bool)
            overlay_color = np.array(self._overlay_color, dtype=np.float32)
            alpha = self._overlay_alpha
            
            # Blend overlay color with image where mask is True
            image = image.astype(np.float32)
            image[mask] = (1 - alpha) * image[mask] + alpha * overlay_color
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Convert to QImage
        qimage = self._numpy_to_qimage(image)
        self._combined_pixmap = QPixmap.fromImage(qimage)
        
        # Scale and display
        self._scale_and_show()

    def _scale_and_show(self) -> None:
        """Scale the pixmap to fit the widget and display it."""
        if self._combined_pixmap is None or self._combined_pixmap.isNull():
            return
        
        # Scale pixmap to fit widget while keeping aspect ratio
        scaled = self._combined_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        """Handle widget resize - rescale image to fit."""
        super().resizeEvent(event)
        self._scale_and_show()

    def clear(self) -> None:
        """Clear the image and overlay, show placeholder."""
        self._base_image = None
        self._overlay_mask = None
        self._combined_pixmap = None
        super().clear()
        self.setText(self._placeholder_text)

    def _numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """
        Convert numpy array to QImage.

        Args:
            image: 2D grayscale (H, W) or 3D RGB (H, W, 3) uint8 array

        Returns:
            QImage object
        """
        # Ensure array is contiguous for QImage
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        if image.ndim == 2:
            # Grayscale image
            h, w = image.shape
            bytes_per_line = w
            qimage = QImage(
                image.data,
                w, h,
                bytes_per_line,
                QImage.Format_Grayscale8
            )
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB image
            h, w, _ = image.shape
            bytes_per_line = w * 3
            qimage = QImage(
                image.data,
                w, h,
                bytes_per_line,
                QImage.Format_RGB888
            )
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Return a copy to avoid data corruption
        return qimage.copy()

