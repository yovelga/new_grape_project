"""
Interactive Image Viewer Widget

QGraphicsView-based widget for displaying images with zoom and pan capabilities.
Supports grayscale/RGB numpy arrays and optional overlay masks.
"""

import numpy as np
from typing import Optional

try:
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
    from PyQt5.QtCore import Qt, QRectF
    from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
except ImportError:
    raise ImportError("PyQt5 is required for ImageViewer widget")


class ImageViewer(QGraphicsView):
    """
    Interactive image viewer with zoom and pan support.

    Features:
    - Display grayscale or RGB numpy arrays
    - Mouse wheel zoom (centered on cursor)
    - Click and drag to pan
    - Maintains aspect ratio
    - Optional overlay mask support

    Usage:
        viewer = ImageViewer()
        viewer.set_image(image_array)
        viewer.set_overlay(mask_array, alpha=0.5)
    """

    def __init__(self, parent=None):
        """Initialize the image viewer."""
        super().__init__(parent)

        # Setup scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Create pixmap items
        self._image_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()

        self.scene.addItem(self._image_item)
        self.scene.addItem(self._overlay_item)

        # State tracking
        self._base_image: Optional[np.ndarray] = None
        self._overlay_mask: Optional[np.ndarray] = None
        self._overlay_alpha: float = 0.5
        self._zoom_factor: float = 1.0
        self._is_panning: bool = False
        self._pan_start_pos = None

        # Configure view
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(50, 50, 50))

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the base image to display.

        Args:
            image: 2D grayscale array (H, W) or 3D RGB array (H, W, 3)
                   Values should be uint8 [0, 255] or float [0, 1]
        """
        if image is None or image.size == 0:
            self._image_item.setPixmap(QPixmap())
            self._base_image = None
            return

        # Store the image
        self._base_image = image.copy()

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to QImage
        qimage = self._numpy_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)

        # Set pixmap
        self._image_item.setPixmap(pixmap)

        # Fit in view on first load
        if self._zoom_factor == 1.0:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()

    def set_overlay(self, mask: Optional[np.ndarray], alpha: float = 0.5) -> None:
        """
        Set an overlay mask to display on top of the base image.

        Args:
            mask: Boolean array (H, W) where True indicates overlay pixels
                  If None, removes the overlay
            alpha: Transparency of overlay (0=transparent, 1=opaque)
        """
        if mask is None:
            self._overlay_item.setPixmap(QPixmap())
            self._overlay_mask = None
            return

        self._overlay_mask = mask.copy()
        self._overlay_alpha = np.clip(alpha, 0.0, 1.0)

        # Create overlay image
        if self._base_image is not None:
            h, w = mask.shape[:2]

            # Create RGBA image for overlay
            overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)

            # Red overlay where mask is True
            overlay_rgba[mask, 0] = 255  # Red channel
            overlay_rgba[mask, 3] = int(self._overlay_alpha * 255)  # Alpha channel

            # Convert to QImage
            qimage = QImage(
                overlay_rgba.data,
                w, h,
                w * 4,
                QImage.Format_RGBA8888
            )

            pixmap = QPixmap.fromImage(qimage)
            self._overlay_item.setPixmap(pixmap)

            # Position overlay on top of image
            self._overlay_item.setPos(self._image_item.pos())

    def clear(self) -> None:
        """Clear both image and overlay."""
        self._image_item.setPixmap(QPixmap())
        self._overlay_item.setPixmap(QPixmap())
        self._base_image = None
        self._overlay_mask = None
        self._zoom_factor = 1.0

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if self._base_image is None:
            return

        # Calculate zoom factor
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Zoom in or out
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # Apply zoom
        self.scale(zoom_factor, zoom_factor)
        self._zoom_factor *= zoom_factor

    def mousePressEvent(self, event):
        """Start panning on left click."""
        if event.button() == Qt.LeftButton:
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle panning when dragging."""
        if self._is_panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()

            # Move scrollbars
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Stop panning on release."""
        if event.button() == Qt.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

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
            # Qt expects RGB888 format
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

    def reset_zoom(self) -> None:
        """Reset zoom to fit the image in view."""
        if self._base_image is not None:
            self.resetTransform()
            self.fitInView(self._image_item, Qt.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
