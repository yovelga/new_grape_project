"""
Minimal demo script for ImageViewer widget.

This script creates a standalone window demonstrating the ImageViewer
capabilities without integrating into the main UI.

Run: python scripts/demo_image_viewer.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
    from PyQt5.QtCore import Qt
except ImportError:
    print("Error: PyQt5 is required to run this demo")
    sys.exit(1)

from app.ui import ImageViewer
from app.utils import normalize_to_uint8, apply_colormap


class DemoWindow(QMainWindow):
    """Demo window showing ImageViewer capabilities."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageViewer Demo")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Add image viewer
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, stretch=1)

        # Add control buttons
        button_layout = QHBoxLayout()

        btn_random = QPushButton("Random Grayscale")
        btn_random.clicked.connect(self.show_random_grayscale)
        button_layout.addWidget(btn_random)

        btn_rgb = QPushButton("Random RGB")
        btn_rgb.clicked.connect(self.show_random_rgb)
        button_layout.addWidget(btn_rgb)

        btn_colormap = QPushButton("Apply Viridis")
        btn_colormap.clicked.connect(self.show_with_colormap)
        button_layout.addWidget(btn_colormap)

        btn_overlay = QPushButton("Toggle Overlay")
        btn_overlay.clicked.connect(self.toggle_overlay)
        button_layout.addWidget(btn_overlay)

        btn_reset = QPushButton("Reset Zoom")
        btn_reset.clicked.connect(self.viewer.reset_zoom)
        button_layout.addWidget(btn_reset)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.viewer.clear)
        button_layout.addWidget(btn_clear)

        layout.addLayout(button_layout)

        # Add instructions
        info = QLabel(
            "Instructions: Use mouse wheel to zoom | Click and drag to pan | "
            "Try the buttons to test different features"
        )
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(info)

        # State
        self.overlay_visible = False
        self.current_image = None

        # Show initial image
        self.show_random_grayscale()

    def show_random_grayscale(self):
        """Display a random grayscale image."""
        img = np.random.randint(0, 255, (400, 600), dtype=np.uint8)
        self.current_image = img
        self.viewer.set_image(img)
        self.overlay_visible = False
        print("Displayed random grayscale image")

    def show_random_rgb(self):
        """Display a random RGB image."""
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        self.current_image = img
        self.viewer.set_image(img)
        self.overlay_visible = False
        print("Displayed random RGB image")

    def show_with_colormap(self):
        """Display a random image with viridis colormap."""
        # Generate random data
        raw = np.random.randn(400, 600) * 50 + 128

        # Normalize
        normalized = normalize_to_uint8(raw, method="percentile")

        # Apply colormap
        colored = apply_colormap(normalized / 255.0, name="viridis")

        self.current_image = colored
        self.viewer.set_image(colored)
        self.overlay_visible = False
        print("Applied viridis colormap")

    def toggle_overlay(self):
        """Toggle overlay on/off."""
        if self.current_image is None:
            print("No image loaded")
            return

        if self.overlay_visible:
            self.viewer.set_overlay(None)
            self.overlay_visible = False
            print("Overlay hidden")
        else:
            # Create random mask
            h, w = self.current_image.shape[:2]
            mask = np.random.rand(h, w) > 0.85
            self.viewer.set_overlay(mask, alpha=0.6)
            self.overlay_visible = True
            print("Overlay shown (random mask)")


def main():
    """Run the demo application."""
    app = QApplication(sys.argv)

    window = DemoWindow()
    window.show()

    print("\n" + "="*60)
    print("ImageViewer Demo Window")
    print("="*60)
    print("\nFeatures to test:")
    print("  • Mouse wheel zoom (centered on cursor)")
    print("  • Click and drag to pan")
    print("  • Buttons to change image type")
    print("  • Toggle overlay on/off")
    print("  • Reset zoom to fit view")
    print("\n" + "="*60 + "\n")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
