"""
Demo script showing DualRGBViewer in action.

Run this to see both RGB types displayed side-by-side.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from app.ui import DualRGBViewer
import pandas as pd

class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual RGB Viewer Demo - HSI vs Camera")
        self.setGeometry(100, 100, 1200, 700)

        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Add dual RGB viewer
        self.dual_viewer = DualRGBViewer()
        layout.addWidget(self.dual_viewer, stretch=1)

        # Add control buttons
        btn_layout = QHBoxLayout()

        btn_load = QPushButton("Load Sample from Dataset")
        btn_load.clicked.connect(self.load_sample)
        btn_layout.addWidget(btn_load)

        btn_reset = QPushButton("Reset Zoom")
        btn_reset.clicked.connect(self.dual_viewer.reset_zoom)
        btn_layout.addWidget(btn_reset)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.dual_viewer.clear)
        btn_layout.addWidget(btn_clear)

        layout.addLayout(btn_layout)

        # Load first sample automatically
        self.load_sample()

    def load_sample(self):
        """Load a sample from the test dataset."""
        csv_path = Path(__file__).parent.parent / "data" / "test_dataset.csv"

        if not csv_path.exists():
            print(f"Dataset not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)

        # Try to find a sample with data
        for _, row in df.head(20).iterrows():
            sample_path = Path(row['image_path'])

            if sample_path.exists():
                success = self.dual_viewer.load_from_folder(sample_path)
                if success:
                    print(f"✓ Loaded sample: {sample_path.name}")
                    print(f"  Grape ID: {row['grape_id']}")
                    print(f"  Week: {row['week_date']}")
                    print(f"  Label: {row['label']}")

                    if self.dual_viewer.get_hsi_image() is not None:
                        print(f"  HSI RGB: {self.dual_viewer.get_hsi_image().shape}")
                    if self.dual_viewer.get_camera_image() is not None:
                        print(f"  Camera RGB: {self.dual_viewer.get_camera_image().shape}")

                    self.setWindowTitle(f"Dual RGB Viewer - {row['grape_id']} / {row['week_date']}")
                    return

        print("No samples with RGB images found")

def main():
    app = QApplication(sys.argv)

    window = DemoWindow()
    window.show()

    print("\n" + "="*70)
    print("Dual RGB Viewer Demo")
    print("="*70)
    print("\nShowing:")
    print("  LEFT:  HSI-Derived RGB (from hyperspectral camera)")
    print("  RIGHT: Camera RGB (from regular camera)")
    print("\nControls:")
    print("  • Mouse wheel to zoom (each viewer independently)")
    print("  • Click and drag to pan")
    print("  • 'Load Sample' to load another sample")
    print("  • 'Reset Zoom' to fit images")
    print("="*70 + "\n")

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
