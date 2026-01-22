"""
Binary Classification Inference UI - Enhanced Debug Mode

Comprehensive debugging UI with folder and dataset modes.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from app.config.settings import settings
from app.utils.logging import setup_logger
from app.ui import ImageViewer
from app.io import ENVIReader, find_both_rgb_images, load_rgb, get_band_by_index
from app.data.dataset import load_dataset_csv
from app.postprocess import PostprocessPipeline, PostprocessConfig
from app.utils import normalize_to_uint8, apply_colormap

error_logger = setup_logger("error_log", str(Path(__file__).parent / "logs"))


def log_error(msg: str, exc: Optional[Exception] = None):
    if exc:
        error_logger.error(f"{msg}\n{traceback.format_exc()}")
    else:
        error_logger.error(msg)


# Worker thread for inference
class InferenceWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cube, model, preprocess_cfg):
        super().__init__()
        self.cube = cube
        self.model = model
        self.preprocess_cfg = preprocess_cfg

    def run(self):
        try:
            from app.inference.prob_map import build_prob_map
            self.progress.emit("Running inference...")
            prob_map = build_prob_map(
                self.cube, self.model, self.preprocess_cfg,
                target_class_index=1, chunk_size=100_000
            )
            self.finished.emit(prob_map)
        except Exception as e:
            log_error("Inference failed", e)
            self.error.emit(str(e))


# Main UI
class DebugInferenceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Classification Inference - Debug Mode")
        self.setGeometry(50, 50, 1600, 1000)

        # State
        self.current_mode = "folder"  # "folder" or "dataset"
        self.dataset_df = None
        self.current_index = 0
        self.current_folder = None
        self.cube = None
        self.wavelengths = None
        self.hsi_rgb = None
        self.camera_rgb = None
        self.model = None
        self.prob_map = None  # Cached prob map
        self.inference_worker = None

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)

        # Sample Source selector
        source_group = QGroupBox("Sample Source")
        source_layout = QVBoxLayout(source_group)

        self.folder_radio = QRadioButton("Folder Mode")
        self.dataset_radio = QRadioButton("Dataset Mode (CSV)")
        self.folder_radio.setChecked(True)
        self.folder_radio.toggled.connect(self.on_mode_changed)
        source_layout.addWidget(self.folder_radio)
        source_layout.addWidget(self.dataset_radio)

        # Folder mode controls
        self.folder_controls = QWidget()
        folder_layout = QVBoxLayout(self.folder_controls)
        folder_layout.setContentsMargins(0, 0, 0, 0)

        folder_btn = QPushButton("Select Folder...")
        folder_btn.clicked.connect(self.select_folder)
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(folder_btn)
        folder_layout.addWidget(self.folder_label)

        source_layout.addWidget(self.folder_controls)

        # Dataset mode controls
        self.dataset_controls = QWidget()
        dataset_layout = QVBoxLayout(self.dataset_controls)
        dataset_layout.setContentsMargins(0, 0, 0, 0)

        csv_btn = QPushButton("Load CSV...")
        csv_btn.clicked.connect(self.load_csv)
        self.csv_label = QLabel("No CSV loaded")
        self.csv_label.setWordWrap(True)
        dataset_layout.addWidget(csv_btn)
        dataset_layout.addWidget(self.csv_label)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.navigate_prev)
        self.next_btn.clicked.connect(self.navigate_next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        dataset_layout.addLayout(nav_layout)

        # Jump to index
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Index:"))
        self.index_spin = QSpinBox()
        self.index_spin.setMinimum(0)
        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self.jump_to_index)
        jump_layout.addWidget(self.index_spin)
        jump_layout.addWidget(jump_btn)
        dataset_layout.addLayout(jump_layout)

        # Search by grape_id
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        search_btn = QPushButton("Find")
        search_btn.clicked.connect(self.search_grape_id)
        search_layout.addWidget(self.search_edit)
        search_layout.addWidget(search_btn)
        dataset_layout.addLayout(search_layout)

        # Sample table (compact)
        self.sample_table = QTableWidget()
        self.sample_table.setMaximumHeight(150)
        self.sample_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sample_table.setSelectionMode(QTableWidget.SingleSelection)
        self.sample_table.itemSelectionChanged.connect(self.on_table_selection)
        dataset_layout.addWidget(QLabel("Samples:"))
        dataset_layout.addWidget(self.sample_table)

        source_layout.addWidget(self.dataset_controls)
        self.dataset_controls.setVisible(False)

        # Load Sample button
        self.load_sample_btn = QPushButton("📂 Load Sample")
        self.load_sample_btn.setEnabled(False)
        self.load_sample_btn.clicked.connect(self.load_current_sample)
        self.load_sample_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        source_layout.addWidget(self.load_sample_btn)

        left_layout.addWidget(source_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self.model_combo = QComboBox()
        self.refresh_models()
        model_layout.addRow("Model:", self.model_combo)
        left_layout.addWidget(model_group)

        # HSI Band Control
        band_group = QGroupBox("HSI Band Viewer")
        band_layout = QVBoxLayout(band_group)

        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(0)
        self.band_slider.valueChanged.connect(self.update_hsi_band_display)

        self.band_label = QLabel("Band: 0 / 0")
        band_layout.addWidget(self.band_label)
        band_layout.addWidget(self.band_slider)

        left_layout.addWidget(band_group)

        # Inference button
        self.run_inference_btn = QPushButton("▶ Run Inference")
        self.run_inference_btn.setEnabled(False)
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; background: #4CAF50; color: white; }")
        left_layout.addWidget(self.run_inference_btn)

        self.progress_label = QLabel("")
        left_layout.addWidget(self.progress_label)

        # Postprocess Controls
        post_group = QGroupBox("Postprocess Controls")
        post_layout = QFormLayout(post_group)

        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0, 1)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setValue(0.5)
        self.thresh_spin.valueChanged.connect(self.rerun_postprocess)

        self.morph_spin = QSpinBox()
        self.morph_spin.setRange(0, 15)
        self.morph_spin.setSingleStep(2)
        self.morph_spin.setValue(0)
        self.morph_spin.valueChanged.connect(self.rerun_postprocess)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000)
        self.min_area_spin.setSingleStep(10)
        self.min_area_spin.setValue(0)
        self.min_area_spin.valueChanged.connect(self.rerun_postprocess)

        self.exclude_border_check = QCheckBox("Exclude Border")
        self.exclude_border_check.stateChanged.connect(self.rerun_postprocess)

        self.border_margin_spin = QSpinBox()
        self.border_margin_spin.setRange(0, 100)
        self.border_margin_spin.setValue(0)
        self.border_margin_spin.valueChanged.connect(self.rerun_postprocess)

        post_layout.addRow("Threshold:", self.thresh_spin)
        post_layout.addRow("Morph Close:", self.morph_spin)
        post_layout.addRow("Min Area:", self.min_area_spin)
        post_layout.addRow("", self.exclude_border_check)
        post_layout.addRow("Border Margin:", self.border_margin_spin)

        self.postprocess_status = QLabel("⚠ Run inference first")
        self.postprocess_status.setStyleSheet("color: orange;")
        post_layout.addRow("", self.postprocess_status)

        left_layout.addWidget(post_group)
        self.disable_postprocess_controls()

        # Stats
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()

        # Right panel: 2x3 grid of viewers
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        grid = QGridLayout()
        grid.setSpacing(5)

        # Create 6 viewers
        self.viewer_rgb = self.create_labeled_viewer("RGB (Camera)")
        self.viewer_hsi_band = self.create_labeled_viewer("HSI Band")
        self.viewer_prob = self.create_labeled_viewer("Probability Map")
        self.viewer_thresh = self.create_labeled_viewer("After Threshold")
        self.viewer_morph = self.create_labeled_viewer("After Morphology")
        self.viewer_final = self.create_labeled_viewer("Final Mask")

        grid.addWidget(self.viewer_rgb, 0, 0)
        grid.addWidget(self.viewer_hsi_band, 0, 1)
        grid.addWidget(self.viewer_prob, 0, 2)
        grid.addWidget(self.viewer_thresh, 1, 0)
        grid.addWidget(self.viewer_morph, 1, 1)
        grid.addWidget(self.viewer_final, 1, 2)

        right_layout.addLayout(grid)

        # Add panels to main
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    def create_labeled_viewer(self, label_text):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)

        label = QLabel(label_text)
        label.setStyleSheet("font-weight: bold; background: #e0e0e0; padding: 4px;")
        label.setAlignment(Qt.AlignCenter)

        viewer = ImageViewer()

        layout.addWidget(label)
        layout.addWidget(viewer)

        widget.viewer = viewer
        return widget

    def refresh_models(self):
        self.model_combo.clear()
        if settings.models_dir.exists():
            for ext in ['.joblib', '.pkl', '.pth']:
                for f in settings.models_dir.glob(f'*{ext}'):
                    self.model_combo.addItem(f.name, str(f))

    def on_mode_changed(self):
        self.current_mode = "folder" if self.folder_radio.isChecked() else "dataset"
        self.folder_controls.setVisible(self.current_mode == "folder")
        self.dataset_controls.setVisible(self.current_mode == "dataset")
        self.load_sample_btn.setEnabled(False)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Sample Folder")
        if folder:
            self.current_folder = Path(folder)
            self.folder_label.setText(f"📁 {self.current_folder.name}")
            self.load_sample_btn.setEnabled(True)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Dataset CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.dataset_df = load_dataset_csv(path)
                self.csv_label.setText(f"✓ {len(self.dataset_df)} samples loaded")
                self.populate_table()
                self.current_index = 0
                self.index_spin.setMaximum(len(self.dataset_df) - 1)
                self.update_navigation_state()
                self.load_sample_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def populate_table(self):
        if self.dataset_df is None:
            return

        self.sample_table.setRowCount(len(self.dataset_df))
        self.sample_table.setColumnCount(3)
        self.sample_table.setHorizontalHeaderLabels(["Grape ID", "Label", "Path"])

        for i, row in self.dataset_df.iterrows():
            self.sample_table.setItem(i, 0, QTableWidgetItem(str(row['grape_id'])))
            self.sample_table.setItem(i, 1, QTableWidgetItem(str(row['label'])))
            self.sample_table.setItem(i, 2, QTableWidgetItem(str(Path(row['image_path']).name)))

        self.sample_table.resizeColumnsToContents()
        self.sample_table.selectRow(0)

    def on_table_selection(self):
        if self.dataset_df is None:
            return
        selected = self.sample_table.selectedIndexes()
        if selected:
            self.current_index = selected[0].row()
            self.index_spin.setValue(self.current_index)

    def navigate_prev(self):
        if self.dataset_df is not None and self.current_index > 0:
            self.current_index -= 1
            self.sample_table.selectRow(self.current_index)
            self.update_navigation_state()

    def navigate_next(self):
        if self.dataset_df is not None and self.current_index < len(self.dataset_df) - 1:
            self.current_index += 1
            self.sample_table.selectRow(self.current_index)
            self.update_navigation_state()

    def jump_to_index(self):
        if self.dataset_df is not None:
            self.current_index = self.index_spin.value()
            self.sample_table.selectRow(self.current_index)

    def search_grape_id(self):
        if self.dataset_df is None:
            return
        search_text = self.search_edit.text()
        mask = self.dataset_df['grape_id'].astype(str).str.contains(search_text, case=False)
        if mask.any():
            idx = mask.idxmax()
            self.current_index = idx
            self.sample_table.selectRow(idx)
        else:
            QMessageBox.information(self, "Not Found", f"No sample with grape_id containing '{search_text}'")

    def update_navigation_state(self):
        if self.dataset_df is None:
            return
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.dataset_df) - 1)
        self.index_spin.setValue(self.current_index)

    def load_current_sample(self):
        try:
            if self.current_mode == "folder":
                folder = self.current_folder
            else:
                if self.dataset_df is None:
                    return
                row = self.dataset_df.iloc[self.current_index]
                folder = Path(row['image_path'])

            if not folder.exists():
                QMessageBox.warning(self, "Warning", f"Folder not found: {folder}")
                return

            self.progress_label.setText("Loading sample...")
            QApplication.processEvents()

            # Load RGB images
            rgb_paths = find_both_rgb_images(folder)
            self.camera_rgb = load_rgb(rgb_paths['camera_rgb']) if rgb_paths['camera_rgb'] else None
            self.hsi_rgb = load_rgb(rgb_paths['hsi_rgb']) if rgb_paths['hsi_rgb'] else None

            # Display RGB
            if self.camera_rgb is not None:
                self.viewer_rgb.viewer.set_image(self.camera_rgb)
            elif self.hsi_rgb is not None:
                self.viewer_rgb.viewer.set_image(self.hsi_rgb)
            else:
                self.viewer_rgb.viewer.clear()

            # Load HSI
            hdr_files = list(folder.glob("*.hdr"))
            if hdr_files:
                reader = ENVIReader(str(hdr_files[0]))
                self.cube = reader.read()
                self.wavelengths = reader.get_wavelengths()

                # Setup band slider
                if self.cube is not None:
                    num_bands = self.cube.shape[2] if self.cube.ndim == 3 else self.cube.shape[0]
                    self.band_slider.setMaximum(num_bands - 1)
                    self.band_slider.setValue(num_bands // 2)
                    self.update_hsi_band_display()

                self.run_inference_btn.setEnabled(True)
                self.progress_label.setText(f"✓ Loaded: {self.cube.shape}")
            else:
                QMessageBox.warning(self, "Warning", "No HSI data (.hdr) found in folder")
                self.cube = None
                self.run_inference_btn.setEnabled(False)

            # Clear previous results
            self.prob_map = None
            self.viewer_prob.viewer.clear()
            self.viewer_thresh.viewer.clear()
            self.viewer_morph.viewer.clear()
            self.viewer_final.viewer.clear()
            self.disable_postprocess_controls()
            self.stats_text.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sample: {e}")
            log_error("Load sample failed", e)
        finally:
            self.progress_label.setText("")

    def update_hsi_band_display(self):
        if self.cube is None:
            return

        band_idx = self.band_slider.value()
        band = get_band_by_index(self.cube, band_idx)

        # Normalize and display
        band_norm = normalize_to_uint8(band, method="percentile")
        band_colored = apply_colormap(band_norm / 255.0, name="viridis")
        self.viewer_hsi_band.viewer.set_image(band_colored)

        # Update label
        if self.wavelengths is not None and band_idx < len(self.wavelengths):
            wl = self.wavelengths[band_idx]
            self.band_label.setText(f"Band: {band_idx} / {self.band_slider.maximum()} ({wl:.1f} nm)")
        else:
            self.band_label.setText(f"Band: {band_idx} / {self.band_slider.maximum()}")

    def run_inference(self):
        if self.cube is None:
            return

        # Load model
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "Warning", "No model selected")
            return

        try:
            import joblib
            self.model = joblib.load(model_path)

            # Create preprocess config
            from app.preprocess.snv import SNV
            preprocess_cfg = {
                'snv': SNV() if True else None  # Always use SNV for now
            }

            # Run in worker thread
            self.inference_worker = InferenceWorker(self.cube, self.model, preprocess_cfg)
            self.inference_worker.finished.connect(self.on_inference_finished)
            self.inference_worker.error.connect(self.on_inference_error)
            self.inference_worker.progress.connect(lambda msg: self.progress_label.setText(msg))

            self.run_inference_btn.setEnabled(False)
            self.inference_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start inference: {e}")
            log_error("Inference start failed", e)

    def on_inference_finished(self, prob_map):
        self.prob_map = prob_map
        self.run_inference_btn.setEnabled(True)
        self.progress_label.setText("✓ Inference complete")

        # Display prob map
        prob_vis = normalize_to_uint8(prob_map, method="percentile")
        prob_colored = apply_colormap(prob_vis / 255.0, name="hot")
        self.viewer_prob.viewer.set_image(prob_colored)

        # Enable postprocess and run
        self.enable_postprocess_controls()
        self.rerun_postprocess()

    def on_inference_error(self, error_msg):
        self.run_inference_btn.setEnabled(True)
        self.progress_label.setText("✗ Inference failed")
        QMessageBox.critical(self, "Inference Error", error_msg)

    def disable_postprocess_controls(self):
        self.thresh_spin.setEnabled(False)
        self.morph_spin.setEnabled(False)
        self.min_area_spin.setEnabled(False)
        self.exclude_border_check.setEnabled(False)
        self.border_margin_spin.setEnabled(False)
        self.postprocess_status.setText("⚠ Run inference first")
        self.postprocess_status.setStyleSheet("color: orange;")

    def enable_postprocess_controls(self):
        self.thresh_spin.setEnabled(True)
        self.morph_spin.setEnabled(True)
        self.min_area_spin.setEnabled(True)
        self.exclude_border_check.setEnabled(True)
        self.border_margin_spin.setEnabled(True)
        self.postprocess_status.setText("✓ Controls active")
        self.postprocess_status.setStyleSheet("color: green;")

    def rerun_postprocess(self):
        if self.prob_map is None:
            return

        try:
            # Build config
            morph_size = self.morph_spin.value()
            if morph_size % 2 == 0 and morph_size > 0:
                morph_size += 1  # Must be odd

            config = PostprocessConfig(
                prob_threshold=self.thresh_spin.value(),
                morph_close_size=morph_size,
                min_blob_area=self.min_area_spin.value(),
                exclude_border=self.exclude_border_check.isChecked(),
                border_margin_px=self.border_margin_spin.value()
            )

            pipeline = PostprocessPipeline(config)
            final_mask, stats, debug = pipeline.run_debug(self.prob_map)

            # Display intermediate stages
            thresh_vis = (debug['mask_threshold'].astype(np.uint8) * 255)
            self.viewer_thresh.viewer.set_image(thresh_vis)

            morph_vis = (debug['mask_after_morph'].astype(np.uint8) * 255)
            self.viewer_morph.viewer.set_image(morph_vis)

            # Final with overlay
            if self.camera_rgb is not None:
                base = self.camera_rgb.copy()
            elif self.hsi_rgb is not None:
                base = self.hsi_rgb.copy()
            else:
                base = np.zeros((*final_mask.shape, 3), dtype=np.uint8)

            self.viewer_final.viewer.set_image(base)
            self.viewer_final.viewer.set_overlay(final_mask, alpha=0.6)

            # Update stats
            stats_text = f"""Blobs Before: {stats['num_blobs_before']}
Blobs After: {stats['num_blobs_after']}
Positive Pixels: {stats['total_positive_pixels']}
Crack Ratio: {stats['crack_ratio']:.4f}
Accepted: {len(stats['accepted_blobs'])}
Rejected: {len(stats['rejected_blobs'])}"""
            self.stats_text.setText(stats_text)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Postprocessing failed: {e}")
            log_error("Postprocess failed", e)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DebugInferenceUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
