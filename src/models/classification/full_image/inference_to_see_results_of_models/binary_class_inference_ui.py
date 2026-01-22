"""
Binary Classification Inference UI - Unified Version

Comprehensive UI combining:
- Tab A: Visual Debug / Image Playground (Folder mode + Dataset browsing)
- Tab B: Dataset Tuning / Scientific Evaluation (Optuna + Final Test)

All non-UI logic delegated to app/* modules. Single runnable entrypoint.
"""

import sys
import traceback
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QGroupBox,
    QFormLayout, QFrame, QMessageBox, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QCheckBox, QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from app.config.settings import settings
from app.utils.logging import setup_logger
from app.ui import ImageViewer
from app.io import ENVIReader, find_both_rgb_images, load_rgb, get_band_by_index
from app.data.dataset import load_dataset_csv
from app.postprocess import PostprocessPipeline, PostprocessConfig
from app.utils import normalize_to_uint8, apply_colormap

# Setup logger
error_logger = setup_logger("error_log", str(Path(__file__).parent / "logs"))


def log_error(msg: str, exc: Optional[Exception] = None):
    """Log error with optional traceback to file."""
    if exc:
        error_logger.error(f"{msg}\n{traceback.format_exc()}")
    else:
        error_logger.error(msg)


def show_error(parent, title: str, message: str):
    """Show user-friendly error dialog."""
    QMessageBox.critical(parent, title, message)


def show_info(parent, title: str, message: str):
    """Show info dialog."""
    QMessageBox.information(parent, title, message)


# ============================================================================
# Worker Threads
# ============================================================================

class InferenceWorker(QThread):
    """Background worker for inference."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cube, model, preprocess_cfg):
        super().__init__()
        self.cube = cube
        self.model = model
        self.preprocess_cfg = preprocess_cfg
        self._stop = False

    def run(self):
        try:
            from app.inference.prob_map import build_prob_map
            self.progress.emit("Running inference...")
            prob_map = build_prob_map(
                self.cube, self.model, self.preprocess_cfg,
                target_class_index=1, chunk_size=100_000
            )
            if not self._stop:
                self.finished.emit(prob_map)
        except Exception as e:
            if not self._stop:
                log_error("Inference failed", e)
                self.error.emit(str(e))

    def stop(self):
        self._stop = True


class GridSearchWorker(QThread):
    """Background worker for grid search."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, prob_map, param_grid, metric="crack_ratio"):
        super().__init__()
        self.prob_map = prob_map
        self.param_grid = param_grid
        self.metric = metric
        self._stop = False

    def run(self):
        try:
            from app.tuning import run_grid_on_prob_map
            total = 1
            for v in self.param_grid.values():
                total *= len(v)
            self.progress.emit(f"Running grid search ({total} combinations)...")
            results_df = run_grid_on_prob_map(self.prob_map, self.param_grid, self.metric)
            if not self._stop:
                self.finished.emit(results_df)
        except Exception as e:
            if not self._stop:
                log_error("Grid search failed", e)
                self.error.emit(str(e))

    def stop(self):
        self._stop = True


class OptunaWorker(QThread):
    """Background worker for Optuna tuning."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, train_df, val_df, prob_map_fn, n_trials, seed, output_dir, metric):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.prob_map_fn = prob_map_fn
        self.n_trials = n_trials
        self.seed = seed
        self.output_dir = output_dir
        self.metric = metric
        self._stop = False

    def run(self):
        try:
            from app.tuning.optuna_runner import run_optuna
            self.progress.emit("Running Optuna tuning...")
            best_params, trials_df = run_optuna(
                train_df=self.train_df,
                val_df=self.val_df,
                prob_map_fn=self.prob_map_fn,
                n_trials=self.n_trials,
                seed=self.seed,
                output_dir=self.output_dir,
                metric=self.metric,
            )
            if not self._stop:
                self.finished.emit({"best_params": best_params, "n_trials": len(trials_df)})
        except Exception as e:
            if not self._stop:
                log_error("Optuna failed", e)
                self.error.emit(str(e))

    def stop(self):
        self._stop = True


class FinalEvalWorker(QThread):
    """Background worker for final test evaluation."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, test_df, prob_map_fn, best_params, output_dir, train_df, val_df):
        super().__init__()
        self.test_df = test_df
        self.prob_map_fn = prob_map_fn
        self.best_params = best_params
        self.output_dir = output_dir
        self.train_df = train_df
        self.val_df = val_df
        self._stop = False

    def run(self):
        try:
            from app.tuning.optuna_runner import evaluate_final
            self.progress.emit("Running final evaluation...")
            metrics, _ = evaluate_final(
                test_df=self.test_df,
                prob_map_fn=self.prob_map_fn,
                best_params=self.best_params,
                output_dir=self.output_dir,
                train_df=self.train_df,
                val_df=self.val_df,
            )
            if not self._stop:
                self.finished.emit(metrics)
        except Exception as e:
            if not self._stop:
                log_error("Final eval failed", e)
                self.error.emit(str(e))

    def stop(self):
        self._stop = True


# ============================================================================
# Tab A: Visual Debug / Image Playground
# ============================================================================

class VisualDebugTab(QWidget):
    """Visual debugging tab with 2x3 viewer grid and live postprocess."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self.current_mode = "folder"
        self.dataset_df = None
        self.current_index = 0
        self.current_folder = None
        self.cube = None
        self.wavelengths = None
        self.hsi_rgb = None
        self.camera_rgb = None
        self.model = None
        self.prob_map = None
        self.grid_results = None


        # Workers
        self.inference_worker = None
        self.grid_worker = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # === Left Panel: Controls ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(420)

        # Sample Source
        source_group = QGroupBox("Sample Source")
        source_layout = QVBoxLayout(source_group)

        self.folder_radio = QRadioButton("Folder Mode")
        self.dataset_radio = QRadioButton("Dataset Mode (CSV)")
        self.folder_radio.setChecked(True)
        self.folder_radio.toggled.connect(self._on_mode_changed)
        source_layout.addWidget(self.folder_radio)
        source_layout.addWidget(self.dataset_radio)

        # Folder controls
        self.folder_controls = QWidget()
        folder_layout = QVBoxLayout(self.folder_controls)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_btn = QPushButton("Select Folder...")
        folder_btn.clicked.connect(self._select_folder)
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(folder_btn)
        folder_layout.addWidget(self.folder_label)
        source_layout.addWidget(self.folder_controls)

        # Dataset controls
        self.dataset_controls = QWidget()
        dataset_layout = QVBoxLayout(self.dataset_controls)
        dataset_layout.setContentsMargins(0, 0, 0, 0)

        csv_btn = QPushButton("Load CSV...")
        csv_btn.clicked.connect(self._load_csv)
        self.csv_label = QLabel("No CSV loaded")
        self.csv_label.setWordWrap(True)
        dataset_layout.addWidget(csv_btn)
        dataset_layout.addWidget(self.csv_label)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self._navigate_prev)
        self.next_btn.clicked.connect(self._navigate_next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        dataset_layout.addLayout(nav_layout)

        # Jump to index
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Index:"))
        self.index_spin = QSpinBox()
        self.index_spin.setMinimum(0)
        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self._jump_to_index)
        jump_layout.addWidget(self.index_spin)
        jump_layout.addWidget(jump_btn)
        dataset_layout.addLayout(jump_layout)

        # Search grape_id
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("grape_id")
        search_btn = QPushButton("Find")
        search_btn.clicked.connect(self._search_grape_id)
        search_layout.addWidget(self.search_edit)
        search_layout.addWidget(search_btn)
        dataset_layout.addLayout(search_layout)

        # Sample table
        self.sample_table = QTableWidget()
        self.sample_table.setMaximumHeight(120)
        self.sample_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sample_table.setSelectionMode(QTableWidget.SingleSelection)
        self.sample_table.itemSelectionChanged.connect(self._on_table_selection)
        dataset_layout.addWidget(self.sample_table)

        source_layout.addWidget(self.dataset_controls)
        self.dataset_controls.setVisible(False)

        # Load Sample button
        self.load_sample_btn = QPushButton("📂 Load Sample")
        self.load_sample_btn.setEnabled(False)
        self.load_sample_btn.clicked.connect(self._load_current_sample)
        self.load_sample_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        source_layout.addWidget(self.load_sample_btn)

        left_layout.addWidget(source_group)

        # Model
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self.model_combo = QComboBox()
        self._refresh_models()
        model_layout.addRow("Model:", self.model_combo)
        left_layout.addWidget(model_group)

        # HSI Band Control
        band_group = QGroupBox("HSI Band Viewer")
        band_layout = QVBoxLayout(band_group)
        self.band_label = QLabel("Band: 0 / 0")
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(0)
        self.band_slider.valueChanged.connect(self._update_hsi_band)

        # Go to wavelength
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("Go to nm:"))
        self.wl_spin = QSpinBox()
        self.wl_spin.setRange(400, 2500)
        self.wl_spin.setValue(700)
        wl_btn = QPushButton("Go")
        wl_btn.clicked.connect(self._go_to_wavelength)
        wl_layout.addWidget(self.wl_spin)
        wl_layout.addWidget(wl_btn)

        band_layout.addWidget(self.band_label)
        band_layout.addWidget(self.band_slider)
        band_layout.addLayout(wl_layout)

        left_layout.addWidget(band_group)


        # Inference
        self.run_inference_btn = QPushButton("▶ Run Inference")
        self.run_inference_btn.setEnabled(False)
        self.run_inference_btn.clicked.connect(self._run_inference)
        self.run_inference_btn.setStyleSheet("font-weight: bold; padding: 10px; background: #4CAF50; color: white;")
        left_layout.addWidget(self.run_inference_btn)

        self.progress_label = QLabel("")
        left_layout.addWidget(self.progress_label)

        # Postprocess Controls
        post_group = QGroupBox("Postprocess Controls (Live)")
        post_layout = QFormLayout(post_group)

        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0, 1)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setValue(0.5)
        self.thresh_spin.valueChanged.connect(self._rerun_postprocess)

        self.morph_spin = QSpinBox()
        self.morph_spin.setRange(0, 15)
        self.morph_spin.setSingleStep(2)
        self.morph_spin.setValue(0)
        self.morph_spin.valueChanged.connect(self._rerun_postprocess)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000)
        self.min_area_spin.setSingleStep(10)
        self.min_area_spin.setValue(0)
        self.min_area_spin.valueChanged.connect(self._rerun_postprocess)

        self.exclude_border_check = QCheckBox("Exclude Border")
        self.exclude_border_check.stateChanged.connect(self._rerun_postprocess)

        self.border_margin_spin = QSpinBox()
        self.border_margin_spin.setRange(0, 100)
        self.border_margin_spin.setValue(0)
        self.border_margin_spin.valueChanged.connect(self._rerun_postprocess)

        post_layout.addRow("Threshold:", self.thresh_spin)
        post_layout.addRow("Morph Close:", self.morph_spin)
        post_layout.addRow("Min Area:", self.min_area_spin)
        post_layout.addRow("", self.exclude_border_check)
        post_layout.addRow("Border Margin:", self.border_margin_spin)

        self.post_status = QLabel("⚠ Run inference first")
        self.post_status.setStyleSheet("color: orange;")
        post_layout.addRow("", self.post_status)

        left_layout.addWidget(post_group)
        # NOTE: _disable_postprocess() is called after grid controls are created

        # Stats
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        stats_layout.addWidget(self.stats_text)
        left_layout.addWidget(stats_group)

        # Grid Search
        grid_group = QGroupBox("Grid Search Playground")
        grid_layout = QVBoxLayout(grid_group)

        grid_btn_row = QHBoxLayout()
        self.run_grid_btn = QPushButton("🔍 Run Grid")
        self.run_grid_btn.setEnabled(False)
        self.run_grid_btn.clicked.connect(self._run_grid_search)
        self.save_grid_btn = QPushButton("💾 Save")
        self.save_grid_btn.setEnabled(False)
        self.save_grid_btn.clicked.connect(self._save_grid_results)
        grid_btn_row.addWidget(self.run_grid_btn)
        grid_btn_row.addWidget(self.save_grid_btn)
        grid_layout.addLayout(grid_btn_row)

        self.grid_status = QLabel("⚠ Run inference first")
        self.grid_status.setStyleSheet("color: orange; font-size: 10px;")
        grid_layout.addWidget(self.grid_status)

        self.grid_table = QTableWidget()
        self.grid_table.setMaximumHeight(150)
        self.grid_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.grid_table.setSelectionMode(QTableWidget.SingleSelection)
        self.grid_table.itemSelectionChanged.connect(self._on_grid_row_selected)
        grid_layout.addWidget(QLabel("Top results (click to preview):"))
        grid_layout.addWidget(self.grid_table)

        left_layout.addWidget(grid_group)

        # Now disable postprocess controls (after all controls created)
        self._disable_postprocess()

        left_layout.addStretch()

        # === Right Panel: 2x3 Viewer Grid ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        grid = QGridLayout()
        grid.setSpacing(5)

        self.viewer_rgb = self._create_viewer("RGB (Camera)")
        self.viewer_hsi = self._create_viewer("HSI Band")
        self.viewer_prob = self._create_viewer("Probability Map")
        self.viewer_thresh = self._create_viewer("After Threshold")
        self.viewer_morph = self._create_viewer("After Morphology")
        self.viewer_final = self._create_viewer("Final Mask + Overlay")

        grid.addWidget(self.viewer_rgb, 0, 0)
        grid.addWidget(self.viewer_hsi, 0, 1)
        grid.addWidget(self.viewer_prob, 0, 2)
        grid.addWidget(self.viewer_thresh, 1, 0)
        grid.addWidget(self.viewer_morph, 1, 1)
        grid.addWidget(self.viewer_final, 1, 2)

        right_layout.addLayout(grid)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    def _create_viewer(self, label_text):
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

    def _refresh_models(self):
        self.model_combo.clear()
        if settings.models_dir.exists():
            for ext in ['.joblib', '.pkl', '.pth']:
                for f in settings.models_dir.glob(f'*{ext}'):
                    self.model_combo.addItem(f.name, str(f))

    def _on_mode_changed(self):
        self.current_mode = "folder" if self.folder_radio.isChecked() else "dataset"
        self.folder_controls.setVisible(self.current_mode == "folder")
        self.dataset_controls.setVisible(self.current_mode == "dataset")
        self.load_sample_btn.setEnabled(False)

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Sample Folder", str(settings.default_search_folder))
        if folder:
            self.current_folder = Path(folder)
            self.folder_label.setText(f"📁 {self.current_folder.name}")
            self.load_sample_btn.setEnabled(True)

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Dataset CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.dataset_df = load_dataset_csv(path)
                self.csv_label.setText(f"✓ {len(self.dataset_df)} samples")
                self._populate_table()
                self.current_index = 0
                self.index_spin.setMaximum(len(self.dataset_df) - 1)
                self._update_nav_state()
                self.load_sample_btn.setEnabled(True)
            except Exception as e:
                show_error(self, "Error", f"Failed to load CSV: {e}")

    def _populate_table(self):
        if self.dataset_df is None:
            return
        self.sample_table.setRowCount(len(self.dataset_df))
        self.sample_table.setColumnCount(3)
        self.sample_table.setHorizontalHeaderLabels(["Grape ID", "Label", "Path"])
        for i, row in self.dataset_df.iterrows():
            self.sample_table.setItem(i, 0, QTableWidgetItem(str(row['grape_id'])))
            self.sample_table.setItem(i, 1, QTableWidgetItem(str(row['label'])))
            self.sample_table.setItem(i, 2, QTableWidgetItem(Path(row['image_path']).name))
        self.sample_table.resizeColumnsToContents()
        self.sample_table.selectRow(0)

    def _on_table_selection(self):
        if self.dataset_df is None:
            return
        selected = self.sample_table.selectedIndexes()
        if selected:
            self.current_index = selected[0].row()
            self.index_spin.setValue(self.current_index)

    def _navigate_prev(self):
        if self.dataset_df is not None and self.current_index > 0:
            self.current_index -= 1
            self.sample_table.selectRow(self.current_index)
            self._update_nav_state()

    def _navigate_next(self):
        if self.dataset_df is not None and self.current_index < len(self.dataset_df) - 1:
            self.current_index += 1
            self.sample_table.selectRow(self.current_index)
            self._update_nav_state()

    def _jump_to_index(self):
        if self.dataset_df is not None:
            self.current_index = self.index_spin.value()
            self.sample_table.selectRow(self.current_index)

    def _search_grape_id(self):
        if self.dataset_df is None:
            return
        text = self.search_edit.text()
        mask = self.dataset_df['grape_id'].astype(str).str.contains(text, case=False)
        if mask.any():
            idx = mask.idxmax()
            self.current_index = idx
            self.sample_table.selectRow(idx)
        else:
            show_info(self, "Not Found", f"No grape_id containing '{text}'")

    def _update_nav_state(self):
        if self.dataset_df is None:
            return
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.dataset_df) - 1)
        self.index_spin.setValue(self.current_index)

    def _load_current_sample(self):
        try:
            if self.current_mode == "folder":
                folder = self.current_folder
            else:
                if self.dataset_df is None:
                    return
                row = self.dataset_df.iloc[self.current_index]
                folder = Path(row['image_path'])

            if not folder or not folder.exists():
                show_error(self, "Error", f"Folder not found: {folder}")
                return

            self.progress_label.setText("Loading sample...")
            QApplication.processEvents()

            # Load RGB
            rgb_paths = find_both_rgb_images(folder)
            self.camera_rgb = load_rgb(rgb_paths['camera_rgb']) if rgb_paths.get('camera_rgb') else None
            self.hsi_rgb = load_rgb(rgb_paths['hsi_rgb']) if rgb_paths.get('hsi_rgb') else None

            # Display RGB without rotation (RGB camera is already in correct orientation)
            if self.camera_rgb is not None:
                self.viewer_rgb.viewer.set_image(self.camera_rgb)
            elif self.hsi_rgb is not None:
                self.viewer_rgb.viewer.set_image(self.hsi_rgb)
            else:
                self.viewer_rgb.viewer.clear()

            # Load HSI - search recursively for .hdr files
            # Common patterns: folder/*.hdr, folder/HS/results/*.hdr, folder/**/*.hdr
            hdr_files = list(folder.glob("*.hdr"))
            if not hdr_files:
                # Try HS/results subfolder (common pattern)
                hdr_files = list(folder.glob("HS/results/*.hdr"))
            if not hdr_files:
                # Try recursive search
                hdr_files = list(folder.glob("**/*.hdr"))

            # Filter to REFLECTANCE files if multiple found
            if len(hdr_files) > 1:
                reflectance_files = [f for f in hdr_files if 'REFLECTANCE' in f.name.upper()]
                if reflectance_files:
                    hdr_files = reflectance_files

            if hdr_files:
                reader = ENVIReader(str(hdr_files[0]))
                self.cube = reader.read()
                self.wavelengths = reader.get_wavelengths()

                if self.cube is not None:
                    # Cube is already in (H, W, C) format from ENVIReader
                    num_bands = self.cube.shape[2]
                    self.band_slider.setMaximum(num_bands - 1)
                    self.band_slider.setValue(num_bands // 2)
                    self._update_hsi_band()

                self.run_inference_btn.setEnabled(True)
                self.progress_label.setText(f"✓ Loaded: {self.cube.shape} (H,W,C) from {hdr_files[0].name}")
            else:
                show_error(self, "Warning", f"No .hdr file found in {folder} or subdirectories")
                self.cube = None
                self.run_inference_btn.setEnabled(False)

            # Clear previous
            self.prob_map = None
            self.grid_results = None
            self.viewer_prob.viewer.clear()
            self.viewer_thresh.viewer.clear()
            self.viewer_morph.viewer.clear()
            self.viewer_final.viewer.clear()
            self._disable_postprocess()
            self.stats_text.clear()

        except Exception as e:
            show_error(self, "Error", f"Failed to load: {e}")
            log_error("Load sample failed", e)
        finally:
            self.progress_label.setText("")

    def _update_hsi_band(self):
        """Update HSI band display - shows grayscale with 90° rotation to match RGB."""
        if self.cube is None:
            return

        band_idx = self.band_slider.value()
        # get_band_by_index returns 2D array (H, W) - correctly handles the cube format
        band = get_band_by_index(self.cube, band_idx)

        # Normalize to uint8 for display
        band_norm = normalize_to_uint8(band, method="percentile")

        # Apply 90° clockwise rotation to match RGB orientation
        # (same as in old project: GIF_HSI.py uses np.rot90(norm, k=-1))
        band_rotated = np.rot90(band_norm, k=-1)

        # Display as GRAYSCALE (2D array goes to Format_Grayscale8)
        self.viewer_hsi.viewer.set_image(band_rotated)

        # Update label with wavelength info
        if self.wavelengths is not None and band_idx < len(self.wavelengths):
            wl = self.wavelengths[band_idx]
            self.band_label.setText(f"Band: {band_idx}/{self.band_slider.maximum()} ({wl:.1f} nm)")
        else:
            self.band_label.setText(f"Band: {band_idx}/{self.band_slider.maximum()}")

    def _go_to_wavelength(self):
        if self.wavelengths is None:
            show_info(self, "Info", "No wavelength data available")
            return
        target = self.wl_spin.value()
        idx = np.argmin(np.abs(self.wavelengths - target))
        self.band_slider.setValue(idx)


    def _run_inference(self):
        if self.cube is None:
            return
        model_path = self.model_combo.currentData()
        if not model_path:
            show_error(self, "Error", "No model selected")
            return
        try:
            import joblib
            self.model = joblib.load(model_path)
            from app.config.types import PreprocessConfig
            preprocess_cfg = PreprocessConfig(
                use_snv=True,
                wavelengths=self.wavelengths,
                wl_min=settings.wl_min if self.wavelengths is not None else None,
                wl_max=settings.wl_max if self.wavelengths is not None else None
            )

            self.inference_worker = InferenceWorker(self.cube, self.model, preprocess_cfg)
            self.inference_worker.finished.connect(self._on_inference_done)
            self.inference_worker.error.connect(self._on_inference_error)
            self.inference_worker.progress.connect(lambda msg: self.progress_label.setText(msg))
            self.run_inference_btn.setEnabled(False)
            self.inference_worker.start()
        except Exception as e:
            show_error(self, "Error", f"Inference failed: {e}")
            log_error("Inference failed", e)

    def _on_inference_done(self, prob_map):
        self.prob_map = prob_map
        self.run_inference_btn.setEnabled(True)
        self.progress_label.setText("✓ Inference complete")

        # Apply 90° rotation to match HSI band orientation
        prob_rotated = np.rot90(prob_map, k=-1)

        # Display prob map with colormap (hot = good for probability)
        prob_vis = normalize_to_uint8(prob_rotated, method="percentile")
        prob_colored = apply_colormap(prob_vis / 255.0, name="hot")
        self.viewer_prob.viewer.set_image(prob_colored)

        self._enable_postprocess()
        self._rerun_postprocess()

    def _on_inference_error(self, msg):
        self.run_inference_btn.setEnabled(True)
        self.progress_label.setText("✗ Inference failed")
        show_error(self, "Inference Error", msg)

    def _disable_postprocess(self):
        for w in [self.thresh_spin, self.morph_spin, self.min_area_spin,
                  self.exclude_border_check, self.border_margin_spin]:
            w.setEnabled(False)
        self.post_status.setText("⚠ Run inference first")
        self.post_status.setStyleSheet("color: orange;")
        self.run_grid_btn.setEnabled(False)
        self.save_grid_btn.setEnabled(False)
        self.grid_status.setText("⚠ Run inference first")
        self.grid_status.setStyleSheet("color: orange; font-size: 10px;")

    def _enable_postprocess(self):
        for w in [self.thresh_spin, self.morph_spin, self.min_area_spin,
                  self.exclude_border_check, self.border_margin_spin]:
            w.setEnabled(True)
        self.post_status.setText("✓ Controls active")
        self.post_status.setStyleSheet("color: green;")
        self.run_grid_btn.setEnabled(True)
        self.grid_status.setText("✓ Grid available")
        self.grid_status.setStyleSheet("color: green; font-size: 10px;")

    def _rerun_postprocess(self):
        if self.prob_map is None:
            return
        try:
            morph = self.morph_spin.value()
            if morph > 0 and morph % 2 == 0:
                morph += 1

            config = PostprocessConfig(
                prob_threshold=self.thresh_spin.value(),
                morph_close_size=morph,
                min_blob_area=self.min_area_spin.value(),
                exclude_border=self.exclude_border_check.isChecked(),
                border_margin_px=self.border_margin_spin.value()
            )
            pipeline = PostprocessPipeline(config)
            final_mask, stats, debug = pipeline.run_debug(self.prob_map)

            # Apply 90° rotation to masks to match HSI orientation
            mask_thresh_rotated = np.rot90(debug['mask_threshold'].astype(np.uint8) * 255, k=-1)
            mask_morph_rotated = np.rot90(debug['mask_after_morph'].astype(np.uint8) * 255, k=-1)
            final_mask_rotated = np.rot90(final_mask, k=-1)

            # Display threshold and morph masks as binary images (rotated)
            self.viewer_thresh.viewer.set_image(mask_thresh_rotated)
            self.viewer_morph.viewer.set_image(mask_morph_rotated)

            # Use RGB base image WITHOUT rotation (RGB is already in correct orientation)
            base = self.camera_rgb.copy() if self.camera_rgb is not None else (
                self.hsi_rgb.copy() if self.hsi_rgb is not None else
                np.zeros((*final_mask.shape, 3), dtype=np.uint8))

            # Display final: RGB base (not rotated) + rotated mask overlay
            self.viewer_final.viewer.set_image(base)
            self.viewer_final.viewer.set_overlay(final_mask_rotated, alpha=0.6)

            self.stats_text.setText(
                f"Blobs Before: {stats['num_blobs_before']}\n"
                f"Blobs After: {stats['num_blobs_after']}\n"
                f"Positive Pixels: {stats['total_positive_pixels']}\n"
                f"Crack Ratio: {stats['crack_ratio']:.4f}"
            )
        except Exception as e:
            log_error("Postprocess failed", e)

    def _run_grid_search(self):
        if self.prob_map is None:
            return
        try:
            from app.tuning import get_default_param_grid
            param_grid = get_default_param_grid()
            total = 1
            for v in param_grid.values():
                total *= len(v)

            reply = QMessageBox.question(self, "Confirm", f"Test {total} combinations?",
                                          QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

            self.grid_worker = GridSearchWorker(self.prob_map, param_grid)
            self.grid_worker.finished.connect(self._on_grid_done)
            self.grid_worker.error.connect(self._on_grid_error)
            self.grid_worker.progress.connect(lambda msg: self.grid_status.setText(msg))
            self.run_grid_btn.setEnabled(False)
            self.grid_worker.start()
        except Exception as e:
            show_error(self, "Error", f"Grid search failed: {e}")

    def _on_grid_done(self, results_df):
        self.grid_results = results_df
        self.run_grid_btn.setEnabled(True)
        self.save_grid_btn.setEnabled(True)
        self.grid_status.setText(f"✓ {len(results_df)} combos")

        top_n = 10
        df = results_df.head(top_n)
        self.grid_table.setRowCount(len(df))
        self.grid_table.setColumnCount(len(df.columns))
        self.grid_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i, (_, row) in enumerate(df.iterrows()):
            for j, (col, val) in enumerate(row.items()):
                item = QTableWidgetItem(f"{val:.4f}" if isinstance(val, float) else str(val))
                self.grid_table.setItem(i, j, item)
        self.grid_table.resizeColumnsToContents()
        self.grid_table.selectRow(0)

    def _on_grid_error(self, msg):
        self.run_grid_btn.setEnabled(True)
        self.grid_status.setText("✗ Failed")
        show_error(self, "Grid Error", msg)

    def _on_grid_row_selected(self):
        if self.grid_results is None:
            return
        selected = self.grid_table.selectedIndexes()
        if not selected:
            return
        row_idx = selected[0].row()
        row = self.grid_results.iloc[row_idx]

        if 'prob_threshold' in row:
            self.thresh_spin.setValue(float(row['prob_threshold']))
        if 'morph_close_size' in row:
            self.morph_spin.setValue(int(row['morph_close_size']))
        if 'min_blob_area' in row:
            self.min_area_spin.setValue(int(row['min_blob_area']))
        if 'exclude_border' in row:
            self.exclude_border_check.setChecked(bool(row['exclude_border']))
        if 'border_margin_px' in row:
            self.border_margin_spin.setValue(int(row['border_margin_px']))

    def _save_grid_results(self):
        if self.grid_results is None:
            return
        try:
            from app.tuning import save_grid_results
            sample_id = self.current_folder.name if self.current_folder else "unknown"
            if self.current_mode == "dataset" and self.dataset_df is not None:
                row = self.dataset_df.iloc[self.current_index]
                sample_id = f"{row['grape_id']}_{row.get('week_date', '')}"
            output_dir = settings.results_dir / "single_sample_grids"
            path = save_grid_results(self.grid_results, output_dir, sample_id)
            show_info(self, "Saved", f"Saved to:\n{path}")
        except Exception as e:
            show_error(self, "Error", f"Save failed: {e}")

    def reset_state(self):
        """Clear all loaded data and reset UI."""
        self.cube = None
        self.wavelengths = None
        self.hsi_rgb = None
        self.camera_rgb = None
        self.prob_map = None
        self.grid_results = None
        self.model = None

        for v in [self.viewer_rgb, self.viewer_hsi, self.viewer_prob,
                  self.viewer_thresh, self.viewer_morph, self.viewer_final]:
            v.viewer.clear()

        self.stats_text.clear()
        self.grid_table.setRowCount(0)
        self._disable_postprocess()
        self.progress_label.setText("")
        self.band_slider.setValue(0)
        self.band_slider.setMaximum(0)
        gc.collect()

    def stop_workers(self):
        """Stop any running workers."""
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait(2000)
        if self.grid_worker and self.grid_worker.isRunning():
            self.grid_worker.stop()
            self.grid_worker.wait(2000)


# ============================================================================
# Tab B: Dataset Tuning
# ============================================================================

class DatasetTuningTab(QWidget):
    """Scientific dataset tuning with Optuna and final test."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.best_params = None
        self.model = None
        self.model_adapter = None
        self.output_dir = None
        self.optuna_worker = None
        self.final_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Data Selection
        data_group = QGroupBox("Dataset Selection")
        data_layout = QFormLayout(data_group)

        trainval_row = QHBoxLayout()
        self.trainval_edit = QLineEdit()
        self.trainval_edit.setPlaceholderText("Train/Val CSV (70/30 split)...")
        self.trainval_edit.setReadOnly(True)
        trainval_btn = QPushButton("Browse...")
        trainval_btn.clicked.connect(lambda: self._select_csv("trainval"))
        trainval_row.addWidget(self.trainval_edit)
        trainval_row.addWidget(trainval_btn)
        data_layout.addRow("Train/Val CSV:", trainval_row)

        test_row = QHBoxLayout()
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("Test CSV (locked, used once)...")
        self.test_edit.setReadOnly(True)
        test_btn = QPushButton("Browse...")
        test_btn.clicked.connect(lambda: self._select_csv("test"))
        test_row.addWidget(self.test_edit)
        test_row.addWidget(test_btn)
        data_layout.addRow("Test CSV:", test_row)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self._refresh_models()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(refresh_btn)
        data_layout.addRow("Model:", model_row)

        layout.addWidget(data_group)

        # Config
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.30)
        config_layout.addRow("Val Split:", self.val_split_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        config_layout.addRow("Seed:", self.seed_spin)

        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(5, 500)
        self.trials_spin.setValue(50)
        config_layout.addRow("N Trials:", self.trials_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["f2", "f1", "accuracy", "macro_f1"])
        config_layout.addRow("Metric:", self.metric_combo)

        layout.addWidget(config_group)

        # Actions
        actions_group = QGroupBox("Pipeline Steps")
        actions_layout = QHBoxLayout(actions_group)

        self.prepare_btn = QPushButton("1. Prepare Splits")
        self.prepare_btn.clicked.connect(self._prepare_splits)
        actions_layout.addWidget(self.prepare_btn)

        self.optuna_btn = QPushButton("2. Run Optuna (Val)")
        self.optuna_btn.clicked.connect(self._run_optuna)
        self.optuna_btn.setEnabled(False)
        actions_layout.addWidget(self.optuna_btn)

        self.final_btn = QPushButton("3. Final Test (Once)")
        self.final_btn.clicked.connect(self._run_final_test)
        self.final_btn.setEnabled(False)
        actions_layout.addWidget(self.final_btn)

        layout.addWidget(actions_group)

        # Progress
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.splits_table = QTableWidget()
        self.splits_table.setColumnCount(4)
        self.splits_table.setHorizontalHeaderLabels(["Split", "Total", "Class 0", "Class 1"])
        self.splits_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.splits_table.setMaximumHeight(100)
        results_layout.addWidget(self.splits_table)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)
        layout.addStretch()

    def _refresh_models(self):
        self.model_combo.clear()
        if settings.models_dir.exists():
            for ext in ['.joblib', '.pkl', '.pth']:
                for f in settings.models_dir.glob(f'*{ext}'):
                    self.model_combo.addItem(f.name, str(f))

    def _select_csv(self, csv_type):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {csv_type} CSV",
                                               str(settings.default_search_folder), "CSV Files (*.csv)")
        if path:
            if csv_type == "trainval":
                self.trainval_edit.setText(path)
            else:
                self.test_edit.setText(path)

    def _prepare_splits(self):
        try:
            trainval_path = self.trainval_edit.text()
            test_path = self.test_edit.text()
            if not trainval_path or not test_path:
                show_error(self, "Error", "Select both CSVs")
                return

            from app.data.dataset import load_and_prepare_splits, get_class_distribution

            self.progress_label.setText("Preparing splits...")
            QApplication.processEvents()

            self.train_df, self.val_df, self.test_df = load_and_prepare_splits(
                trainval_path, test_path,
                val_size=self.val_split_spin.value(),
                random_state=self.seed_spin.value()
            )

            self.splits_table.setRowCount(3)
            for i, (name, df) in enumerate([("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]):
                dist = get_class_distribution(df)
                self.splits_table.setItem(i, 0, QTableWidgetItem(name))
                self.splits_table.setItem(i, 1, QTableWidgetItem(str(len(df))))
                self.splits_table.setItem(i, 2, QTableWidgetItem(str(dist.get(0, 0))))
                self.splits_table.setItem(i, 3, QTableWidgetItem(str(dist.get(1, 0))))

            self.progress_label.setText("✓ Splits ready")
            self.optuna_btn.setEnabled(True)
            self.results_text.append(f"[{datetime.now():%H:%M:%S}] Splits prepared.\n")

        except Exception as e:
            log_error("Prepare splits failed", e)
            show_error(self, "Error", str(e))

    def _load_model(self):
        model_path = self.model_combo.currentData()
        if not model_path:
            raise ValueError("Select a model")
        if self.model is None:
            from app.models.loader_new import load_model
            from app.models.adapters_new import SklearnAdapter
            self.model = load_model(model_path)
            self.model_adapter = SklearnAdapter(self.model, name=Path(model_path).stem)

    def _create_prob_map_fn(self) -> Callable[[str], np.ndarray]:
        from app.io.envi import ENVIReader
        from app.inference.prob_map import build_prob_map
        from app.config.types import PreprocessConfig
        adapter = self.model_adapter

        def fn(image_path: str) -> np.ndarray:
            folder = Path(image_path)
            if folder.is_file():
                folder = folder.parent

            # Search for .hdr files (recursive pattern)
            hdr = list(folder.glob("*.hdr"))
            if not hdr:
                hdr = list(folder.glob("HS/results/*.hdr"))
            if not hdr:
                hdr = list(folder.glob("**/*.hdr"))

            # Prefer REFLECTANCE files
            if len(hdr) > 1:
                refl = [f for f in hdr if 'REFLECTANCE' in f.name.upper()]
                if refl:
                    hdr = refl

            if not hdr:
                raise FileNotFoundError(f"No .hdr in {folder} or subdirectories")

            reader = ENVIReader(str(hdr[0]))
            cube = reader.read()
            wl = reader.get_wavelengths()
            cfg = PreprocessConfig(use_snv=True, wavelengths=wl,
                                   wl_min=settings.wl_min if wl else None,
                                   wl_max=settings.wl_max if wl else None)
            return build_prob_map(cube, adapter, cfg, target_class_index=1, chunk_size=100_000)
        return fn

    def _run_optuna(self):
        try:
            if self.val_df is None:
                show_error(self, "Error", "Prepare splits first")
                return
            self._load_model()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = str(settings.results_dir / f"tuning_{timestamp}")
            prob_fn = self._create_prob_map_fn()

            self.optuna_btn.setEnabled(False)
            self.progress_label.setText("Running Optuna...")

            self.optuna_worker = OptunaWorker(
                self.train_df, self.val_df, prob_fn,
                self.trials_spin.value(), self.seed_spin.value(),
                self.output_dir, self.metric_combo.currentText()
            )
            self.optuna_worker.finished.connect(self._on_optuna_done)
            self.optuna_worker.error.connect(self._on_optuna_error)
            self.optuna_worker.progress.connect(lambda m: self.progress_label.setText(m))
            self.optuna_worker.start()

        except Exception as e:
            log_error("Optuna setup failed", e)
            show_error(self, "Error", str(e))
            self.optuna_btn.setEnabled(True)

    def _on_optuna_done(self, result):
        self.best_params = result["best_params"]
        self.optuna_btn.setEnabled(True)
        self.final_btn.setEnabled(True)
        self.progress_label.setText("✓ Optuna complete")
        self.results_text.append(
            f"\n[{datetime.now():%H:%M:%S}] Optuna done:\n"
            f"  Trials: {result['n_trials']}\n"
            f"  Best: {self.best_params}\n"
        )

    def _on_optuna_error(self, msg):
        self.optuna_btn.setEnabled(True)
        self.progress_label.setText("✗ Optuna failed")
        show_error(self, "Optuna Error", msg)

    def _run_final_test(self):
        try:
            if self.best_params is None:
                show_error(self, "Error", "Run Optuna first")
                return
            prob_fn = self._create_prob_map_fn()

            self.final_btn.setEnabled(False)
            self.progress_label.setText("Running final test...")

            self.final_worker = FinalEvalWorker(
                self.test_df, prob_fn, self.best_params,
                self.output_dir, self.train_df, self.val_df
            )
            self.final_worker.finished.connect(self._on_final_done)
            self.final_worker.error.connect(self._on_final_error)
            self.final_worker.progress.connect(lambda m: self.progress_label.setText(m))
            self.final_worker.start()

        except Exception as e:
            log_error("Final test failed", e)
            show_error(self, "Error", str(e))
            self.final_btn.setEnabled(True)

    def _on_final_done(self, metrics):
        self.final_btn.setEnabled(True)
        self.progress_label.setText("✓ Final test complete")
        self.results_text.append(
            f"\n[{datetime.now():%H:%M:%S}] FINAL TEST:\n"
            f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n"
            f"  Precision: {metrics.get('precision', 0):.4f}\n"
            f"  Recall: {metrics.get('recall', 0):.4f}\n"
            f"  F1: {metrics.get('f1', 0):.4f}\n"
            f"  F2: {metrics.get('f2', 0):.4f}\n"
            f"  Output: {self.output_dir}\n"
        )
        show_info(self, "Complete", f"F2: {metrics.get('f2', 0):.4f}\nSaved to: {self.output_dir}")

    def _on_final_error(self, msg):
        self.final_btn.setEnabled(True)
        self.progress_label.setText("✗ Final test failed")
        show_error(self, "Final Test Error", msg)

    def reset_state(self):
        """Reset tuning state."""
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.best_params = None
        self.model = None
        self.model_adapter = None
        self.splits_table.setRowCount(0)
        self.results_text.clear()
        self.optuna_btn.setEnabled(False)
        self.final_btn.setEnabled(False)
        self.progress_label.setText("")

    def stop_workers(self):
        if self.optuna_worker and self.optuna_worker.isRunning():
            self.optuna_worker.stop()
            self.optuna_worker.wait(2000)
        if self.final_worker and self.final_worker.isRunning():
            self.final_worker.stop()
            self.final_worker.wait(2000)


# ============================================================================
# Main Window
# ============================================================================

class MainWindow(QMainWindow):
    """Main window with tabs for visual debug and dataset tuning."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Classification Inference UI")
        self.setMinimumSize(1400, 900)
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header with reset button
        header_layout = QHBoxLayout()
        header = QLabel("Binary Classification Inference UI")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(header)
        header_layout.addStretch()

        reset_btn = QPushButton("🔄 Reset UI State")
        reset_btn.clicked.connect(self._reset_all)
        header_layout.addWidget(reset_btn)
        layout.addLayout(header_layout)

        # Tabs
        self.tabs = QTabWidget()

        self.visual_tab = VisualDebugTab()
        self.tabs.addTab(self.visual_tab, "Visual Debug")

        self.tuning_tab = DatasetTuningTab()
        self.tabs.addTab(self.tuning_tab, "Dataset Tuning")

        layout.addWidget(self.tabs)
        self.statusBar().showMessage("Ready")

    def _reset_all(self):
        """Reset all state in both tabs."""
        reply = QMessageBox.question(self, "Confirm Reset",
                                      "Clear all loaded data and reset UI?",
                                      QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.visual_tab.reset_state()
            self.tuning_tab.reset_state()
            self.statusBar().showMessage("UI state cleared")

    def closeEvent(self, event):
        """Clean shutdown."""
        self.visual_tab.stop_workers()
        self.tuning_tab.stop_workers()
        gc.collect()
        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Application entrypoint."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
