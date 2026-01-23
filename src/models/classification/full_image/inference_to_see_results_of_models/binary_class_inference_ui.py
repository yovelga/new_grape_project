"""
Binary Classification Inference UI - Unified Version

Comprehensive UI combining:
- Tab A: Visual Debug / Image Playground (Interactive image analysis)
- Tab B: Optuna Tuning (Hyperparameter optimization)

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
    QRadioButton, QCheckBox, QGridLayout, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from app.config.settings import settings
from app.utils.logging import setup_logger
from app.ui import ImageViewer, OptunaTabWidget
from app.models import ModelManager
from app.io import ENVIReader, load_rgb, get_band_by_index, find_hsi_rgb, find_canon_rgb
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

    def __init__(self, cube, model, preprocess_cfg, target_class_index):
        super().__init__()
        self.cube = cube
        self.model = model
        self.preprocess_cfg = preprocess_cfg
        self.target_class_index = target_class_index
        self._stop = False

    def run(self):
        try:
            from app.inference.prob_map import build_prob_map
            self.progress.emit("Running inference...")
            prob_map = build_prob_map(
                self.cube, self.model, self.preprocess_cfg,
                target_class_index=self.target_class_index, chunk_size=100_000
            )
            if not self._stop:
                self.finished.emit(prob_map)
        except Exception as e:
            if not self._stop:
                log_error("Inference failed", e)
                self.error.emit(str(e))

    def stop(self):
        self._stop = True

# ============================================================================
# Tab A: Visual Debug / Image Playground
# ============================================================================

class VisualDebugTab(QWidget):
    """Visual debugging tab with 2x3 viewer grid and live postprocess. CSV-only mode."""

    def __init__(self, model_manager: ModelManager, parent=None):
        super().__init__(parent)

        # Model manager (shared)
        self.model_manager = model_manager

        # State (CSV-only, no folder mode)
        self.dataset_df = None
        self.current_index = 0
        self.cube = None
        self.wavelengths = None
        self.hsi_rgb = None
        self.camera_rgb = None
        self.model = None
        self.prob_map = None

        # Patch analysis state
        self.patch_flag_mask = None
        self.patch_stats = None

        # Workers
        self.inference_worker = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # =====================================================================
        # MAIN HORIZONTAL SPLITTER: Left Controls (1/3) | Right Images (2/3)
        # =====================================================================
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(5)

        # =====================================================================
        # LEFT SIDE: Controls Sidebar with internal TOP/BOTTOM splitter
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(0)
        left_layout.setContentsMargins(0, 0, 4, 0)

        # Vertical splitter inside left panel: TOP (Sample Selection) | BOTTOM (Controls)
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setHandleWidth(4)

        # ---------------------------------------------------------------------
        # LEFT TOP: Sample Selection (CSV-only, Excel-like table)
        # ---------------------------------------------------------------------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setSpacing(4)
        top_layout.setContentsMargins(4, 4, 4, 4)

        # CSV Load row
        csv_row = QHBoxLayout()
        csv_row.setSpacing(4)
        csv_btn = QPushButton("📄 Load CSV...")
        csv_btn.clicked.connect(self._load_csv)
        csv_row.addWidget(csv_btn)
        self.csv_label = QLabel("No CSV loaded")
        self.csv_label.setStyleSheet("font-size: 11px; color: #666;")
        csv_row.addWidget(self.csv_label, stretch=1)
        top_layout.addLayout(csv_row)

        # Navigation row: Prev | Index | Go | Next + Auto-run checkbox
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setFixedWidth(65)
        self.prev_btn.clicked.connect(self._navigate_prev)
        nav_row.addWidget(self.prev_btn)

        self.index_spin = QSpinBox()
        self.index_spin.setMinimum(0)
        self.index_spin.setFixedWidth(60)
        nav_row.addWidget(self.index_spin)

        jump_btn = QPushButton("Go")
        jump_btn.setFixedWidth(35)
        jump_btn.clicked.connect(self._jump_to_index)
        nav_row.addWidget(jump_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setFixedWidth(65)
        self.next_btn.clicked.connect(self._navigate_next)
        nav_row.addWidget(self.next_btn)

        # Auto-run inference on navigation
        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(True)
        self.auto_run_check.setToolTip("Automatically run inference when navigating with Prev/Next")
        self.auto_run_check.setStyleSheet("font-size: 10px;")
        nav_row.addWidget(self.auto_run_check)
        nav_row.addStretch()
        top_layout.addLayout(nav_row)

        # Search row
        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        search_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("grape_id")
        self.search_edit.returnPressed.connect(self._search_grape_id)
        search_row.addWidget(self.search_edit, stretch=1)
        search_btn = QPushButton("Find")
        search_btn.setFixedWidth(45)
        search_btn.clicked.connect(self._search_grape_id)
        search_row.addWidget(search_btn)
        top_layout.addLayout(search_row)

        # CSV Table (Excel-like, expandable)
        self.sample_table = QTableWidget()
        self.sample_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sample_table.setSelectionMode(QTableWidget.SingleSelection)
        self.sample_table.itemSelectionChanged.connect(self._on_table_selection)
        self.sample_table.cellDoubleClicked.connect(self._on_table_double_click)
        self.sample_table.setStyleSheet("font-size: 11px;")
        self.sample_table.horizontalHeader().setStretchLastSection(True)
        top_layout.addWidget(self.sample_table, stretch=1)

        # Load Sample button (prominent)
        self.load_sample_btn = QPushButton("📂 LOAD SELECTED SAMPLE")
        self.load_sample_btn.setEnabled(False)
        self.load_sample_btn.clicked.connect(self._load_current_sample)
        self.load_sample_btn.setMinimumHeight(36)
        self.load_sample_btn.setStyleSheet(
            "font-weight: bold; font-size: 12px; background: #2196F3; color: white;"
        )
        top_layout.addWidget(self.load_sample_btn)

        left_splitter.addWidget(top_widget)

        # ---------------------------------------------------------------------
        # LEFT BOTTOM: Model + Filters + Stats/Grid (scrollable)
        # ---------------------------------------------------------------------
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setSpacing(6)
        bottom_layout.setContentsMargins(4, 4, 4, 4)

        # ----- HSI Band Panel -----
        band_group = QGroupBox("HSI Band")
        band_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        band_layout = QVBoxLayout(band_group)
        band_layout.setSpacing(3)

        self.band_label = QLabel("Band: 0 / 0")
        self.band_label.setStyleSheet("font-size: 10px;")
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(0)
        self.band_slider.valueChanged.connect(self._update_hsi_band)

        wl_row = QHBoxLayout()
        wl_row.setSpacing(2)
        wl_row.addWidget(QLabel("nm:"))
        self.wl_spin = QSpinBox()
        self.wl_spin.setRange(400, 2500)
        self.wl_spin.setValue(700)
        self.wl_spin.setFixedWidth(60)
        wl_btn = QPushButton("Go")
        wl_btn.setFixedWidth(35)
        wl_btn.clicked.connect(self._go_to_wavelength)
        wl_row.addWidget(self.wl_spin)
        wl_row.addWidget(wl_btn)
        wl_row.addStretch()

        band_layout.addWidget(self.band_label)
        band_layout.addWidget(self.band_slider)
        band_layout.addLayout(wl_row)
        bottom_layout.addWidget(band_group)

        # ----- Model Panel (compact) -----
        model_group = QGroupBox("Model")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        model_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(2)
        model_layout.setContentsMargins(6, 8, 6, 6)

        # Model selection row: Path edit + Browse + Load buttons
        model_select_row = QHBoxLayout()
        model_select_row.setSpacing(4)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("No model selected")
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setStyleSheet("font-size: 10px;")
        model_select_row.addWidget(self.model_path_edit, stretch=1)
        
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.setStyleSheet("font-size: 10px;")
        browse_model_btn.clicked.connect(self._browse_model)
        model_select_row.addWidget(browse_model_btn)
        
        load_model_btn = QPushButton("Load")
        load_model_btn.setStyleSheet("font-size: 10px; font-weight: bold;")
        load_model_btn.clicked.connect(self._load_selected_model)
        model_select_row.addWidget(load_model_btn)
        
        model_layout.addLayout(model_select_row)
        
        # Model status label
        self.model_status_label = QLabel("⚠ No model loaded")
        self.model_status_label.setStyleSheet(
            "color: #e67e22; font-size: 9px; font-weight: bold; "
            "padding: 3px; background-color: #fef5e7; border-radius: 3px;"
        )
        model_layout.addWidget(self.model_status_label)

        class_row = QHBoxLayout()
        class_row.setSpacing(4)
        class_row.addWidget(QLabel("Target:"))
        self.target_class_combo = QComboBox()
        self.target_class_combo.setStyleSheet("font-size: 10px;")
        self._set_target_class_options(2)
        class_row.addWidget(self.target_class_combo, stretch=1)
        model_layout.addLayout(class_row)

        preprocess_label = QLabel(f"WL {settings.wl_min}-{settings.wl_max} + SNV (always ON)")
        preprocess_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        model_layout.addWidget(preprocess_label)

        run_row = QHBoxLayout()
        run_row.setSpacing(4)
        self.run_inference_btn = QPushButton("▶ RUN")
        self.run_inference_btn.setEnabled(False)
        self.run_inference_btn.clicked.connect(self._run_inference)
        self.run_inference_btn.setFixedHeight(26)
        self.run_inference_btn.setStyleSheet("font-weight: bold; font-size: 11px; background: #4CAF50; color: white;")
        run_row.addWidget(self.run_inference_btn)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-size: 10px;")
        run_row.addWidget(self.progress_label, stretch=1)
        model_layout.addLayout(run_row)

        bottom_layout.addWidget(model_group)

        # ----- Filters Panel (Live Postprocess) -----
        filter_group = QGroupBox("Filters (Live Postprocess)")
        filter_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setSpacing(4)

        # Row 0: Threshold (high precision) + Morph
        filter_layout.addWidget(QLabel("Thresh:"), 0, 0)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.0, 1.0)
        self.thresh_spin.setDecimals(8)
        self.thresh_spin.setSingleStep(0.001)
        self.thresh_spin.setValue(0.9)
        self.thresh_spin.valueChanged.connect(self._on_threshold_changed)
        filter_layout.addWidget(self.thresh_spin, 0, 1)

        filter_layout.addWidget(QLabel("Morph:"), 0, 2)
        self.morph_spin = QSpinBox()
        self.morph_spin.setRange(0, 15)
        self.morph_spin.setSingleStep(2)
        self.morph_spin.setValue(0)
        self.morph_spin.valueChanged.connect(self._rerun_postprocess)
        filter_layout.addWidget(self.morph_spin, 0, 3)

        # Row 1: Min Area + Border
        filter_layout.addWidget(QLabel("MinArea:"), 1, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000)
        self.min_area_spin.setSingleStep(10)
        self.min_area_spin.setValue(0)
        self.min_area_spin.valueChanged.connect(self._rerun_postprocess)
        filter_layout.addWidget(self.min_area_spin, 1, 1)

        filter_layout.addWidget(QLabel("Border:"), 1, 2)
        self.border_margin_spin = QSpinBox()
        self.border_margin_spin.setRange(0, 100)
        self.border_margin_spin.setValue(0)
        self.border_margin_spin.valueChanged.connect(self._rerun_postprocess)
        filter_layout.addWidget(self.border_margin_spin, 1, 3)

        # Row 2: Exclude border + Label
        self.exclude_border_check = QCheckBox("Exclude Border")
        self.exclude_border_check.setStyleSheet("font-size: 10px;")
        self.exclude_border_check.stateChanged.connect(self._rerun_postprocess)
        filter_layout.addWidget(self.exclude_border_check, 2, 0, 1, 2)

        filter_layout.addWidget(QLabel("Label:"), 2, 2)
        self.label_combo = QComboBox()
        self.label_combo.addItems(["Auto", "Regular", "Crack"])
        self.label_combo.setStyleSheet("font-size: 10px;")
        self.label_combo.currentIndexChanged.connect(self._rerun_postprocess)
        filter_layout.addWidget(self.label_combo, 2, 3)

        self.post_status = QLabel("⚠ Run inference first")
        self.post_status.setStyleSheet("color: orange; font-size: 9px;")
        filter_layout.addWidget(self.post_status, 3, 0, 1, 4)

        bottom_layout.addWidget(filter_group)

        # ----- Stats + Patch Analysis Panel -----
        stats_group = QGroupBox("Stats & Patch Analysis")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(3)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(70)
        self.stats_text.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self.stats_text)

        # Patch Analysis Controls
        patch_controls_row = QHBoxLayout()
        patch_controls_row.setSpacing(4)
        
        patch_controls_row.addWidget(QLabel("Patch Size:"))
        self.patch_size_combo = QComboBox()
        self.patch_size_combo.addItems(["4", "8", "16", "32", "64", "128"])
        self.patch_size_combo.setCurrentIndex(3)  # Default: 32
        self.patch_size_combo.setStyleSheet("font-size: 10px;")
        self.patch_size_combo.currentIndexChanged.connect(self._rerun_patch_analysis)
        patch_controls_row.addWidget(self.patch_size_combo)
        
        patch_controls_row.addWidget(QLabel("Patch Crack %:"))
        self.patch_thresh_spin = QDoubleSpinBox()
        self.patch_thresh_spin.setRange(0.0, 100.0)
        self.patch_thresh_spin.setDecimals(1)
        self.patch_thresh_spin.setSingleStep(1.0)
        self.patch_thresh_spin.setValue(10.0)
        self.patch_thresh_spin.setStyleSheet("font-size: 10px;")
        self.patch_thresh_spin.valueChanged.connect(self._rerun_patch_analysis)
        patch_controls_row.addWidget(self.patch_thresh_spin)
        
        stats_layout.addLayout(patch_controls_row)

        bottom_layout.addWidget(stats_group)
        bottom_layout.addStretch()

        left_splitter.addWidget(bottom_widget)

        # Set default splitter sizes: TOP 40%, BOTTOM 60% (compact CSV table)
        left_splitter.setSizes([400, 600])

        left_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_widget)

        # =====================================================================
        # RIGHT SIDE: 2x3 Image Grid
        # =====================================================================
        images_widget = QWidget()
        images_layout = QVBoxLayout(images_widget)
        images_layout.setContentsMargins(4, 0, 0, 0)
        images_layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        # Row 1: RGB views
        self.viewer_rgb_cam = self._create_viewer("RGB (Camera)")
        self.viewer_rgb_hsi = self._create_viewer("RGB (HSI)")
        self.viewer_rgb_extra = self._create_viewer("Results on RGB")
        self.viewer_rgb_patch = self._create_viewer("Patch Map on RGB")

        # Row 2: HSI + results
        self.viewer_hsi_band = self._create_viewer("HSI Band (Grayscale)")
        self.viewer_prob_hsi = self._create_viewer("Probability Map")
        self.viewer_blob_hsi = self._create_viewer("Blobs on HSI")
        self.viewer_hsi_patch = self._create_viewer("Patch Map on HSI")

        grid.addWidget(self.viewer_rgb_cam, 0, 0)
        grid.addWidget(self.viewer_rgb_hsi, 0, 1)
        grid.addWidget(self.viewer_rgb_extra, 0, 2)
        grid.addWidget(self.viewer_rgb_patch, 0, 3)

        grid.addWidget(self.viewer_hsi_band, 1, 0)
        grid.addWidget(self.viewer_prob_hsi, 1, 1)
        grid.addWidget(self.viewer_blob_hsi, 1, 2)
        grid.addWidget(self.viewer_hsi_patch, 1, 3)

        # Equal stretch for all cells
        for i in range(2):
            grid.setRowStretch(i, 1)
        for j in range(4):
            grid.setColumnStretch(j, 1)

        images_layout.addLayout(grid, stretch=1)
        main_splitter.addWidget(images_widget)

        # Set default splitter sizes: LEFT 33%, RIGHT 67%
        main_splitter.setSizes([330, 670])

        main_layout.addWidget(main_splitter)

        # Disable postprocess initially
        self._disable_postprocess()

    def _on_table_double_click(self, row, col):
        """Load sample on double-click."""
        self.current_index = row
        self._load_current_sample()

    def keyPressEvent(self, event):
        """Handle Enter key to load selected sample."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.sample_table.hasFocus() and self.dataset_df is not None:
                self._load_current_sample()
        super().keyPressEvent(event)

    def _create_viewer(self, label_text):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        label = QLabel(label_text)
        label.setStyleSheet("font-weight: bold; background: #e0e0e0; padding: 4px; font-size: 11px;")
        label.setAlignment(Qt.AlignCenter)
        viewer = ImageViewer()
        # Let viewer expand to fill available space
        layout.addWidget(label)
        layout.addWidget(viewer, stretch=1)
        widget.viewer = viewer
        widget.label = label
        return widget

    def _set_target_class_options(self, n_classes: int) -> None:
        current = self.target_class_combo.currentData()
        self.target_class_combo.blockSignals(True)
        self.target_class_combo.clear()
        for idx in range(n_classes):
            if n_classes == 2 and idx == 1:
                label = f"{idx} - Positive"
            elif n_classes == 2 and idx == 0:
                label = f"{idx} - Negative"
            else:
                label = f"{idx}"
            self.target_class_combo.addItem(label, idx)
        if current is not None and 0 <= int(current) < n_classes:
            self.target_class_combo.setCurrentIndex(int(current))
        else:
            default_idx = 1 if n_classes > 1 else 0
            self.target_class_combo.setCurrentIndex(default_idx)
        self.target_class_combo.blockSignals(False)


    def _browse_model(self):
        """Browse for model file."""
        default_dir = str(settings.models_dir) if settings.models_dir.exists() else str(Path.home())
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            default_dir,
            "Model Files (*.joblib *.pkl *.pth);;All Files (*.*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def _load_selected_model(self):
        """Load the selected model using ModelManager."""
        model_path = self.model_path_edit.text()
        if not model_path or model_path == "No model selected":
            show_error(self, "Error", "Please select a model file first")
            return
        
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            show_error(self, "Error", f"Model file not found: {model_path}")
            return
        
        try:
            # Load model using shared model manager
            model_info = self.model_manager.load_model(model_path_obj)
            self.model = self.model_manager.get_model()  # Get actual model object
            
            # Update target class options
            self._set_target_class_options(model_info.n_classes)
            
            # Update status
            self.model_status_label.setText(f"✓ Loaded: {model_path_obj.name}")
            self.model_status_label.setStyleSheet(
                "color: #27ae60; font-size: 9px; font-weight: bold; "
                "padding: 3px; background-color: #e8f8f5; border-radius: 3px;"
            )
            
            # Enable inference button if HSI is loaded
            if self.cube is not None:
                self.run_inference_btn.setEnabled(True)
            
            self.progress_label.setText(f"Model loaded: {model_info.n_classes} classes")
            
        except Exception as e:
            show_error(self, "Error", f"Failed to load model: {e}")
            log_error("Model load failed", e)
            self.model_status_label.setText("✗ Load failed")
            self.model_status_label.setStyleSheet(
                "color: #e74c3c; font-size: 9px; font-weight: bold; "
                "padding: 3px; background-color: #fadbd8; border-radius: 3px;"
            )

    def _load_csv(self):
        # Default to project's data folder or DATA_DIR from settings
        default_dir = ""
        if hasattr(settings, 'data_dir') and settings.data_dir and Path(settings.data_dir).exists():
            default_dir = str(settings.data_dir)
        else:
            # Use the local data/ folder in this UI project
            ui_project_data = Path(__file__).parent / "data"
            if not ui_project_data.exists():
                ui_project_data.mkdir(parents=True, exist_ok=True)
            default_dir = str(ui_project_data)
        
        path, _ = QFileDialog.getOpenFileName(self, "Load Dataset CSV", default_dir, "CSV Files (*.csv)")
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
            # Auto-set label combo based on sample label from dataset
            row = self.dataset_df.iloc[self.current_index]
            label = row.get('label', None)
            if label == 0:
                self.label_combo.setCurrentIndex(1)  # "0 - Regular (FP=red)"
            elif label == 1:
                self.label_combo.setCurrentIndex(2)  # "1 - Crack (TP=green)"
            else:
                self.label_combo.setCurrentIndex(0)  # Unknown

    def _navigate_prev(self):
        """Navigate to previous sample. Auto-runs inference if checkbox is checked."""
        if self.dataset_df is not None and self.current_index > 0:
            self.current_index -= 1
            self.sample_table.selectRow(self.current_index)
            self._update_nav_state()
            self._auto_load_and_run()

    def _navigate_next(self):
        """Navigate to next sample. Auto-runs inference if checkbox is checked."""
        if self.dataset_df is not None and self.current_index < len(self.dataset_df) - 1:
            self.current_index += 1
            self.sample_table.selectRow(self.current_index)
            self._update_nav_state()
            self._auto_load_and_run()

    def _auto_load_and_run(self):
        """Load current sample and auto-run inference if enabled."""
        if not self.auto_run_check.isChecked():
            return
        # Cancel any running inference first
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait(500)
        # Load sample then run inference
        self._load_current_sample()
        # Schedule inference after sample is loaded
        if self.cube is not None and self.run_inference_btn.isEnabled():
            self._run_inference()

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
        """Load sample from CSV dataset (CSV-only mode)."""
        try:
            if self.dataset_df is None:
                show_error(self, "Error", "No CSV loaded. Please load a CSV file first.")
                return
            row = self.dataset_df.iloc[self.current_index]
            folder = Path(row['image_path'])

            if not folder or not folder.exists():
                show_error(self, "Error", f"Folder not found: {folder}")
                return

            self.progress_label.setText("Loading sample...")
            QApplication.processEvents()

            # Load RGB images
            canon_path, canon_searched = find_canon_rgb(folder)
            hsi_path = find_hsi_rgb(folder)
            self.camera_rgb = load_rgb(canon_path) if canon_path else None
            self.hsi_rgb = load_rgb(hsi_path) if hsi_path else None

            # Load HSI - search recursively for .hdr files
            hdr_files = list(folder.glob("*.hdr"))
            if not hdr_files:
                hdr_files = list(folder.glob("HS/results/*.hdr"))
            if not hdr_files:
                hdr_files = list(folder.glob("**/*.hdr"))

            # Filter to REFLECTANCE files if multiple found
            if len(hdr_files) > 1:
                reflectance_files = [f for f in hdr_files if 'REFLECTANCE' in f.name.upper()]
                if reflectance_files:
                    hdr_files = reflectance_files

            # If we found hdr, try to load RGB from same dir/HS/results
            if hdr_files:
                hdr_dir = hdr_files[0].parent
                candidate_rgbs = list(hdr_dir.glob("RGB*.png")) + list(hdr_dir.glob("*RGB*.png"))
                if not candidate_rgbs and hdr_dir.name.lower() != "results" and hdr_dir.parent.name.lower() == "results":
                    candidate_rgbs = list(hdr_dir.parent.glob("RGB*.png"))
                if candidate_rgbs and self.hsi_rgb is None:
                    self.hsi_rgb = load_rgb(candidate_rgbs[0])
                reader = ENVIReader(str(hdr_files[0]))
                self.cube = reader.read()
                self.wavelengths = reader.get_wavelengths()

                if self.cube is not None:
                    # Cube is already in (H, W, C) format from ENVIReader
                    num_bands = self.cube.shape[2]
                    self.band_slider.setMaximum(num_bands - 1)
                    self.band_slider.setValue(num_bands // 2)
                    self._update_hsi_band()

                # Only enable inference if model is also loaded
                if self.model is not None:
                    self.run_inference_btn.setEnabled(True)
                self.progress_label.setText(f"✓ Loaded: {self.cube.shape} (H,W,C) from {hdr_files[0].name}")
            else:
                show_error(self, "Warning", f"No .hdr file found in {folder} or subdirectories")
                self.cube = None
                self.run_inference_btn.setEnabled(False)

            # Display RGBs in first row (after potential hdr rgb load)
            if self.camera_rgb is not None:
                self.viewer_rgb_cam.label.setText("RGB (Camera)")
                self.viewer_rgb_cam.viewer.set_image(self.camera_rgb)
            else:
                self.viewer_rgb_cam.viewer.clear()
                self.viewer_rgb_cam.label.setText("RGB (Camera) - Canon RGB not found")
                if canon_searched:
                    error_logger.info("Canon RGB not found. Searched:\n" + "\n".join(canon_searched))
            if self.hsi_rgb is not None:
                self.viewer_rgb_hsi.viewer.set_image(self.hsi_rgb)
            else:
                self.viewer_rgb_hsi.viewer.clear()
            # Extra slot shows HSI RGB before inference; results overlay after inference
            if self.hsi_rgb is not None:
                self.viewer_rgb_extra.viewer.set_image(self.hsi_rgb)
            elif self.camera_rgb is not None:
                self.viewer_rgb_extra.viewer.set_image(self.camera_rgb)
            else:
                self.viewer_rgb_extra.viewer.clear()

            # Clear previous outputs
            self.prob_map = None
            self.grid_results = None
            for v in [self.viewer_prob_hsi, self.viewer_blob_hsi]:
                v.viewer.clear()
            self.viewer_prob_hsi.label.setText("Model Result on HSI")
            self._disable_postprocess()
            self.stats_text.clear()

            # Auto-run inference after successful load
            if self.cube is not None and self.run_inference_btn.isEnabled():
                QApplication.processEvents()  # Let UI update first
                self._run_inference()

        except Exception as e:
            show_error(self, "Error", f"Failed to load: {e}")
            log_error("Load sample failed", e)
        finally:
            self.progress_label.setText("")

    def _get_current_hsi_gray(self) -> np.ndarray:
        """Return current HSI band (rotated grayscale uint8) for overlay bases."""
        if self.cube is None:
            return np.zeros((256, 256), dtype=np.uint8)
        band_idx = self.band_slider.value()
        band = get_band_by_index(self.cube, band_idx)
        band_norm = normalize_to_uint8(band, method="percentile")
        band_rotated = np.rot90(band_norm, k=-1)
        return band_rotated

    def _update_hsi_band(self):
        """Update HSI band display - shows GRAYSCALE with 90° rotation to match RGB."""
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
        self.viewer_hsi_band.viewer.set_image(band_rotated)
        self._last_band_rotated = band_rotated

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
        # stop previous worker and clear outputs
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait(2000)
        self.prob_map = None
        self.viewer_prob_hsi.viewer.clear()
        self.viewer_blob_hsi.viewer.clear()
        self._disable_postprocess()

        if self.model is None:
            show_error(self, "Error", "No model loaded. Please load a model first.")
            return
        try:

            from app.config.types import PreprocessConfig
            preprocess_cfg = PreprocessConfig(
                use_snv=True,
                wavelengths=self.wavelengths,
                wl_min=settings.wl_min if self.wavelengths is not None else None,
                wl_max=settings.wl_max if self.wavelengths is not None else None
            )
            
            # Share model with Optuna tab via parent window
            self._share_model_with_optuna_tab(self.model, preprocess_cfg)

            target_idx = int(self.target_class_combo.currentData())
            if target_idx >= self.model.n_classes:
                show_error(self, "Error", f"Target class index out of range (n_classes={self.model.n_classes})")
                self.run_inference_btn.setEnabled(True)
                return
            self.inference_worker = InferenceWorker(self.cube, self.model, preprocess_cfg, target_idx)
            self.inference_worker.finished.connect(lambda pm, t=target_idx: self._on_inference_done(pm, t))
            self.inference_worker.error.connect(self._on_inference_error)
            self.inference_worker.progress.connect(lambda msg: self.progress_label.setText(msg))
            self.run_inference_btn.setEnabled(False)
            self.inference_worker.start()
        except Exception as e:
            show_error(self, "Error", f"Inference failed: {e}")
            log_error("Inference failed", e)

    def _update_prob_visualization(self):
        """Update probability heat map visualization based on current threshold."""
        if self.prob_map is None:
            return
        
        # Rotate prob map to match HSI orientation
        prob_rotated = np.rot90(self.prob_map, k=-1)
        base_gray = self._get_current_hsi_gray()
        base_rgb = np.stack([base_gray] * 3, axis=-1)

        # Use threshold from UI to filter probability display
        threshold = self.thresh_spin.value()
        above_threshold_mask = prob_rotated > threshold
        
        # Create colored heat map showing probability gradients
        prob_vis = normalize_to_uint8(prob_rotated, method="percentile")
        prob_colored = apply_colormap(prob_vis / 255.0, name="hot")
        
        # Blend colored heat map only where prob > threshold, otherwise show base gray
        blended = base_rgb.copy()
        blended[above_threshold_mask] = (0.5 * prob_colored[above_threshold_mask] + 0.5 * base_rgb[above_threshold_mask]).astype(np.uint8)
        
        self.viewer_prob_hsi.viewer.set_image(blended)

    def _on_inference_done(self, prob_map, target_idx=None):
        self.prob_map = prob_map
        self.run_inference_btn.setEnabled(True)
        self.progress_label.setText("✓ Inference complete")

        if target_idx is not None:
            self.viewer_prob_hsi.label.setText(f"Model Result on HSI (Class {target_idx})")
        
        self._update_prob_visualization()
        self._enable_postprocess()
        self._rerun_postprocess()

    def _on_inference_error(self, msg):
        self.run_inference_btn.setEnabled(True)
        self.progress_label.setText("✗ Inference failed")
        show_error(self, "Inference Error", msg)

    def _disable_postprocess(self):
        for w in [self.thresh_spin, self.morph_spin, self.min_area_spin,
                  self.exclude_border_check, self.border_margin_spin,
                  self.patch_size_combo, self.patch_thresh_spin]:
            w.setEnabled(False)
        self.post_status.setText("⚠ Run inference first")
        self.post_status.setStyleSheet("color: orange;")

    def _enable_postprocess(self):
        for w in [self.thresh_spin, self.morph_spin, self.min_area_spin,
                  self.exclude_border_check, self.border_margin_spin,
                  self.patch_size_combo, self.patch_thresh_spin]:
            w.setEnabled(True)
        self.post_status.setText("✓ Controls active")
        self.post_status.setStyleSheet("color: green;")

    def _on_threshold_changed(self):
        """Update both probability visualization and postprocess when threshold changes."""
        self._update_prob_visualization()
        self._rerun_postprocess()

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

            final_mask_rotated = np.rot90(final_mask, k=-1)
            threshold_mask = debug.get('mask_threshold', None)
            threshold_mask_rotated = np.rot90(threshold_mask, k=-1) if threshold_mask is not None else None

            base_gray = self._get_current_hsi_gray()
            base_rgb = np.stack([base_gray] * 3, axis=-1)

            label_idx = self.label_combo.currentIndex()
            if label_idx == 1:
                overlay_color = (255, 0, 0)
            elif label_idx == 2:
                overlay_color = (0, 255, 0)
            else:
                overlay_color = (0, 255, 255)

            self.viewer_blob_hsi.viewer.set_image(base_rgb)
            self.viewer_blob_hsi.viewer.set_overlay(final_mask_rotated, alpha=0.6, color=overlay_color)
            self._update_rgb_extra_results(final_mask_rotated, threshold_mask_rotated)
            # Stats
            accepted_count = len(stats.get('accepted_blobs', []))
            rejected_count = len(stats.get('rejected_blobs', []))
            self.stats_text.setText(
                f"Blobs Before: {stats['num_blobs_before']}\n"
                f"Blobs After: {stats['num_blobs_after']}\n"
                f"Accepted: {accepted_count} | Rejected: {rejected_count}\n"
                f"Positive Pixels: {stats['total_positive_pixels']}\n"
                f"Crack Ratio: {stats['crack_ratio']:.4f}"
            )
            
            # Run patch analysis
            self._run_patch_analysis(final_mask_rotated, threshold_mask_rotated)
            
        except Exception as e:
            log_error("Postprocess failed", e)

    def _apply_mask_overlay(self, base_rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
        if mask is None:
            return base_rgb
        if mask.dtype != bool:
            mask = mask.astype(bool)
        blended = base_rgb.copy()
        overlay = np.zeros_like(base_rgb, dtype=np.uint8)
        overlay[:, :] = color
        blended[mask] = (alpha * overlay[mask] + (1 - alpha) * base_rgb[mask]).astype(np.uint8)
        return blended

    def _resize_rgb_to_mask(self, rgb: np.ndarray, mask_shape: tuple) -> np.ndarray:
        if rgb.shape[0] == mask_shape[0] and rgb.shape[1] == mask_shape[1]:
            return rgb
        from PIL import Image
        resized = Image.fromarray(rgb).resize((mask_shape[1], mask_shape[0]), resample=Image.BILINEAR)
        return np.array(resized, dtype=np.uint8)

    def _update_rgb_extra_results(self, final_mask: np.ndarray, threshold_mask: Optional[np.ndarray]) -> None:
        # Use HSI RGB for results overlay (not Canon RGB)
        base = None
        if self.hsi_rgb is not None:
            base = self.hsi_rgb
        elif self.camera_rgb is not None:
            base = self.camera_rgb
        if base is None:
            base = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
        base = self._resize_rgb_to_mask(base, final_mask.shape)
        result = base.copy()
        if threshold_mask is not None:
            result = self._apply_mask_overlay(result, threshold_mask, color=(255, 255, 0), alpha=0.35)
        result = self._apply_mask_overlay(result, final_mask, color=(0, 255, 0), alpha=0.6)
        self.viewer_rgb_extra.viewer.set_image(result)

    def _run_patch_analysis(self, final_mask_rotated: np.ndarray, threshold_mask_rotated: Optional[np.ndarray] = None):
        """Run patch analysis on the blob results (final mask after postprocessing)."""
        try:
            # Use final blob mask (results after all postprocessing)
            mask = final_mask_rotated
            if mask is None:
                return
            
            # Get parameters from UI
            patch_size = int(self.patch_size_combo.currentText())
            patch_threshold_pct = self.patch_thresh_spin.value()
            
            H, W = mask.shape
            patch_flag_mask = np.zeros((H, W), dtype=bool)
            
            total_patches = 0
            flagged_patches = 0
            max_pct = 0.0
            flagged_pcts = []
            
            # Loop over patches
            for y in range(0, H, patch_size):
                for x in range(0, W, patch_size):
                    y2 = min(y + patch_size, H)
                    x2 = min(x + patch_size, W)
                    
                    patch = mask[y:y2, x:x2]
                    total_patches += 1
                    
                    # Compute crack percentage in this patch
                    ratio = np.mean(patch.astype(float))
                    pct = ratio * 100.0
                    
                    if pct > max_pct:
                        max_pct = pct
                    
                    # Flag patch if above threshold
                    if pct >= patch_threshold_pct:
                        patch_flag_mask[y:y2, x:x2] = True
                        flagged_patches += 1
                        flagged_pcts.append(pct)
            
            # Store results
            self.patch_flag_mask = patch_flag_mask
            avg_flagged_pct = np.mean(flagged_pcts) if flagged_pcts else 0.0
            
            self.patch_stats = {
                'total_patches': total_patches,
                'flagged_patches': flagged_patches,
                'max_pct': max_pct,
                'avg_flagged_pct': avg_flagged_pct,
                'patch_size': patch_size,
                'threshold_pct': patch_threshold_pct
            }
            
            # Update visualization
            self._update_patch_visualization(final_mask_rotated)
            
        except Exception as e:
            log_error("Patch analysis failed", e)

    def _update_patch_visualization(self, final_mask_rotated: np.ndarray):
        """Update patch visualization in both RGB and HSI viewers."""
        if self.patch_flag_mask is None:
            return
        
        try:
            # Patch Map on HSI
            base_gray = self._get_current_hsi_gray()
            base_rgb_hsi = np.stack([base_gray] * 3, axis=-1)
            result_hsi = self._apply_mask_overlay(base_rgb_hsi, self.patch_flag_mask, color=(0, 255, 255), alpha=0.5)
            self.viewer_hsi_patch.viewer.set_image(result_hsi)
            
            # Patch Map on RGB
            base = None
            if self.hsi_rgb is not None:
                base = self.hsi_rgb
            elif self.camera_rgb is not None:
                base = self.camera_rgb
            if base is None:
                base = np.zeros((*final_mask_rotated.shape, 3), dtype=np.uint8)
            if base.ndim == 2:
                base = np.stack([base] * 3, axis=-1)
            base = self._resize_rgb_to_mask(base, final_mask_rotated.shape)
            result_rgb = self._apply_mask_overlay(base, self.patch_flag_mask, color=(0, 255, 255), alpha=0.5)
            self.viewer_rgb_patch.viewer.set_image(result_rgb)
            
        except Exception as e:
            log_error("Patch visualization failed", e)

    def _rerun_patch_analysis(self):
        """Re-run patch analysis when controls change (no arguments needed)."""
        if self.prob_map is None:
            return
        # Re-run postprocess to get fresh masks, which will trigger patch analysis
        self._rerun_postprocess()

    def reset_state(self):
        """Clear all loaded data and reset UI."""
        self.stop_workers()
        self.cube = None
        self.wavelengths = None
        self.hsi_rgb = None
        self.camera_rgb = None
        self.prob_map = None
        self.patch_flag_mask = None
        self.patch_stats = None
        self.model = None

        for v in [self.viewer_rgb_cam, self.viewer_rgb_hsi, self.viewer_rgb_extra, self.viewer_rgb_patch,
                  self.viewer_hsi_band, self.viewer_prob_hsi, self.viewer_blob_hsi, self.viewer_hsi_patch]:
            v.viewer.clear()
        self.viewer_rgb_cam.label.setText("RGB (Camera)")

        self.stats_text.clear()
        self._disable_postprocess()
        self.progress_label.setText("")
        self.band_slider.setValue(0)
        self.band_slider.setMaximum(0)
        self.label_combo.setCurrentIndex(0)  # Reset to Unknown
        self._set_target_class_options(2)
        self.viewer_prob_hsi.label.setText("Model Result on HSI")
        gc.collect()
    
    def _share_model_with_optuna_tab(self, model, preprocess_cfg):
        """Share loaded model with Optuna tab via parent window."""
        try:
            # Navigate to parent MainWindow
            parent = self.parent()
            while parent is not None:
                if isinstance(parent, QMainWindow) and hasattr(parent, 'optuna_tab'):
                    parent.optuna_tab.set_model_and_config(model, preprocess_cfg)
                    error_logger.info("Model shared with Optuna tab")
                    break
                parent = parent.parent()
        except Exception as e:
            error_logger.warning(f"Could not share model with Optuna tab: {e}")

    def stop_workers(self):
        """Stop any running workers."""
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait(2000)


# ============================================================================
# Main Window
# ============================================================================

class MainWindow(QMainWindow):
    """Main window with tabs for visual debug and Optuna tuning."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Classification Inference UI")
        self.setMinimumSize(1000, 650)
        
        # Create shared model manager
        self.model_manager = ModelManager()
        
        self._init_ui()
        # Show maximized by default for best image viewing experience
        self.showMaximized()

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

        self.visual_tab = VisualDebugTab(model_manager=self.model_manager)
        self.tabs.addTab(self.visual_tab, "Visual Debug")
        
        # Add Optuna hyperparameter tuning tab with shared model manager
        self.optuna_tab = OptunaTabWidget(model_manager=self.model_manager)
        self.tabs.addTab(self.optuna_tab, "Optuna Tuning")

        layout.addWidget(self.tabs)
        self.statusBar().showMessage("Ready")

    def _reset_all(self):
        """Reset all state in both tabs."""
        reply = QMessageBox.question(self, "Confirm Reset",
                                      "Clear all loaded data and reset UI?",
                                      QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.visual_tab.reset_state()
            self.optuna_tab.reset_state()
            self.statusBar().showMessage("UI state cleared")

    def closeEvent(self, event):
        """Clean shutdown."""
        self.visual_tab.stop_workers()
        self.optuna_tab.stop_workers()
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

class MainWindow(QMainWindow):
    """Main window with tabs for visual debug and dataset tuning."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Classification Inference UI")
        self.setMinimumSize(1000, 650)
        
        # Create shared model manager
        self.model_manager = ModelManager()
        
        self._init_ui()
        # Show maximized by default for best image viewing experience
        self.showMaximized()

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

        self.visual_tab = VisualDebugTab(model_manager=self.model_manager)
        self.tabs.addTab(self.visual_tab, "Visual Debug")
        
        # Add Optuna hyperparameter tuning tab with shared model manager
        self.optuna_tab = OptunaTabWidget(model_manager=self.model_manager)
        self.tabs.addTab(self.optuna_tab, "Optuna Tuning")

        layout.addWidget(self.tabs)
        self.statusBar().showMessage("Ready")

    def _reset_all(self):
        """Reset all state in both tabs."""
        reply = QMessageBox.question(self, "Confirm Reset",
                                      "Clear all loaded data and reset UI?",
                                      QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.visual_tab.reset_state()
            self.optuna_tab.reset_state()
            self.statusBar().showMessage("UI state cleared")

    def closeEvent(self, event):
        """Clean shutdown."""
        self.visual_tab.stop_workers()
        self.optuna_tab.stop_workers()
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
