"""
Binary Classification Inference UI - Enhanced Debug Mode

Main entrypoint for the inference application with comprehensive debugging.

Features:
- Folder mode: Debug single sample from folder
- Dataset mode: Browse train/val CSV with navigation
- Dual RGB display (HSI-derived + camera)
- Interactive HSI band viewer
- 2x3 grid of debug stages (RGB, HSI, ProbMap, Threshold, Morph, Final)
- Live postprocessing controls
- Worker threads for responsive UI

All non-UI logic is delegated to app/ modules. UI orchestrates only.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Check for PyQt5
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QFileDialog, QComboBox,
        QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QGroupBox,
        QFormLayout, QSplitter, QFrame, QMessageBox, QProgressBar,
        QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
        QSlider, QRadioButton, QButtonGroup, QCheckBox, QGridLayout
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    print("ERROR: PyQt5 is required. Install with: pip install PyQt5")
    sys.exit(1)

# Import backend modules
from app.config.settings import settings
from app.utils.logging import logger, setup_logger
from app.ui import ImageViewer
from app.io import (
    ENVIReader, find_both_rgb_images, load_rgb,
    get_band_by_index, get_band_by_wavelength
)
from app.data.dataset import load_dataset_csv
from app.postprocess import PostprocessPipeline, PostprocessConfig
from app.utils import normalize_to_uint8, apply_colormap

# Setup file logger
error_logger = setup_logger("error_log", str(Path(__file__).parent / "logs"))


def log_error(msg: str, exc: Optional[Exception] = None):
    """Log error to file with traceback."""
    if exc:
        error_logger.error(f"{msg}\n{traceback.format_exc()}")
    else:
        error_logger.error(msg)


def show_error(parent, title: str, message: str):
    """Show error dialog."""
    QMessageBox.critical(parent, title, message)


def show_info(parent, title: str, message: str):
    """Show info dialog."""
    QMessageBox.information(parent, title, message)


# ============================================================================
# Worker Threads
# ============================================================================

class InferenceWorker(QThread):
    """Worker thread for running inference."""

    finished = pyqtSignal(object)  # prob_map
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cube: np.ndarray, model, preprocess_cfg, parent=None):
        super().__init__(parent)
        self.cube = cube
        self.model = model
        self.preprocess_cfg = preprocess_cfg

    def run(self):
        try:
            from app.inference.prob_map import build_prob_map

            self.progress.emit("Running inference...")
            prob_map = build_prob_map(
                self.cube,
                self.model,
                self.preprocess_cfg,
                target_class_index=1,
                chunk_size=100_000
            )
            self.finished.emit(prob_map)
        except Exception as e:
            log_error("Inference failed", e)
            self.error.emit(str(e))


    finished = pyqtSignal(dict)  # best_params
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, train_df, val_df, prob_map_fn, n_trials, seed, output_dir, metric, parent=None):
        super().__init__(parent)
        self.train_df = train_df
        self.val_df = val_df
        self.prob_map_fn = prob_map_fn
        self.n_trials = n_trials
        self.seed = seed
        self.output_dir = output_dir
        self.metric = metric

    def run(self):
        try:
            from app.tuning.optuna_runner import run_optuna

            self.progress.emit("Running Optuna tuning (this may take a while)...")
            best_params, trials_df = run_optuna(
                train_df=self.train_df,
                val_df=self.val_df,
                prob_map_fn=self.prob_map_fn,
                n_trials=self.n_trials,
                seed=self.seed,
                output_dir=self.output_dir,
                metric=self.metric,
            )
            self.finished.emit({"best_params": best_params, "n_trials": len(trials_df)})
        except Exception as e:
            log_error("Optuna tuning failed", e)
            self.error.emit(str(e))


class FinalEvalWorker(QThread):
    """Worker thread for final test evaluation."""

    finished = pyqtSignal(dict)  # metrics
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, test_df, prob_map_fn, best_params, output_dir, train_df, val_df, parent=None):
        super().__init__(parent)
        self.test_df = test_df
        self.prob_map_fn = prob_map_fn
        self.best_params = best_params
        self.output_dir = output_dir
        self.train_df = train_df
        self.val_df = val_df

    def run(self):
        try:
            from app.tuning.optuna_runner import evaluate_final

            self.progress.emit("Running final evaluation on test set...")
            metrics, per_sample_df = evaluate_final(
                test_df=self.test_df,
                prob_map_fn=self.prob_map_fn,
                best_params=self.best_params,
                output_dir=self.output_dir,
                train_df=self.train_df,
                val_df=self.val_df,
            )
            self.finished.emit(metrics)
        except Exception as e:
            log_error("Final evaluation failed", e)
            self.error.emit(str(e))


# ============================================================================
# Tab A: Single Sample Inference
# ============================================================================

class SingleSampleTab(QWidget):
    """Tab for single sample inference."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cube = None
        self.wavelengths = None
        self.model = None
        self.model_adapter = None
        self.prob_map = None
        self.current_mask = None
        self.worker = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === Data Selection Group ===
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)

        # Folder selection
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder containing HSI data...")
        self.folder_edit.setReadOnly(True)
        folder_btn = QPushButton("Browse...")
        folder_btn.clicked.connect(self._select_folder)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(folder_btn)
        data_layout.addRow("HSI Folder:", folder_row)

        # Model selection
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        self._refresh_models()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(refresh_btn)
        data_layout.addRow("Model:", model_row)

        layout.addWidget(data_group)

        # === Inference Controls Group ===
        inference_group = QGroupBox("Inference")
        inference_layout = QVBoxLayout(inference_group)

        # Options row
        options_layout = QHBoxLayout()

        self.snv_check = QPushButton("✓ SNV")
        self.snv_check.setCheckable(True)
        self.snv_check.setChecked(True)
        options_layout.addWidget(self.snv_check)

        options_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        options_layout.addWidget(self.threshold_spin)

        options_layout.addStretch()

        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self._run_inference)
        self.run_btn.setEnabled(False)
        options_layout.addWidget(self.run_btn)

        inference_layout.addLayout(options_layout)

        # Progress
        self.progress_label = QLabel("")
        inference_layout.addWidget(self.progress_label)

        layout.addWidget(inference_group)

        # === Results Display ===
        results_group = QGroupBox("Results")
        results_layout = QHBoxLayout(results_group)

        # Prob map display
        prob_frame = QFrame()
        prob_frame.setFrameStyle(QFrame.StyledPanel)
        prob_layout = QVBoxLayout(prob_frame)
        prob_layout.addWidget(QLabel("Probability Map"))
        self.prob_label = QLabel()
        self.prob_label.setMinimumSize(300, 300)
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.prob_label.setStyleSheet("background-color: #f0f0f0;")
        prob_layout.addWidget(self.prob_label)
        results_layout.addWidget(prob_frame)

        # Mask display
        mask_frame = QFrame()
        mask_frame.setFrameStyle(QFrame.StyledPanel)
        mask_layout = QVBoxLayout(mask_frame)
        mask_layout.addWidget(QLabel("Thresholded Mask"))
        self.mask_label = QLabel()
        self.mask_label.setMinimumSize(300, 300)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet("background-color: #f0f0f0;")
        mask_layout.addWidget(self.mask_label)
        results_layout.addWidget(mask_frame)

        layout.addWidget(results_group)

        # === Stats Display ===
        self.stats_label = QLabel("Load data and run inference to see results.")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def _refresh_models(self):
        """Refresh model list from settings directory."""
        self.model_combo.clear()

        models_dir = settings.models_dir
        if models_dir.exists():
            for ext in ['.joblib', '.pkl', '.pth']:
                for model_file in models_dir.glob(f'*{ext}'):
                    self.model_combo.addItem(model_file.name, str(model_file))

        if self.model_combo.count() == 0:
            self.model_combo.addItem("(No models found)", None)

    def _select_folder(self):
        """Select folder containing HSI data."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select HSI Folder",
            str(settings.default_search_folder)
        )
        if folder:
            self.folder_edit.setText(folder)
            self._load_hsi_from_folder(folder)

    def _load_hsi_from_folder(self, folder: str):
        """Load HSI data from folder."""
        try:
            from app.io.envi import ENVIReader

            folder_path = Path(folder)

            # Find .hdr file
            hdr_files = list(folder_path.glob("*.hdr"))
            if not hdr_files:
                show_error(self, "Error", f"No .hdr file found in {folder}")
                return

            hdr_path = hdr_files[0]

            self.progress_label.setText(f"Loading {hdr_path.name}...")
            QApplication.processEvents()

            reader = ENVIReader(str(hdr_path))
            self.cube = reader.read()
            self.wavelengths = reader.get_wavelengths()

            self.progress_label.setText(
                f"Loaded: {self.cube.shape[0]}x{self.cube.shape[1]} pixels, "
                f"{self.cube.shape[2]} bands"
            )

            self._check_ready()

        except Exception as e:
            log_error("Failed to load HSI", e)
            show_error(self, "Load Error", f"Failed to load HSI data:\n{str(e)}")

    def _check_ready(self):
        """Check if ready to run inference."""
        model_path = self.model_combo.currentData()
        ready = self.cube is not None and model_path is not None
        self.run_btn.setEnabled(ready)

    def _run_inference(self):
        """Run inference on loaded cube."""
        try:
            model_path = self.model_combo.currentData()
            if not model_path:
                show_error(self, "Error", "Please select a model")
                return

            # Load model if needed
            if self.model is None or self.model_adapter is None:
                self._load_model(model_path)

            # Create preprocess config
            from app.config.types import PreprocessConfig

            preprocess_cfg = PreprocessConfig(
                use_snv=self.snv_check.isChecked(),
                wavelengths=self.wavelengths,
                wl_min=settings.wl_min if self.wavelengths is not None else None,
                wl_max=settings.wl_max if self.wavelengths is not None else None,
            )

            # Run inference in worker thread
            self.run_btn.setEnabled(False)
            self.progress_label.setText("Running inference...")

            self.worker = InferenceWorker(self.cube, self.model_adapter, preprocess_cfg)
            self.worker.finished.connect(self._on_inference_finished)
            self.worker.error.connect(self._on_inference_error)
            self.worker.progress.connect(lambda msg: self.progress_label.setText(msg))
            self.worker.start()

        except Exception as e:
            log_error("Inference setup failed", e)
            show_error(self, "Error", f"Failed to start inference:\n{str(e)}")
            self.run_btn.setEnabled(True)

    def _load_model(self, model_path: str):
        """Load model and create adapter."""
        from app.models.loader_new import load_model
        from app.models.adapters_new import SklearnAdapter

        self.model = load_model(model_path)
        self.model_adapter = SklearnAdapter(self.model, name=Path(model_path).stem)

    def _on_inference_finished(self, prob_map: np.ndarray):
        """Handle inference completion."""
        self.prob_map = prob_map
        self.run_btn.setEnabled(True)
        self.progress_label.setText("Inference complete!")

        self._update_displays()

    def _on_inference_error(self, error_msg: str):
        """Handle inference error."""
        self.run_btn.setEnabled(True)
        self.progress_label.setText("Inference failed")
        show_error(self, "Inference Error", error_msg)

    def _on_threshold_changed(self):
        """Handle threshold change - rerun postprocess only."""
        if self.prob_map is not None:
            self._update_displays()

    def _update_displays(self):
        """Update probability map and mask displays."""
        if self.prob_map is None:
            return

        threshold = self.threshold_spin.value()

        # Apply postprocessing
        from app.postprocess.pipeline import PostprocessConfig, PostprocessPipeline

        config = PostprocessConfig(prob_threshold=threshold)
        pipeline = PostprocessPipeline(config)

        self.current_mask, stats = pipeline.run(self.prob_map)

        # Update stats
        self.stats_label.setText(
            f"Positive pixels: {stats['total_positive_pixels']:,} | "
            f"Crack ratio: {stats['crack_ratio']:.4f} | "
            f"Blobs: {stats['num_blobs_after']}"
        )

        # Display probability map
        self._display_image(self.prob_label, self.prob_map, cmap='viridis')

        # Display mask
        self._display_image(self.mask_label, self.current_mask.astype(np.uint8) * 255, cmap='gray')

    def _display_image(self, label: QLabel, image: np.ndarray, cmap: str = 'gray'):
        """Display numpy array in QLabel."""
        # Normalize to 0-255
        if image.dtype == bool:
            img_uint8 = image.astype(np.uint8) * 255
        elif image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        # Apply colormap for probability map
        if cmap == 'viridis' and img_uint8.ndim == 2:
            # Simple viridis-like mapping
            img_rgb = np.zeros((*img_uint8.shape, 3), dtype=np.uint8)
            img_rgb[:, :, 0] = (255 - img_uint8)  # R
            img_rgb[:, :, 1] = img_uint8  # G
            img_rgb[:, :, 2] = (img_uint8 * 0.5).astype(np.uint8)  # B
        elif img_uint8.ndim == 2:
            img_rgb = np.stack([img_uint8] * 3, axis=-1)
        else:
            img_rgb = img_uint8

        h, w = img_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit label
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)


# ============================================================================
# Tab B: Dataset Tuning and Evaluation
# ============================================================================

class DatasetTab(QWidget):
    """Tab for dataset tuning and evaluation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.best_params = None
        self.model = None
        self.model_adapter = None
        self.output_dir = None
        self.worker = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === Data Selection Group ===
        data_group = QGroupBox("Dataset Selection")
        data_layout = QFormLayout(data_group)

        # Train/Val CSV
        trainval_row = QHBoxLayout()
        self.trainval_edit = QLineEdit()
        self.trainval_edit.setPlaceholderText("Select train/val CSV...")
        self.trainval_edit.setReadOnly(True)
        trainval_btn = QPushButton("Browse...")
        trainval_btn.clicked.connect(lambda: self._select_csv("trainval"))
        trainval_row.addWidget(self.trainval_edit)
        trainval_row.addWidget(trainval_btn)
        data_layout.addRow("Train/Val CSV:", trainval_row)

        # Test CSV
        test_row = QHBoxLayout()
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("Select test CSV...")
        self.test_edit.setReadOnly(True)
        test_btn = QPushButton("Browse...")
        test_btn.clicked.connect(lambda: self._select_csv("test"))
        test_row.addWidget(self.test_edit)
        test_row.addWidget(test_btn)
        data_layout.addRow("Test CSV:", test_row)

        # Model selection
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        self._refresh_models()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(refresh_btn)
        data_layout.addRow("Model:", model_row)

        layout.addWidget(data_group)

        # === Configuration Group ===
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.30)
        config_layout.addRow("Val Split Size:", self.val_split_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        config_layout.addRow("Random Seed:", self.seed_spin)

        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(5, 500)
        self.trials_spin.setValue(50)
        config_layout.addRow("N Trials:", self.trials_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["f2", "f1", "accuracy", "macro_f1"])
        config_layout.addRow("Metric:", self.metric_combo)

        layout.addWidget(config_group)

        # === Actions Group ===
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.prepare_btn = QPushButton("1. Prepare Splits")
        self.prepare_btn.clicked.connect(self._prepare_splits)
        actions_layout.addWidget(self.prepare_btn)

        self.optuna_btn = QPushButton("2. Run Optuna")
        self.optuna_btn.clicked.connect(self._run_optuna)
        self.optuna_btn.setEnabled(False)
        actions_layout.addWidget(self.optuna_btn)

        self.final_btn = QPushButton("3. Run Final Test")
        self.final_btn.clicked.connect(self._run_final_test)
        self.final_btn.setEnabled(False)
        actions_layout.addWidget(self.final_btn)

        layout.addWidget(actions_group)

        # === Progress ===
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # === Results Display ===
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        # Splits table
        self.splits_table = QTableWidget()
        self.splits_table.setColumnCount(4)
        self.splits_table.setHorizontalHeaderLabels(["Split", "Total", "Class 0", "Class 1"])
        self.splits_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.splits_table.setMaximumHeight(120)
        results_layout.addWidget(self.splits_table)

        # Results text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        layout.addStretch()

    def _refresh_models(self):
        """Refresh model list from settings directory."""
        self.model_combo.clear()

        models_dir = settings.models_dir
        if models_dir.exists():
            for ext in ['.joblib', '.pkl', '.pth']:
                for model_file in models_dir.glob(f'*{ext}'):
                    self.model_combo.addItem(model_file.name, str(model_file))

        if self.model_combo.count() == 0:
            self.model_combo.addItem("(No models found)", None)

    def _select_csv(self, csv_type: str):
        """Select CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {csv_type} CSV",
            str(settings.default_search_folder),
            "CSV Files (*.csv)"
        )
        if path:
            if csv_type == "trainval":
                self.trainval_edit.setText(path)
            else:
                self.test_edit.setText(path)

    def _prepare_splits(self):
        """Prepare train/val/test splits."""
        try:
            trainval_path = self.trainval_edit.text()
            test_path = self.test_edit.text()

            if not trainval_path or not test_path:
                show_error(self, "Error", "Please select both CSV files")
                return

            from app.data.dataset import load_and_prepare_splits, get_class_distribution

            self.progress_label.setText("Loading and splitting datasets...")
            QApplication.processEvents()

            self.train_df, self.val_df, self.test_df = load_and_prepare_splits(
                trainval_path,
                test_path,
                val_size=self.val_split_spin.value(),
                random_state=self.seed_spin.value()
            )

            # Update splits table
            self.splits_table.setRowCount(3)

            for i, (name, df) in enumerate([
                ("Train", self.train_df),
                ("Val", self.val_df),
                ("Test", self.test_df)
            ]):
                dist = get_class_distribution(df)
                self.splits_table.setItem(i, 0, QTableWidgetItem(name))
                self.splits_table.setItem(i, 1, QTableWidgetItem(str(len(df))))
                self.splits_table.setItem(i, 2, QTableWidgetItem(str(dist.get(0, 0))))
                self.splits_table.setItem(i, 3, QTableWidgetItem(str(dist.get(1, 0))))

            self.progress_label.setText("Splits prepared successfully!")
            self.optuna_btn.setEnabled(True)

            self.results_text.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Splits prepared:\n"
                f"  Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}\n"
            )

        except Exception as e:
            log_error("Failed to prepare splits", e)
            show_error(self, "Error", f"Failed to prepare splits:\n{str(e)}")

    def _load_model(self):
        """Load model if not already loaded."""
        model_path = self.model_combo.currentData()
        if not model_path:
            raise ValueError("Please select a model")

        if self.model is None:
            from app.models.loader_new import load_model
            from app.models.adapters_new import SklearnAdapter

            self.model = load_model(model_path)
            self.model_adapter = SklearnAdapter(self.model, name=Path(model_path).stem)

    def _create_prob_map_fn(self) -> Callable[[str], np.ndarray]:
        """Create probability map function for tuning."""
        from app.io.envi import ENVIReader
        from app.inference.prob_map import build_prob_map
        from app.config.types import PreprocessConfig

        model_adapter = self.model_adapter

        def prob_map_fn(image_path: str) -> np.ndarray:
            # Find .hdr file in the folder
            folder = Path(image_path)
            if folder.is_file():
                folder = folder.parent

            hdr_files = list(folder.glob("*.hdr"))
            if not hdr_files:
                raise FileNotFoundError(f"No .hdr file in {folder}")

            reader = ENVIReader(str(hdr_files[0]))
            cube = reader.read()
            wavelengths = reader.get_wavelengths()

            preprocess_cfg = PreprocessConfig(
                use_snv=True,
                wavelengths=wavelengths,
                wl_min=settings.wl_min if wavelengths is not None else None,
                wl_max=settings.wl_max if wavelengths is not None else None,
            )

            return build_prob_map(
                cube, model_adapter, preprocess_cfg,
                target_class_index=1, chunk_size=100_000
            )

        return prob_map_fn

    def _run_optuna(self):
        """Run Optuna hyperparameter tuning."""
        try:
            if self.val_df is None:
                show_error(self, "Error", "Please prepare splits first")
                return

            self._load_model()

            # Create output directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = str(settings.results_dir / f"tuning_{timestamp}")

            prob_map_fn = self._create_prob_map_fn()

            self.optuna_btn.setEnabled(False)
            self.progress_label.setText("Running Optuna (this may take a while)...")

            self.worker = OptunaWorker(
                self.train_df, self.val_df, prob_map_fn,
                self.trials_spin.value(), self.seed_spin.value(),
                self.output_dir, self.metric_combo.currentText()
            )
            self.worker.finished.connect(self._on_optuna_finished)
            self.worker.error.connect(self._on_optuna_error)
            self.worker.progress.connect(lambda msg: self.progress_label.setText(msg))
            self.worker.start()

        except Exception as e:
            log_error("Optuna setup failed", e)
            show_error(self, "Error", f"Failed to start Optuna:\n{str(e)}")
            self.optuna_btn.setEnabled(True)

    def _on_optuna_finished(self, result: dict):
        """Handle Optuna completion."""
        self.best_params = result["best_params"]
        self.optuna_btn.setEnabled(True)
        self.final_btn.setEnabled(True)
        self.progress_label.setText("Optuna complete!")

        self.results_text.append(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Optuna Results:\n"
            f"  Trials: {result['n_trials']}\n"
            f"  Best params: {self.best_params}\n"
            f"  Output: {self.output_dir}\n"
        )

    def _on_optuna_error(self, error_msg: str):
        """Handle Optuna error."""
        self.optuna_btn.setEnabled(True)
        self.progress_label.setText("Optuna failed")
        show_error(self, "Optuna Error", error_msg)

    def _run_final_test(self):
        """Run final evaluation on test set."""
        try:
            if self.best_params is None:
                show_error(self, "Error", "Please run Optuna first")
                return

            prob_map_fn = self._create_prob_map_fn()

            self.final_btn.setEnabled(False)
            self.progress_label.setText("Running final evaluation...")

            self.worker = FinalEvalWorker(
                self.test_df, prob_map_fn, self.best_params,
                self.output_dir, self.train_df, self.val_df
            )
            self.worker.finished.connect(self._on_final_finished)
            self.worker.error.connect(self._on_final_error)
            self.worker.progress.connect(lambda msg: self.progress_label.setText(msg))
            self.worker.start()

        except Exception as e:
            log_error("Final eval setup failed", e)
            show_error(self, "Error", f"Failed to start evaluation:\n{str(e)}")
            self.final_btn.setEnabled(True)

    def _on_final_finished(self, metrics: dict):
        """Handle final evaluation completion."""
        self.final_btn.setEnabled(True)
        self.progress_label.setText("Final evaluation complete!")

        self.results_text.append(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] FINAL TEST RESULTS:\n"
            f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n"
            f"  Precision: {metrics.get('precision', 0):.4f}\n"
            f"  Recall:    {metrics.get('recall', 0):.4f}\n"
            f"  F1:        {metrics.get('f1', 0):.4f}\n"
            f"  F2:        {metrics.get('f2', 0):.4f}\n"
            f"  Samples:   {metrics.get('n_samples', 0)}\n"
            f"\nResults saved to: {self.output_dir}\n"
        )

        show_info(self, "Complete",
            f"Final evaluation complete!\n\n"
            f"F2 Score: {metrics.get('f2', 0):.4f}\n\n"
            f"Results saved to:\n{self.output_dir}"
        )

    def _on_final_error(self, error_msg: str):
        """Handle final evaluation error."""
        self.final_btn.setEnabled(True)
        self.progress_label.setText("Final evaluation failed")
        show_error(self, "Evaluation Error", error_msg)


# ============================================================================
# Main Window
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window with tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Classification Inference UI")
        self.setMinimumSize(900, 700)

        self._init_ui()

    def _init_ui(self):
        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Header
        header = QLabel("Binary Classification Inference")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()

        # Tab A: Single Sample
        self.single_tab = SingleSampleTab()
        self.tabs.addTab(self.single_tab, "Single Sample")

        # Tab B: Dataset
        self.dataset_tab = DatasetTab()
        self.tabs.addTab(self.dataset_tab, "Dataset Tuning")

        layout.addWidget(self.tabs)

        # Status bar
        self.statusBar().showMessage("Ready")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entrypoint."""
    logger.info("=" * 60)
    logger.info("Binary Classification Inference UI")
    logger.info("=" * 60)

    # Validate settings
    errors = settings.validate()
    if errors:
        logger.warning("Configuration warnings:")
        for error in errors:
            logger.warning(f"  - {error}")

    if not PYQT5_AVAILABLE:
        print("=" * 60)
        print("PyQt5 not installed - running in console mode")
        print("=" * 60)
        print("\nTo use the GUI, install PyQt5:")
        print("  pip install PyQt5")
        print("\nAvailable modules:")
        print("  ✓ app.io.envi - ENVI hyperspectral reader")
        print("  ✓ app.preprocess.spectral - SNV, wavelength filtering")
        print("  ✓ app.models.loader_new - Model loading")
        print("  ✓ app.models.adapters_new - Model adapters")
        print("  ✓ app.inference.prob_map - Probability map generation")
        print("  ✓ app.postprocess.pipeline - Postprocessing pipeline")
        print("  ✓ app.tuning.optuna_runner - Optuna hyperparameter tuning")
        print("  ✓ app.metrics.classification - Classification metrics")
        print("=" * 60)

        # Demo of programmatic usage
        print("\nProgrammatic usage example:")
        print("  from app.tuning import run_full_tuning_pipeline")
        print("  from app.data.dataset import load_and_prepare_splits")
        print("  ")
        print("  train_df, val_df, test_df = load_and_prepare_splits(...)")
        print("  results = run_full_tuning_pipeline(train_df, val_df, test_df, ...)")

        return 0

    # Create and run application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    logger.info("UI started")

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())


