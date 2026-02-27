"""
Binary Classification Inference UI - PyQt5 Viewer Application

This module contains the PyQt5 GUI application for interactive visualization
of binary classification (CRACK vs REGULAR) inference results.

Goal: See where we get False Positives for the balanced binary models.

Uses the same preprocessing as training:
- Wavelength filtering (450-925 nm)
- SNV normalization per spectrum

Available balanced models:
- Logistic_Regression_(L1)_Balanced.pkl
- PLS-DA_Balanced.pkl
- Random_Forest_Balanced.pkl
- SVM_(RBF)_Balanced.pkl
- XGBoost_Balanced.pkl
"""
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[5]

import os
import sys
import logging
import time
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import joblib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QStatusBar,
    QDoubleSpinBox, QComboBox, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import spectral as spy

# Add project path
project_path = Path(__file__).resolve().parents[5]
sys.path.append(str(project_path))

# Import preprocessing function
from src.preprocessing.spectral_preprocessing import _snv

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Binary.Classifier.UI")

# ===== Config =====
MODELS_DIR = Path(str(_PROJECT_ROOT / r"experiments/pixel_level_classifier_2_classes/models"))
DEFAULT_SEARCH_FOLDER = str(_PROJECT_ROOT / r"data/raw")
DEFAULT_DATASET_CSV = str(_PROJECT_ROOT / r"src/preprocessing/prepare_dataset_for_full_image_classification/hole_image/first_date_dataset/early_detection_dataset.csv")

# Balanced models only
AVAILABLE_MODELS = {
    "Logistic Regression (L1) Balanced": MODELS_DIR / "Logistic_Regression_(L1)_Balanced.pkl",
    "PLS-DA Balanced": MODELS_DIR / "PLS-DA_Balanced.pkl",
    "Random Forest Balanced": MODELS_DIR / "Random_Forest_Balanced.pkl",
    "SVM (RBF) Balanced": MODELS_DIR / "SVM_(RBF)_Balanced.pkl",
    "XGBoost Balanced": MODELS_DIR / "XGBoost_Balanced.pkl",
}

# Wavelength range (same as training)
WL_MIN = 450
WL_MAX = 925


def load_cube(hdr_path: str) -> np.ndarray:
    """Load hyperspectral cube from ENVI header."""
    # Try different data file extensions
    base_path = hdr_path.replace(".hdr", "")
    dat_candidates = [base_path + ".raw", base_path + ".dat", base_path + ".bin"]

    dat_path = None
    for candidate in dat_candidates:
        if os.path.exists(candidate):
            dat_path = candidate
            break

    if dat_path is None:
        # Let spectral library try to find it automatically
        dat_path = hdr_path.replace(".hdr", ".raw")

    logger.info(f"Loading HSI cube: {hdr_path}")
    logger.info(f"Data file: {dat_path}")

    cube = np.array(spy.envi.open(hdr_path, dat_path).load())
    logger.info(f"Loaded cube shape: {cube.shape}, dtype: {cube.dtype}")
    return cube


def get_wavelengths_from_hdr(hdr_path: str) -> np.ndarray:
    """Extract wavelengths from HDR file."""
    try:
        # Try different data file extensions
        base_path = hdr_path.replace(".hdr", "")
        dat_candidates = [base_path + ".raw", base_path + ".dat", base_path + ".bin"]

        dat_path = None
        for candidate in dat_candidates:
            if os.path.exists(candidate):
                dat_path = candidate
                break

        if dat_path is None:
            dat_path = hdr_path.replace(".hdr", ".raw")

        hdr = spy.envi.open(hdr_path, dat_path)
        if hasattr(hdr, 'bands') and hasattr(hdr.bands, 'centers'):
            return np.array(hdr.bands.centers)
        # Try to read metadata
        if hasattr(hdr, 'metadata') and 'wavelength' in hdr.metadata:
            wl = hdr.metadata['wavelength']
            return np.array([float(w) for w in wl])
    except Exception as e:
        logger.warning(f"Could not extract wavelengths from HDR: {e}")
    return None


def preprocess_cube_for_inference(cube: np.ndarray, wavelengths: np.ndarray = None) -> tuple:
    """
    Apply the same preprocessing as training:
    1. Filter wavelengths to [WL_MIN, WL_MAX]
    2. Apply SNV normalization per spectrum

    Returns:
        X: (N, C) array of preprocessed spectra
        band_indices: indices of selected bands
    """
    H, W, C = cube.shape

    # Get band indices for wavelength filtering
    if wavelengths is not None:
        band_indices = []
        for i, wl in enumerate(wavelengths):
            if WL_MIN <= wl <= WL_MAX:
                band_indices.append(i)
        band_indices = np.array(band_indices)
        logger.info(f"Selected {len(band_indices)} bands in range [{WL_MIN}, {WL_MAX}] nm")
    else:
        # If no wavelengths available, use all bands
        logger.warning("No wavelength info available, using all bands")
        band_indices = np.arange(C)

    # Select bands
    cube_filtered = cube[:, :, band_indices]

    # Reshape to (N, C)
    X_raw = cube_filtered.reshape(-1, len(band_indices))

    # Handle NaN/Inf
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply SNV normalization
    X = _snv(X_raw.astype(np.float32))

    logger.info(f"Preprocessed cube: input {H}x{W}x{C} -> output {X.shape}")
    return X, band_indices


def predict_proba_any(model, X):
    """Try common attributes to find a classifier with predict_proba."""
    for attr in (None, "estimator_", "clf", "model", "lda", "tree", "classifier", "classifier_", "steps"):
        try:
            if attr is None:
                m = model
            elif attr == "steps":
                # For sklearn Pipeline
                if hasattr(model, 'steps'):
                    m = model.steps[-1][1]  # Get last step
                else:
                    continue
            else:
                m = getattr(model, attr, None)
                if m is None:
                    continue

            if hasattr(m, "predict_proba"):
                return m.predict_proba(X)
        except Exception:
            continue

    # For sklearn Pipeline, try directly
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)

    raise AttributeError("No predict_proba on provided model")


def load_model(model_path: Path):
    """Load a pickled model."""
    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Model loaded: {type(model)}")
    return model


def find_rgb_image(folder_path: str) -> Optional[str]:
    """Find RGB image in folder."""
    import glob

    # Check for RGB subfolder
    rgb_folder = os.path.join(folder_path, "RGB")
    if os.path.exists(rgb_folder):
        for pattern in ["*.JPG", "*.jpg", "*.png", "*.PNG"]:
            matches = glob.glob(os.path.join(rgb_folder, pattern))
            if matches:
                return matches[0]

    # Check in main folder
    for pattern in ["*canon*.jpg", "*canon*.png", "*RGB*.jpg", "*RGB*.png", "*.JPG", "*.jpg", "*.png"]:
        matches = glob.glob(os.path.join(folder_path, pattern), recursive=False)
        if matches:
            return matches[0]
    return None


def find_hdr_file(folder_path: str) -> Optional[str]:
    """Find HDR file in folder."""
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".hdr") and "white" not in f.lower() and "dark" not in f.lower():
                return os.path.join(root, f)
    return None


def colorize_binary_mask(gray: np.ndarray, mask: np.ndarray, color: tuple = (0, 215, 255)) -> np.ndarray:
    """Create colored overlay on grayscale for binary mask."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = np.array(color, dtype=np.uint8)
    alpha = 0.5
    rgb[mask] = (alpha * color + (1 - alpha) * rgb[mask]).astype(np.uint8)
    return rgb


def draw_fp_markers(image: np.ndarray, fp_mask: np.ndarray, color: tuple = (0, 0, 255)) -> np.ndarray:
    """Draw red dots on false positive locations."""
    result = image.copy()

    # Find contours of FP blobs
    contours, _ = cv2.findContours(
        fp_mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Draw red circle for FP
            cv2.circle(result, (cx, cy), 5, color, -1)
            cv2.circle(result, (cx, cy), 7, (0, 0, 0), 1)

    return result


class BinaryClassInferenceViewer(QMainWindow):
    """Main UI for binary classification inference and FP visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Classification Inference - False Positive Analysis")
        self.setGeometry(100, 100, 1800, 900)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # State
        self.hsi_cube: Optional[np.ndarray] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.hdr_path: Optional[str] = None
        self.rgb_image: Optional[np.ndarray] = None
        self.folder_path: str = ""
        self.current_band: int = 0

        # Model state
        self.current_model = None
        self.current_model_name: str = ""

        # Detection results
        self.prob_map: Optional[np.ndarray] = None
        self.detection_mask: Optional[np.ndarray] = None

        # Threshold
        self.prob_threshold = 0.5

        # Ground truth (if available)
        self.ground_truth_mask: Optional[np.ndarray] = None
        self.sample_label: Optional[int] = None  # 0=REGULAR, 1=CRACK

        # Navigation
        self.available_folders: List[str] = []
        self.current_folder_idx: int = -1

        # Dataset navigation
        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_current_idx: int = -1

        self._build_ui()
        self._load_first_model()
        self._load_dataset(DEFAULT_DATASET_CSV)  # Auto-load default dataset

    def _build_ui(self):
        """Build the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Image panels ---
        images_row = QHBoxLayout()
        layout.addLayout(images_row)

        # Panel 1: HSI Band
        panel1 = QVBoxLayout()
        panel1.addWidget(QLabel("1. HSI Band (Gray)"))
        self.hsi_band_label = QLabel()
        self.hsi_band_label.setFixedSize(400, 400)
        self.hsi_band_label.setAlignment(Qt.AlignCenter)
        self.hsi_band_label.setStyleSheet("border: 2px solid #333; background-color: #1a1a1a;")
        panel1.addWidget(self.hsi_band_label)
        images_row.addLayout(panel1)

        # Panel 2: Detection Probability Map
        panel2 = QVBoxLayout()
        panel2.addWidget(QLabel("2. Probability Map (CRACK)"))
        self.prob_map_label = QLabel()
        self.prob_map_label.setFixedSize(400, 400)
        self.prob_map_label.setAlignment(Qt.AlignCenter)
        self.prob_map_label.setStyleSheet("border: 2px solid #333; background-color: #1a1a1a;")
        panel2.addWidget(self.prob_map_label)
        images_row.addLayout(panel2)

        # Panel 3: Binary Detection
        panel3 = QVBoxLayout()
        panel3.addWidget(QLabel("3. Binary Detection (Thresholded)"))
        self.detection_label = QLabel()
        self.detection_label.setFixedSize(400, 400)
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.detection_label.setStyleSheet("border: 2px solid #333; background-color: #1a1a1a;")
        panel3.addWidget(self.detection_label)
        images_row.addLayout(panel3)

        # Panel 4: FP Overlay on RGB
        panel4 = QVBoxLayout()
        panel4.addWidget(QLabel("4. False Positives on RGB (Red=FP)"))
        self.fp_label = QLabel()
        self.fp_label.setFixedSize(400, 400)
        self.fp_label.setAlignment(Qt.AlignCenter)
        self.fp_label.setStyleSheet("border: 2px solid #f44336; background-color: #1a1a1a;")
        panel4.addWidget(self.fp_label)
        images_row.addLayout(panel4)

        # --- Controls ---
        controls_row = QHBoxLayout()
        layout.addLayout(controls_row)

        # Load CSV button
        load_csv_btn = QPushButton("📋 Load CSV")
        load_csv_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        load_csv_btn.clicked.connect(self._choose_csv)
        controls_row.addWidget(load_csv_btn)

        # Load folder button
        load_btn = QPushButton("📁 Load Folder")
        load_btn.setStyleSheet("padding: 8px;")
        load_btn.clicked.connect(self._choose_folder)
        controls_row.addWidget(load_btn)

        # Dataset Navigation
        self.prev_btn = QPushButton("← Prev Sample")
        self.prev_btn.clicked.connect(self._load_prev_sample)
        self.prev_btn.setEnabled(False)
        controls_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next Sample →")
        self.next_btn.clicked.connect(self._load_next_sample)
        self.next_btn.setEnabled(False)
        controls_row.addWidget(self.next_btn)

        self.folder_label = QLabel("Sample: N/A")
        self.folder_label.setMinimumWidth(300)
        controls_row.addWidget(self.folder_label)

        controls_row.addStretch()

        # Model selector
        controls_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(250)
        for name in AVAILABLE_MODELS.keys():
            self.model_combo.addItem(name)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        controls_row.addWidget(self.model_combo)

        # Threshold control
        controls_row.addWidget(QLabel("Prob Threshold:"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.05)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(self.prob_threshold)
        self.thr_spin.valueChanged.connect(self._on_threshold_changed)
        controls_row.addWidget(self.thr_spin)

        # Band slider
        controls_row.addWidget(QLabel("Band:"))
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(100)
        self.band_slider.setValue(0)
        self.band_slider.setMinimumWidth(100)
        self.band_slider.valueChanged.connect(self._on_band_changed)
        controls_row.addWidget(self.band_slider)

        self.band_value_label = QLabel("0")
        controls_row.addWidget(self.band_value_label)

        # --- Action buttons ---
        actions_row = QHBoxLayout()
        layout.addLayout(actions_row)

        run_btn = QPushButton("🔬 Run Inference")
        run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #4CAF50; color: white;")
        run_btn.clicked.connect(self._run_inference)
        actions_row.addWidget(run_btn)

        # Sample label selector (for FP analysis)
        actions_row.addWidget(QLabel("Sample Label:"))
        self.label_combo = QComboBox()
        self.label_combo.addItem("0 - REGULAR (Expected no crack)", 0)
        self.label_combo.addItem("1 - CRACK (Expected crack)", 1)
        self.label_combo.addItem("Unknown", -1)
        self.label_combo.setCurrentIndex(2)  # Default to unknown
        self.label_combo.currentIndexChanged.connect(self._on_label_changed)
        actions_row.addWidget(self.label_combo)

        actions_row.addStretch()

        # Stats display
        self.stats_label = QLabel("Stats: N/A")
        self.stats_label.setStyleSheet("font-family: monospace; padding: 5px; background-color: #e3f2fd;")
        actions_row.addWidget(self.stats_label)

        # --- Info box ---
        info_box = QGroupBox("Preprocessing Info")
        info_layout = QVBoxLayout(info_box)
        info_text = f"""
        <b>Preprocessing Pipeline (same as training):</b><br>
        1. Wavelength filtering: {WL_MIN} - {WL_MAX} nm<br>
        2. SNV (Standard Normal Variate) normalization per spectrum<br>
        <br>
        <b>Models:</b> Balanced models from experiments/pixel_level_classifier_2_classes/models/<br>
        <b>Classes:</b> 0 = REGULAR, 1 = CRACK
        """
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_box)

        self.status_bar.showMessage("Load a folder with HSI data to start.")

    def _load_first_model(self):
        """Load the first model on startup."""
        if AVAILABLE_MODELS:
            first_name = list(AVAILABLE_MODELS.keys())[0]
            first_path = AVAILABLE_MODELS[first_name]
            try:
                self.current_model = load_model(first_path)
                self.current_model_name = first_name
                logger.info(f"Loaded model: {first_name}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                QMessageBox.warning(self, "Model Load Error", f"Failed to load model:\n{e}")

    def _on_model_changed(self, index: int):
        """Handle model selection change."""
        model_names = list(AVAILABLE_MODELS.keys())
        if 0 <= index < len(model_names):
            name = model_names[index]
            path = AVAILABLE_MODELS[name]
            try:
                self.current_model = load_model(path)
                self.current_model_name = name
                logger.info(f"Switched to model: {name}")
                self.status_bar.showMessage(f"Model loaded: {name}")

                # Re-run inference if we have data
                if self.hsi_cube is not None:
                    self._run_inference()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                QMessageBox.warning(self, "Model Load Error", f"Failed to load model:\n{e}")

    def _on_threshold_changed(self, value: float):
        """Handle threshold change."""
        self.prob_threshold = value
        if self.prob_map is not None:
            self._update_detection_display()

    def _on_band_changed(self, value: int):
        """Handle band slider change."""
        self.current_band = value
        self.band_value_label.setText(str(value))
        self._update_band_display()

    def _on_label_changed(self, index: int):
        """Handle sample label change."""
        self.sample_label = self.label_combo.currentData()
        if self.prob_map is not None:
            self._update_detection_display()

    def _choose_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select HSI Data Folder", DEFAULT_SEARCH_FOLDER
        )
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder_path: str):
        """Load HSI data from folder."""
        self.folder_path = folder_path
        self.folder_label.setText(f"Folder: {os.path.basename(folder_path)}")

        # Update current folder index if in available folders
        if folder_path in self.available_folders:
            self.current_folder_idx = self.available_folders.index(folder_path)

        # Enable navigation buttons
        self.prev_btn.setEnabled(self.current_folder_idx > 0)
        self.next_btn.setEnabled(self.current_folder_idx < len(self.available_folders) - 1)

        # Find HDR file
        hdr_path = find_hdr_file(folder_path)
        if not hdr_path:
            QMessageBox.warning(self, "No HDR File", "Could not find HDR file in selected folder.")
            return

        self.hdr_path = hdr_path

        # Load cube
        try:
            self.hsi_cube = load_cube(hdr_path)
            self.wavelengths = get_wavelengths_from_hdr(hdr_path)

            # Update band slider
            H, W, C = self.hsi_cube.shape
            self.band_slider.setMaximum(C - 1)
            self.band_slider.setValue(C // 2)

            self._update_band_display()

            # Find RGB image
            rgb_path = find_rgb_image(folder_path)
            if rgb_path:
                self.rgb_image = cv2.imread(rgb_path)
                self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
                logger.info(f"Loaded RGB image: {rgb_path}")
            else:
                self.rgb_image = None
                logger.warning("No RGB image found")

            self.status_bar.showMessage(f"Loaded: {os.path.basename(folder_path)} | Cube: {self.hsi_cube.shape}")

        except Exception as e:
            logger.error(f"Failed to load cube: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load HSI cube:\n{e}")

    def _update_band_display(self):
        """Update the band display panel."""
        if self.hsi_cube is None:
            return

        band = self.hsi_cube[:, :, self.current_band]
        band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self._set_label_image(self.hsi_band_label, cv2.cvtColor(band_norm, cv2.COLOR_GRAY2RGB))

    def _run_inference(self):
        """Run inference on the loaded cube."""
        if self.hsi_cube is None:
            QMessageBox.warning(self, "No Data", "Please load HSI data first.")
            return

        if self.current_model is None:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return

        self.status_bar.showMessage("Running inference...")
        QApplication.processEvents()

        try:
            t0 = time.time()

            # Preprocess cube (wavelength filtering + SNV)
            X, band_indices = preprocess_cube_for_inference(self.hsi_cube, self.wavelengths)

            # Get predictions
            H, W, C = self.hsi_cube.shape

            # Predict in batches to avoid memory issues
            batch_size = 50000
            n_samples = X.shape[0]
            probs_list = []

            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                try:
                    batch_probs = predict_proba_any(self.current_model, batch)
                    # Get probability for class 1 (CRACK)
                    if batch_probs.ndim == 2 and batch_probs.shape[1] >= 2:
                        probs_list.append(batch_probs[:, 1])
                    else:
                        probs_list.append(batch_probs.ravel())
                except Exception as e:
                    logger.error(f"Prediction failed for batch: {e}")
                    probs_list.append(np.zeros(batch.shape[0]))

            probs = np.concatenate(probs_list)
            self.prob_map = probs.reshape(H, W)

            elapsed = time.time() - t0
            logger.info(f"Inference complete in {elapsed:.2f}s")
            self.status_bar.showMessage(f"Inference complete ({elapsed:.2f}s)")

            # Update displays
            self._update_prob_display()
            self._update_detection_display()

        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            QMessageBox.critical(self, "Inference Error", f"Inference failed:\n{e}")

    def _update_prob_display(self):
        """Update probability map display."""
        if self.prob_map is None:
            return

        # Create heatmap
        prob_norm = (self.prob_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        self._set_label_image(self.prob_map_label, heatmap)

    def _update_detection_display(self):
        """Update detection and FP display."""
        if self.prob_map is None:
            return

        H, W = self.prob_map.shape

        # Create binary mask
        self.detection_mask = self.prob_map >= self.prob_threshold

        # Get base gray image
        band = self.hsi_cube[:, :, self.current_band]
        band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Detection overlay (yellow on detections)
        detection_vis = colorize_binary_mask(band_norm, self.detection_mask, color=(0, 255, 255))
        self._set_label_image(self.detection_label, detection_vis)

        # FP visualization
        if self.rgb_image is not None:
            rgb_resized = cv2.resize(self.rgb_image, (W, H))
        else:
            rgb_resized = cv2.cvtColor(band_norm, cv2.COLOR_GRAY2RGB)

        # Calculate stats
        n_positive = int(self.detection_mask.sum())
        total_pixels = H * W
        percent_positive = 100.0 * n_positive / total_pixels

        # Determine FP based on sample label
        sample_label = self.label_combo.currentData()

        if sample_label == 0:
            # Sample is REGULAR -> all detections are False Positives
            fp_mask = self.detection_mask
            fp_count = n_positive
            fp_vis = draw_fp_markers(rgb_resized, fp_mask, color=(255, 0, 0))  # Red for FP
            stats_text = f"REGULAR sample: {fp_count} FPs ({percent_positive:.2f}% of image)"
        elif sample_label == 1:
            # Sample is CRACK -> detections are True Positives (show in green)
            fp_mask = np.zeros_like(self.detection_mask)
            fp_vis = draw_fp_markers(rgb_resized, self.detection_mask, color=(0, 255, 0))  # Green for TP
            stats_text = f"CRACK sample: {n_positive} TPs ({percent_positive:.2f}% of image)"
        else:
            # Unknown label - show all detections in yellow
            fp_vis = draw_fp_markers(rgb_resized, self.detection_mask, color=(255, 255, 0))  # Yellow
            stats_text = f"Unknown label: {n_positive} detections ({percent_positive:.2f}% of image)"

        self._set_label_image(self.fp_label, fp_vis)
        self.stats_label.setText(stats_text)

    def _set_label_image(self, label: QLabel, image: np.ndarray):
        """Set image on a QLabel."""
        h, w = image.shape[:2]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        bytes_per_line = 3 * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit label
        scaled = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _load_prev_folder(self):
        """Load previous folder in navigation."""
        if self.current_folder_idx > 0:
            self.current_folder_idx -= 1
            self._load_folder(self.available_folders[self.current_folder_idx])
            self._update_nav_buttons()

    def _load_next_folder(self):
        """Load next folder in navigation."""
        if self.current_folder_idx < len(self.available_folders) - 1:
            self.current_folder_idx += 1
            self._load_folder(self.available_folders[self.current_folder_idx])
            self._update_nav_buttons()

    def _update_nav_buttons(self):
        """Update navigation button states."""
        self.prev_btn.setEnabled(self.current_folder_idx > 0)
        self.next_btn.setEnabled(self.current_folder_idx < len(self.available_folders) - 1)

    def _choose_csv(self):
        """Open CSV file selection dialog."""
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset CSV",
            os.path.dirname(DEFAULT_DATASET_CSV),
            "CSV Files (*.csv)"
        )
        if csv_path:
            self._load_dataset(csv_path)

    def _load_dataset(self, csv_path: str):
        """Load dataset from CSV file."""
        try:
            self.dataset_df = pd.read_csv(csv_path)
            self.dataset_current_idx = -1

            # Validate required columns
            required_cols = ['grape_id', 'image_path', 'label']
            missing = [c for c in required_cols if c not in self.dataset_df.columns]
            if missing:
                QMessageBox.warning(self, "Invalid CSV", f"Missing columns: {missing}")
                return

            n_samples = len(self.dataset_df)
            n_regular = (self.dataset_df['label'] == 0).sum()
            n_crack = (self.dataset_df['label'] == 1).sum()

            logger.info(f"Loaded dataset: {n_samples} samples ({n_regular} REGULAR, {n_crack} CRACK)")
            self.status_bar.showMessage(f"Dataset loaded: {n_samples} samples | Use Next/Prev to navigate")

            # Enable navigation buttons
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)

            # Load first sample
            if n_samples > 0:
                self._load_sample_by_idx(0)

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset:\n{e}")

    def _load_sample_by_idx(self, idx: int):
        """Load a sample from the dataset by index."""
        if self.dataset_df is None or idx < 0 or idx >= len(self.dataset_df):
            return

        self.dataset_current_idx = idx
        row = self.dataset_df.iloc[idx]

        grape_id = row['grape_id']
        folder_path = row['image_path']
        label = int(row['label'])

        # Update label selector
        if label == 0:
            self.label_combo.setCurrentIndex(0)  # REGULAR
        elif label == 1:
            self.label_combo.setCurrentIndex(1)  # CRACK
        else:
            self.label_combo.setCurrentIndex(2)  # Unknown

        # Update folder label
        self.folder_label.setText(f"[{idx+1}/{len(self.dataset_df)}] {grape_id} | Label: {'REGULAR' if label == 0 else 'CRACK'}")

        # Load the folder
        self._load_folder(folder_path)

    def _load_prev_sample(self):
        """Load previous sample from dataset."""
        if self.dataset_df is not None and self.dataset_current_idx > 0:
            self._load_sample_by_idx(self.dataset_current_idx - 1)

    def _load_next_sample(self):
        """Load next sample from dataset."""
        if self.dataset_df is not None and self.dataset_current_idx < len(self.dataset_df) - 1:
            self._load_sample_by_idx(self.dataset_current_idx + 1)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = BinaryClassInferenceViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
