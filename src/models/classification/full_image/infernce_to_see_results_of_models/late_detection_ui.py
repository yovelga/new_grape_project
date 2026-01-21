"""
Late Detection UI - PyQt5 Viewer Application

This module contains the PyQt5 GUI application for interactive visualization
of late detection results with 4-panel layout:
1. HSI Band (default: band 138)
2. HSI Detection (binary mask overlay)
3. HSI Patch Grid (grid-based visualization)
4. RGB Image (Canon/standard RGB)
"""

import os
import sys
import logging
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QStatusBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QInputDialog,
    QMessageBox, QLineEdit, QGroupBox, QTableWidget, QTableWidgetItem,
    QPlainTextEdit, QProgressBar, QTabWidget, QGridLayout, QProgressDialog,
    QDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QTextCursor

# SAM2 imports for segmentation
project_path = Path(__file__).resolve().parents[4]
sys.path.append(str(project_path))
from src.preprocessing.MaskGenerator.segment_object_module import create_point_segmenter
from src.preprocessing.MaskGenerator.mask_generator_module import (
    initial_settings,
    initialize_sam2_predictor,
)

# Import preprocessing for binary models (SNV normalization + wavelength filtering)
from src.preprocessing.spectral_preprocessing import _snv

# Import core functions
from late_detection_core import (
    load_cube, load_model_and_scaler, per_pixel_probs, analyze_grid,
    filter_blobs_by_shape, filter_blobs_advanced, find_band_index_for_wavelength, find_scaler,
    get_wavelengths_from_hdr, GRID_COLOR_BUCKETS
)

# Import grid search functions
from grid_search_blob_patch import (
    GridSearchConfig, run_grid_search, load_probability_maps_from_results,
    evaluate_single_combination
)
from run_late_detection_inference import prepare_and_run_inference

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HSI.Patch.UI")

# ===== Config =====
AVAILABLE_MODELS = {
    "NEW LDA Multi-class": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\lda_model_multi_class.joblib",
    "LDA F1-Optimized (CRACK)": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_sam2_with_F1\lda_model_multi_class_f1_score.joblib",
    "OLD LDA [1=CRACK, 0=regular]": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\pixel_level\simple_classification_leave_one_out\comare_all_models\models\LDA_Balanced.pkl",
    "XGBoost Row1 Final": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_sam2_with_F1_XGBOOST\xgboost_row1_final.joblib",
    # Binary Balanced Models (use SNV + wavelength filtering preprocessing)
    "Binary: Logistic Regression (L1) Balanced": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_2_classes\models\Logistic_Regression_(L1)_Balanced.pkl",
    "Binary: PLS-DA Balanced": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_2_classes\models\PLS-DA_Balanced.pkl",
    "Binary: Random Forest Balanced": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_2_classes\models\Random_Forest_Balanced.pkl",
    "Binary: SVM (RBF) Balanced": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_2_classes\models\SVM_(RBF)_Balanced.pkl",
    "Binary: XGBoost Balanced": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_2_classes\models\XGBoost_Balanced.pkl",

    # === 3-Class Models (maxpixel experiment) ===
    "[3Class] Logistic Regression (L1)": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\3class\Logistic_Regression_(L1)_Balanced.pkl",
    "[3Class] Random Forest": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\3class\Random_Forest_Balanced.pkl",
    "[3Class] SVM (RBF)": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\3class\SVM_(RBF)_Balanced.pkl",
    "[3Class] XGBoost": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\3class\XGBoost_Balanced.pkl",
    # === Multi-Class Models (maxpixel experiment) ===
    "[MultiClass] Logistic Regression (L1)": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\multiclass\Logistic_Regression_(L1)_Balanced.pkl",
    "[MultiClass] Random Forest": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\multiclass\Random_Forest_Balanced.pkl",
    "[MultiClass] SVM (RBF)": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\multiclass\SVM_(RBF)_Balanced.pkl",
    "[MultiClass] XGBoost": r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_maxpixel_per_class_like_crack\models\multiclass\XGBoost_Balanced.pkl",
}
DEFAULT_SEARCH_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"
# DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\hole_image\late_detection_test\dataset_csvs\row1_only_cracked.csv"
# DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\hole_image\late_detection_test\dataset_csvs\row1_all_weeks.csv"
DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\hole_image\generate_test_row_1\dataset_csvs\row1_clean_pre_august_plus_tagged_cracks.csv"
# DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\hole_image\late_detection\late_detection_dataset.csv"
# DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\hole_image\first_date_dataset\early_detection_dataset.csv"
# DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\hole_image\early_detection\early_detection_dataset.csv"
RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model\Results"


# ===== Helper functions for UI =====

def color_for_percent(p: float):
    """Return color for given percentage based on buckets."""
    for thr, col in GRID_COLOR_BUCKETS:
        if p >= thr:
            return col
    return None


def overlay_on_band(band_img: np.ndarray, grid_stats: List[Dict], alpha: float = 0.35) -> np.ndarray:
    """Overlay grid visualization on band image."""
    rgb = cv2.cvtColor(band_img, cv2.COLOR_GRAY2RGB)
    over = rgb.copy()
    for c in grid_stats:
        col = color_for_percent(c["percent_cracked"])
        if col:
            r0, c0, r1, c1 = c["row0"], c["col0"], c["row1"], c["col1"]
            cv2.rectangle(over, (c0, r0), (c1 - 1, r1 - 1), col, -1)
            cv2.rectangle(over, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 0), 1)
    merged = cv2.addWeighted(over, max(alpha, 0.45), rgb, 1 - max(alpha, 0.45), 0)
    return merged


def colorize_binary_mask(gray, mask):
    """Create yellow overlay on grayscale for binary mask."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = np.array([0, 215, 255], dtype=np.uint8)  # yellow
    alpha = 0.35
    rgb[mask] = (alpha * color + (1 - alpha) * rgb[mask]).astype(np.uint8)
    return rgb


def build_hsi_rgb_composite(cube: np.ndarray, band_idx: int) -> np.ndarray:
    """Build RGB composite from HSI cube (3-band composite)."""
    try:
        B = cube.shape[2]
        # Choose three bands spread across spectrum
        b_idx = max(0, min(B - 1, int(B * 0.1)))
        g_idx = max(0, min(B - 1, int(B * 0.5)))
        r_idx = max(0, min(B - 1, int(B * 0.9)))

        hsi_r = cv2.normalize(cube[:, :, r_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsi_g = cv2.normalize(cube[:, :, g_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsi_b = cv2.normalize(cube[:, :, b_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsi_rgb = cv2.merge([hsi_r, hsi_g, hsi_b])  # R,G,B order
        return hsi_rgb
    except Exception as e:
        logger.warning(f"Failed to build HSI RGB composite: {e}")


# ===== SAM Segmentation Helper Functions =====

def extract_blob_centroids(binary_mask: np.ndarray, max_blobs: int = 100) -> List[Tuple[int, int]]:
    """
    Extract centroid coordinates from all connected components in a binary mask.

    Args:
        binary_mask: Binary detection mask (0 or 255/True/False)
        max_blobs: Maximum number of blobs to process (largest by area)

    Returns:
        List of (x, y) tuples representing blob centroids
    """
    if binary_mask.dtype == bool:
        mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    else:
        mask_uint8 = binary_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(-areas)
    sorted_indices = sorted_indices[:max_blobs]

    centroid_points = []
    for idx in sorted_indices:
        label_idx = idx + 1
        cx, cy = centroids[label_idx]
        centroid_points.append((int(cx), int(cy)))

    logger.info(f"Extracted {len(centroid_points)} blob centroids from {num_labels-1} total blobs")
    return centroid_points


def create_sam_segment_overlay(
    base_image: np.ndarray,
    masks: List[np.ndarray],
    prob_map: Optional[np.ndarray] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create colored overlay showing SAM segments on base image.

    Args:
        base_image: RGB or grayscale base image (H, W) or (H, W, 3) - should be 512x512
        masks: List of boolean masks from SAM segmentation (must match base_image size) - 512x512
        prob_map: Optional probability map to color segments by intensity - should also be 512x512
        alpha: Overlay transparency (0-1)

    Returns:
        RGB overlay image with colored segments

    Note: After refactoring, base_image, masks, and prob_map should all be 512x512 and aligned.
    """
    rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB) if len(base_image.shape) == 2 else base_image.copy()
    overlay = rgb.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255),
    ]

    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue

        color = np.array(colors[i % len(colors)], dtype=np.uint8)

        # Apply probability modulation if available
        if prob_map is not None:
            try:
                # With refactoring, dimensions should always match (512x512)
                if mask.shape == prob_map.shape:
                    # Direct indexing - optimal path
                    avg_prob = np.mean(prob_map[mask])
                    intensity = np.clip(avg_prob, 0.3, 1.0)
                    color = (color * intensity).astype(np.uint8)
                else:
                    # Fallback: resize prob_map if dimensions still mismatch
                    logger.warning(f"Dimension mismatch: mask {mask.shape} vs prob_map {prob_map.shape}")
                    prob_map_resized = cv2.resize(prob_map, (mask.shape[1], mask.shape[0]),
                                                  interpolation=cv2.INTER_LINEAR)
                    avg_prob = np.mean(prob_map_resized[mask])
                    intensity = np.clip(avg_prob, 0.3, 1.0)
                    color = (color * intensity).astype(np.uint8)
            except Exception as e:
                # If any error, just use base color without probability modulation
                logger.warning(f"Could not use prob_map for segment {i}: {e}")
                pass

        # Apply color with transparency
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

        # Draw contours for better visibility
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

    return overlay.astype(np.uint8)



# ===== Custom Logging Handler for UI =====

class QTextEditLogger(logging.Handler, QObject):
    """Custom logging handler that emits signals for GUI updates."""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        """Emit log record as signal."""
        msg = self.format(record)
        self.log_signal.emit(msg)


# ===== Crop Gallery Dialog =====

class CropGalleryDialog(QDialog):
    """
    Dialog to display a gallery of crop images in a scrollable grid.
    Shows CNN candidate crops for visual inspection.
    """

    def __init__(self, crops: List[np.ndarray], parent=None, title_suffix: str = ""):
        """
        Initialize crop gallery dialog.

        Args:
            crops: List of numpy arrays (RGB images) to display
            parent: Parent widget
            title_suffix: Optional suffix to add to window title (e.g., " - Segment Only")
        """
        super().__init__(parent)
        self.crops = crops
        self.setWindowTitle(f"CNN Crop Gallery ({len(crops)} crops){title_suffix}")
        self.setModal(True)
        self.resize(1000, 700)

        self._build_ui()

    def _build_ui(self):
        """Build the gallery UI with scrollable grid of crops."""
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(f"Total Crops: {len(self.crops)} | Click and scroll to inspect")
        info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e3f2fd;")
        layout.addWidget(info_label)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for grid
        container = QWidget()
        grid_layout = QGridLayout(container)
        grid_layout.setSpacing(10)

        # Add crops to grid (4 columns)
        cols = 4
        for i, crop in enumerate(self.crops):
            row = i // cols
            col = i % cols

            # Create label for crop
            crop_label = QLabel()
            crop_label.setFixedSize(220, 220)
            crop_label.setAlignment(Qt.AlignCenter)
            crop_label.setStyleSheet("border: 2px solid #2196F3; background-color: white;")

            # Convert numpy array to QPixmap
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

            h, w, ch = crop.shape
            bytes_per_line = ch * w
            q_img = QImage(crop.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(210, 210, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            crop_label.setPixmap(scaled_pixmap)

            # Create container with label and index
            crop_container = QWidget()
            crop_layout = QVBoxLayout(crop_container)
            crop_layout.setContentsMargins(0, 0, 0, 0)

            # Index label
            index_label = QLabel(f"Crop #{i}")
            index_label.setAlignment(Qt.AlignCenter)
            index_label.setStyleSheet("font-weight: bold; color: #2196F3;")

            crop_layout.addWidget(index_label)
            crop_layout.addWidget(crop_label)

            # Add size info
            size_label = QLabel(f"{w}x{h}px")
            size_label.setAlignment(Qt.AlignCenter)
            size_label.setStyleSheet("font-size: 9pt; color: #666;")
            crop_layout.addWidget(size_label)

            grid_layout.addWidget(crop_container, row, col)

        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #4CAF50; color: white;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ===== Grid Search Worker Thread =====

class GridSearchWorker(QThread):
    """Background worker thread for running grid search pipeline."""

    # Signals
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)  # current, total, message
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, dataset_csv, model_path, output_dir=None):
        super().__init__()
        self.dataset_csv = dataset_csv
        self.model_path = model_path
        self.output_dir = output_dir or os.path.dirname(__file__)
        self.should_stop = False

    def log(self, message):
        """Emit log message."""
        self.log_signal.emit(message)

    def run(self):
        """Run the grid search pipeline in background."""
        try:
            self.log("="*80)
            self.log("STARTING GRID SEARCH PIPELINE")
            self.log("="*80)
            self.log("")

            # Output file paths
            base_output = os.path.join(self.output_dir, "late_detection_with_prob_maps.csv")
            grid_output = os.path.join(self.output_dir, "grid_search_blob_patch_results.csv")
            test_output = os.path.join(self.output_dir, "test_set_blob_patch_results.csv")

            # Stage 1: Check if probability maps already exist
            self.log("Stage 1: Checking probability maps...")
            if os.path.exists(base_output):
                self.log(f"✓ Found existing probability maps: {base_output}")
                response = "reuse"  # For automation, reuse existing
                self.log("  → Reusing existing probability maps")
            else:
                response = "compute"

            if response == "compute" or not os.path.exists(base_output):
                self.log("Stage 1: Computing probability maps...")
                self.progress_signal.emit(0, 100, "Computing probability maps...")

                base_df = prepare_and_run_inference(
                    self.dataset_csv,
                    self.model_path,
                    base_output
                )

                self.log(f"✓ Computed probability maps for {len(base_df)} samples")
                self.log(f"  Dev samples: {len(base_df[base_df['row'] == 1])}")
                self.log(f"  Test samples: {len(base_df[base_df['row'] == 2])}")
            else:
                base_df = pd.read_csv(base_output)
                self.log(f"✓ Loaded {len(base_df)} samples from existing probability maps")

            if self.should_stop:
                self.finished_signal.emit(False, "Cancelled by user")
                return

            self.log("")
            self.log("="*80)
            self.log("Stage 2: Running Grid Search on Dev Set")
            self.log("="*80)

            # Load probability maps for dev set
            self.progress_signal.emit(10, 100, "Loading dev set probability maps...")
            prob_maps_dict, labels_dict = load_probability_maps_from_results(base_df, row_filter=1)

            if len(prob_maps_dict) == 0:
                self.finished_signal.emit(False, "No valid dev samples found!")
                return

            self.log(f"✓ Loaded {len(prob_maps_dict)} dev set probability maps")
            self.log(f"  Label distribution: {dict(pd.Series(list(labels_dict.values())).value_counts())}")
            self.log("")

            # Configure grid search - FULL MORPHOLOGY SEARCH WITHOUT PATCHES
            self.log("Configuring grid search hyperparameters...")
            config = GridSearchConfig(
                prob_thr_candidates=np.arange(0.6, 1.00, 0.02).tolist(),
                min_blob_size_candidates=np.arange(0, 501, 50).tolist(),
                # Enable all morphology filter combinations
                circularity_min_candidates=[0.0, 0.25, 0.5,0.6],
                circularity_max_candidates=[0.6, 0.8, 1.0],
                aspect_ratio_limit_candidates=np.arange(0,5,0.1).tolist(),
                solidity_max_candidates=np.arange(0.4,1,0.1).tolist(),
                # Disable patch-based analysis (use single large patch = entire image)
                patch_size_candidates=[520*520],  # Very large patch = entire image, no subdivision
                patch_pixel_ratio_candidates=[0.0],  # Only need 0% of pixels to trigger (always true)
                global_threshold_candidates=[0.001]
            )

            # Log detailed parameter space
            self.log("")
            self.log("📊 Hyperparameter Search Space:")
            self.log(f"  • Probability thresholds: {len(config.prob_thr_candidates)} values")
            self.log(f"    └─ Range: {config.prob_thr_candidates}")
            self.log(f"  • Blob size thresholds: {len(config.min_blob_size_candidates)} values")
            self.log(f"    └─ Range: [{min(config.min_blob_size_candidates)}, ..., {max(config.min_blob_size_candidates)}]")
            self.log(f"  • Circularity min: {len(config.circularity_min_candidates)} values {config.circularity_min_candidates}")
            self.log(f"  • Circularity max: {len(config.circularity_max_candidates)} values {config.circularity_max_candidates}")
            self.log(f"  • Aspect ratio limit: {len(config.aspect_ratio_limit_candidates)} values {config.aspect_ratio_limit_candidates}")
            self.log(f"  • Solidity max: {len(config.solidity_max_candidates)} values {config.solidity_max_candidates}")
            self.log(f"  • Patch size: {config.patch_size_candidates} (patches disabled)")
            self.log(f"  • Patch pixel ratio: {config.patch_pixel_ratio_candidates}")
            self.log(f"  • Global thresholds: {len(config.global_threshold_candidates)} values {config.global_threshold_candidates}")

            total_combos = config.total_combinations()
            morphology_combos = (len(config.circularity_min_candidates) *
                                len(config.circularity_max_candidates) *
                                len(config.aspect_ratio_limit_candidates) *
                                len(config.solidity_max_candidates))

            self.log("")
            self.log(f"🔬 Total combinations: {total_combos:,}")
            self.log(f"   └─ Morphology filters alone: {morphology_combos} combinations")

            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            self.log(f"⚡ Parallel processing: {cpu_count} CPU cores")

            # Estimate time
            est_time_per_combo = 0.15  # seconds (rough estimate)
            est_total_sec = (total_combos * est_time_per_combo) / cpu_count
            est_minutes = est_total_sec / 60
            self.log(f"⏱️  Estimated time: {est_minutes:.1f} - {est_minutes*2:.1f} minutes")
            self.log("")

            # Run grid search with progress updates (use all CPU cores for parallel processing)
            self.log("🚀 Starting grid search...")
            import time
            start_time = time.time()
            self.progress_signal.emit(20, 100, f"Grid search: 0/{total_combos} (0%)")

            # Define progress callback
            def progress_update(current, total, message):
                # Update progress bar and log
                progress_pct = int(20 + (60 * current / total))  # 20-80% range for grid search
                self.progress_signal.emit(progress_pct, 100, message)
                # Log every 5% increment
                pct = 100.0 * current / total
                if int(pct) % 5 == 0 and current > 0:
                    elapsed = time.time() - start_time
                    self.log(f"📊 {pct:.0f}% complete ({current:,}/{total:,}) | Elapsed: {elapsed:.1f}s")

            results = run_grid_search(prob_maps_dict, labels_dict, config, n_jobs=-1,
                                     progress_callback=progress_update)

            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60

            if self.should_stop:
                self.finished_signal.emit(False, "Cancelled by user")
                return

            # Log completion stats
            self.log("")
            self.log("="*80)
            self.log("✅ GRID SEARCH COMPLETED")
            self.log("="*80)
            self.log(f"⏱️  Total time: {elapsed_minutes:.2f} minutes ({elapsed_time:.1f} seconds)")
            self.log(f"📊 Tested: {len(results):,} combinations")
            self.log(f"⚡ Speed: {len(results)/elapsed_time:.1f} combinations/second")
            self.log(f"🎯 Valid results: {results['f1_score'].notna().sum()}/{len(results)}")
            self.log("")

            # Save results
            results.to_csv(grid_output, index=False)
            self.log(f"💾 Saved full results to: {grid_output}")

            # Get best parameters
            if len(results) > 0:
                best = results.iloc[0]
                self.log("")
                self.log("="*80)
                self.log("🏆 BEST HYPERPARAMETERS (Ranked by F1 Score)")
                self.log("="*80)
                self.log("Pixel Classification:")
                self.log(f"  • Probability threshold:  {best['prob_thr']:.4f}")
                self.log("")
                self.log("Blob Filtering:")
                self.log(f"  • Min blob size:          {int(best['min_blob_size'])} pixels")
                self.log(f"  • Circularity min:        {best.get('circularity_min', 'None')}")
                self.log(f"  • Circularity max:        {best.get('circularity_max', 'None')}")
                self.log(f"  • Aspect ratio limit:     {best.get('aspect_ratio_limit', 'None')}")
                self.log(f"  • Solidity max:           {best.get('solidity_max', 'None')}")
                self.log("")
                self.log("Patch Analysis:")
                self.log(f"  • Patch size:             {int(best['patch_size'])} (disabled)")
                self.log(f"  • Patch pixel ratio:      {best['patch_pixel_ratio']:.4f}")
                self.log("")
                self.log("Image-Level Decision:")
                self.log(f"  • Global threshold:       {best['global_threshold']:.4f}")
                self.log("")
                self.log("🎯 DEV SET PERFORMANCE:")
                self.log(f"  • Accuracy:   {best['accuracy']:.4f}  ({best['accuracy']*100:.2f}%)")
                self.log(f"  • Precision:  {best['precision']:.4f}  ({best['precision']*100:.2f}%)")
                self.log(f"  • Recall:     {best['recall']:.4f}  ({best['recall']*100:.2f}%)")
                self.log(f"  • F1 Score:   {best['f1_score']:.4f}  ({best['f1_score']*100:.2f}%)")
                self.log(f"  • ROC AUC:    {best['roc_auc']:.4f}  ({best['roc_auc']*100:.2f}%)")

                # Show top 5 configurations
                self.log("")
                self.log("📋 Top 5 Configurations:")
                for i in range(min(5, len(results))):
                    row = results.iloc[i]
                    self.log(f"  #{i+1}: F1={row['f1_score']:.4f}, "
                            f"Prec={row['precision']:.4f}, "
                            f"Rec={row['recall']:.4f} | "
                            f"prob_thr={row['prob_thr']:.3f}, "
                            f"blob_size={int(row['min_blob_size'])}, "
                            f"global_thr={row['global_threshold']:.3f}")

                # Stage 3: Evaluate on test set
                self.log("")
                self.log("="*80)
                self.log("Stage 3: Evaluating Best Parameters on Test Set")
                self.log("="*80)

                self.progress_signal.emit(80, 100, "Evaluating on test set...")

                # Load test set probability maps
                prob_maps_test, labels_test = load_probability_maps_from_results(base_df, row_filter=2)

                if len(prob_maps_test) > 0:
                    self.log(f"✓ Loaded {len(prob_maps_test)} test set probability maps")
                    self.log(f"  Label distribution: {dict(pd.Series(list(labels_test.values())).value_counts())}")

                    # Evaluate best params on test
                    self.log("")
                    self.log("Applying best hyperparameters to test set...")
                    test_start = time.time()

                    test_metrics = evaluate_single_combination(
                        prob_maps_test,
                        labels_test,
                        prob_thr=best['prob_thr'],
                        min_blob_size=int(best['min_blob_size']),
                        circularity_min=best.get('circularity_min', None),
                        circularity_max=best.get('circularity_max', None),
                        aspect_ratio_min=best.get('aspect_ratio_min', None),
                        aspect_ratio_limit=best.get('aspect_ratio_limit', None),
                        solidity_min=best.get('solidity_min', None),
                        solidity_max=best.get('solidity_max', None),
                        patch_size=int(best['patch_size']),
                        patch_pixel_ratio=best['patch_pixel_ratio'],
                        global_threshold=best['global_threshold']
                    )

                    test_elapsed = time.time() - test_start

                    # Save test results
                    test_df = pd.DataFrame([test_metrics])
                    test_df.to_csv(test_output, index=False)
                    self.log(f"💾 Saved test results to: {test_output}")

                    self.log("")
                    self.log("="*80)
                    self.log("🎯 TEST SET PERFORMANCE:")
                    self.log("="*80)
                    self.log(f"  • Accuracy:   {test_metrics['accuracy']:.4f}  ({test_metrics['accuracy']*100:.2f}%)")
                    self.log(f"  • Precision:  {test_metrics['precision']:.4f}  ({test_metrics['precision']*100:.2f}%)")
                    self.log(f"  • Recall:     {test_metrics['recall']:.4f}  ({test_metrics['recall']*100:.2f}%)")
                    self.log(f"  • F1 Score:   {test_metrics['f1_score']:.4f}  ({test_metrics['f1_score']*100:.2f}%)")
                    self.log(f"  • ROC AUC:    {test_metrics['roc_auc']:.4f}  ({test_metrics['roc_auc']*100:.2f}%)")
                    self.log(f"⏱️  Evaluation time: {test_elapsed:.2f} seconds")

                    # Compare dev vs test
                    self.log("")
                    self.log("📊 Dev vs Test Comparison:")
                    f1_diff = test_metrics['f1_score'] - best['f1_score']
                    acc_diff = test_metrics['accuracy'] - best['accuracy']
                    self.log(f"  • F1 Score:   {best['f1_score']:.4f} (dev) → {test_metrics['f1_score']:.4f} (test)  "
                            f"[{f1_diff:+.4f}]")
                    self.log(f"  • Accuracy:   {best['accuracy']:.4f} (dev) → {test_metrics['accuracy']:.4f} (test)  "
                            f"[{acc_diff:+.4f}]")

                    if abs(f1_diff) < 0.05:
                        self.log("  ✅ Good generalization (difference < 5%)")
                    elif f1_diff < -0.05:
                        self.log("  ⚠️  Possible overfitting (test score lower)")
                    else:
                        self.log("  📈 Test score higher than dev")
                    self.log(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
                    self.log(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
                    self.log("")
                    self.log(f"✓ Saved test results to: {test_output}")
                else:
                    self.log("⚠ No test samples found")

            self.progress_signal.emit(100, 100, "Complete!")

            self.log("")
            self.log("="*80)
            self.log("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            self.log("="*80)
            self.log("")
            self.log(f"Output files:")
            self.log(f"  1. {base_output}")
            self.log(f"  2. {grid_output}")
            self.log(f"  3. {test_output}")

            self.finished_signal.emit(True, "Grid search completed successfully!")

        except Exception as e:
            error_msg = f"Grid search failed: {str(e)}"
            self.log("")
            self.log("="*80)
            self.log("ERROR!")
            self.log("="*80)
            self.log(error_msg)
            logger.exception("Grid search exception:")
            self.finished_signal.emit(False, error_msg)

    def stop(self):
        """Request the worker to stop."""
        self.should_stop = True


# ===== Optuna Worker Thread =====

class OptunaWorker(QThread):
    """Background worker thread that runs an Optuna study and reports progress to the UI."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # emits current trial number
    finished_signal = pyqtSignal(bool, dict)  # success, best_result dict

    def __init__(self, dataset_csv: str, model_path: str, n_trials: int = 100, output_dir: str = None):
        super().__init__()
        self.dataset_csv = dataset_csv
        self.model_path = model_path
        self.n_trials = int(n_trials)
        self.output_dir = output_dir or os.path.dirname(__file__)
        self.should_stop = False
        self._study = None

    def log(self, msg: str):
        self.log_signal.emit(str(msg))

    def stop(self):
        """Request the worker to stop. This attempts to stop the running Optuna study."""
        self.should_stop = True
        try:
            if self._study is not None:
                self._study.stop()
                self.log("Optuna study stop requested")
        except Exception:
            pass

    def run(self):
        try:
            self.log("Starting Optuna optimization...")

            # Load probability maps and labels from results CSV
            try:
                import pandas as _pd
                df = _pd.read_csv(self.dataset_csv)

                # If 'status' column doesn't exist, add it with default 'ok' value
                if 'status' not in df.columns:
                    self.log("Note: 'status' column not found in dataset, adding default values")
                    df['status'] = 'ok'

                # If 'prob_map_path' column doesn't exist, try to construct it
                if 'prob_map_path' not in df.columns and 'image_path' in df.columns:
                    self.log("Note: 'prob_map_path' column not found, will need probability maps computed")
                    # This will be handled by prepare_and_run_inference

            except Exception as e:
                self.log(f"Failed to read dataset CSV: {e}")
                self.finished_signal.emit(False, {"error": str(e)})
                return

            from grid_search_blob_patch import load_probability_maps_from_results, evaluate_single_combination

            # Check if we need to compute probability maps first
            if 'prob_map_path' not in df.columns or df['prob_map_path'].isna().all():
                self.log("Probability maps not found in dataset, computing them first...")
                try:
                    from run_late_detection_inference import prepare_and_run_inference
                    base_output = os.path.join(self.output_dir, "optuna_prob_maps.csv")
                    df = prepare_and_run_inference(self.dataset_csv, self.model_path, base_output)
                    self.log(f"✓ Computed probability maps for {len(df)} samples")
                except Exception as e:
                    self.log(f"Failed to compute probability maps: {e}")
                    self.finished_signal.emit(False, {"error": str(e)})
                    return

            prob_maps_dict, labels_dict = load_probability_maps_from_results(df, row_filter=1)
            if len(prob_maps_dict) == 0:
                msg = "No probability maps loaded for dev set"
                self.log(msg)
                self.finished_signal.emit(False, {"error": msg})
                return

            total = self.n_trials

            # Import optuna locally and handle missing dependency
            try:
                import optuna
            except Exception as e:
                self.log(f"Optuna is not installed or failed to import: {e}")
                self.finished_signal.emit(False, {"error": "optuna not installed"})
                return

            # Define callback to report progress after each trial
            def _optuna_callback(study, trial):
                # trial.number is zero-based
                cur = trial.number + 1
                self.progress_signal.emit(cur)

                # Get all metrics from user attributes
                acc = trial.user_attrs.get("acc", 0.0)
                prec = trial.user_attrs.get("prec", 0.0)
                rec = trial.user_attrs.get("rec", 0.0)
                f1 = trial.user_attrs.get("f1", 0.0)
                f2 = trial.user_attrs.get("f2", 0.0)
                auc = trial.user_attrs.get("auc", 0.0)

                # Format parameters for display
                params_list = []
                for key, value in trial.params.items():
                    if isinstance(value, float):
                        params_list.append(f"{key}={value:.4f}")
                    else:
                        params_list.append(f"{key}={value}")
                params_str = " | ".join(params_list)

                # Log all metrics and parameters
                metrics_str = f"ACC={acc:.4f} | PREC={prec:.4f} | REC={rec:.4f} | F1={f1:.4f} | F2={f2:.4f} | AUC={auc:.4f}"
                # Log trial number, parameters and metrics on a single line
                self.log(f"Trial {cur}/{total}: Params: {params_str} | Metrics: {metrics_str}")

                # Allow external stop
                if self.should_stop:
                    study.stop()

            # Objective function uses the same evaluation helper
            ### change parameters hare
            def objective(trial: optuna.Trial) -> float:
                # Suggest parameters
                prob_thr = trial.suggest_float("prob_thr", 0.96, 1.0)
                morph_size = trial.suggest_categorical("morph_size", [0] + list(range(1, 16, 1)))
                min_blob_size = trial.suggest_int("min_blob_size", 0, 1000)

                # Circularity filtering
                use_circ = trial.suggest_categorical("use_circularity", [False, True])
                circularity_min = None
                circularity_max = None
                if use_circ:
                    circularity_min = trial.suggest_float("circularity_min", 0.0, 0.5)
                    circularity_max = trial.suggest_float("circularity_max", max(0.6, circularity_min), 1.0)

                # Aspect ratio filtering
                use_ar = trial.suggest_categorical("use_aspect_ratio", [False, True])
                aspect_ratio_min = None
                aspect_ratio_limit = None
                if use_ar:
                    aspect_ratio_min = trial.suggest_float("aspect_ratio_min", 1.0, 6.0)
                    aspect_ratio_limit = trial.suggest_float("aspect_ratio_limit", max(3.0, aspect_ratio_min), 10.0)

                # Solidity filtering
                use_sol = trial.suggest_categorical("use_solidity", [False, True])
                solidity_min = None
                solidity_max = None
                if use_sol:
                    solidity_min = trial.suggest_float("solidity_min", 0.0, 0.5)
                    solidity_max = trial.suggest_float("solidity_max", max(0.6, solidity_min), 1.0)

                # Patch-level parameters
                patch_size = trial.suggest_categorical("patch_size", [32, 64, 128])
                patch_pixel_ratio = trial.suggest_float("patch_pixel_ratio", 0.01, 0.30)
                global_threshold = trial.suggest_float("global_threshold", 0.001, 0.1)

                # Evaluate
                try:
                    metrics = evaluate_single_combination(
                        prob_maps_dict,
                        labels_dict,
                        prob_thr=prob_thr,
                        min_blob_size=min_blob_size,
                        circularity_min=circularity_min,
                        circularity_max=circularity_max,
                        aspect_ratio_min=aspect_ratio_min,
                        aspect_ratio_limit=aspect_ratio_limit,
                        solidity_min=solidity_min,
                        solidity_max=solidity_max,
                        patch_size=int(patch_size),
                        patch_pixel_ratio=float(patch_pixel_ratio),
                        global_threshold=float(global_threshold),
                        morph_size=int(morph_size)
                    )
                except Exception as e:
                    self.log(f"Evaluation failed: {e}")
                    return 0.0

                # Extract all metrics
                acc = metrics.get("accuracy", 0.0)
                prec = metrics.get("precision", 0.0)
                rec = metrics.get("recall", 0.0)
                f1 = metrics.get("f1_score", 0.0)
                auc = metrics.get("roc_auc", 0.0)

                # Compute F2
                if (prec is None) or (rec is None) or (prec + rec == 0):
                    f2 = 0.0
                else:
                    beta2 = 4.0
                    f2 = (1 + beta2) * (prec * rec) / (beta2 * prec + rec)

                # Store metrics for callback to display
                trial.set_user_attr("acc", acc)
                trial.set_user_attr("prec", prec)
                trial.set_user_attr("rec", rec)
                trial.set_user_attr("f1", f1)
                trial.set_user_attr("f2", f2)
                trial.set_user_attr("auc", auc)

                return float(f2)

            # Create and run study
            study = optuna.create_study(direction="maximize")
            self._study = study

            # Run optimization (blocking in thread)
            study.optimize(objective, n_trials=self.n_trials, callbacks=[_optuna_callback])

            # When finished, emit best params/value with ALL metrics
            best = study.best_trial
            best_info = {
                "best_value": float(best.value) if best.value is not None else None,
                "best_params": dict(best.params)
            }

            # Extract all metrics from best trial
            acc = best.user_attrs.get("acc", 0.0)
            prec = best.user_attrs.get("prec", 0.0)
            rec = best.user_attrs.get("rec", 0.0)
            f1 = best.user_attrs.get("f1", 0.0)
            f2 = best.user_attrs.get("f2", 0.0)
            auc = best.user_attrs.get("auc", 0.0)

            # Format parameters for display
            self.log("=" * 80)
            self.log("🎉 OPTUNA OPTIMIZATION COMPLETE!")
            self.log("=" * 80)
            self.log("")
            self.log("📊 BEST METRICS:")
            self.log(f"  • F2 Score:    {f2:.4f} (optimized metric)")
            self.log(f"  • F1 Score:    {f1:.4f}")
            self.log(f"  • Accuracy:    {acc:.4f}")
            self.log(f"  • Precision:   {prec:.4f}")
            self.log(f"  • Recall:      {rec:.4f}")
            self.log(f"  • AUC:         {auc:.4f}")
            self.log("")
            self.log("⚙️  BEST PARAMETERS:")
            for key, value in sorted(best.params.items()):
                if isinstance(value, float):
                    self.log(f"  • {key:25s} = {value:.4f}")
                else:
                    self.log(f"  • {key:25s} = {value}")
            self.log("")
            self.log(f"✓ Optimization completed after {len(study.trials)} trials")
            self.log("=" * 80)

            self.finished_signal.emit(True, best_info)

        except Exception as e:
            self.log(f"OptunaWorker failed: {e}")
            self.finished_signal.emit(False, {"error": str(e)})


# ===== Main UI Class =====

class HSILateDetectionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Late Detection Viewer - 6 Panel Layout")
        self.setGeometry(100, 100, 2400, 800)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # State variables
        self.hsi_cube: Optional[np.ndarray] = None
        self.hdr_path: Optional[str] = None
        self.rgb_image: Optional[np.ndarray] = None
        self.reflectance_image: Optional[np.ndarray] = None  # REFLECTANCE_*.png for SAM visualization
        self.cnn_candidates_image: Optional[np.ndarray] = None  # Panel 5: CNN candidates with bboxes
        self.current_band: int = 0
        self.folder_path: str = ""

        # Navigation
        self.current_cluster_id: Optional[str] = None
        self.current_date: Optional[str] = None
        self.available_clusters: List[str] = []

        # Model
        self.lda_path: Optional[str] = None
        self.scaler_path: Optional[str] = None
        self.available_model_names: List[str] = list(AVAILABLE_MODELS.keys())
        self.available_model_paths: List[str] = list(AVAILABLE_MODELS.values())

        # Multi-class support
        self.current_pos_idx: int = 0
        self.model_classes: np.ndarray = np.array([])

        # Detection results cache
        self.last_detection_prob_map: Optional[np.ndarray] = None
        self.last_detection_mask: Optional[np.ndarray] = None
        self.last_detection_mask_unfiltered: Optional[np.ndarray] = None  # Before filters

        # SAM2 segmentation infrastructure
        self.sam2_segmenter = None
        self.last_sam_segments: List[np.ndarray] = []
        self.last_sam_overlay: Optional[np.ndarray] = None
        self.max_sam_blobs = 50  # Default max blobs to segment
        self.last_cluster_mask: Optional[np.ndarray] = None  # Cluster ROI mask for filtering
        self.use_cluster_filter = False  # Toggle for cluster mask filtering

        # CNN Classification (for grape/not-grape)
        self.cnn_model: Optional[torch.nn.Module] = None
        self.cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_transform: Optional[transforms.Compose] = None
        self.cnn_classifications: List[Dict] = []  # Store classification results
        self.show_popup_before_cnn = True  # Toggle for showing gallery before CNN

        # Dataset state
        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_current_index: int = -1

        # Runtime params
        self.cell_size = 64
        self.pix_thr = 0.96  # Default threshold
        self.best_global_threshold = 0.05  # Global threshold from grid search
        # Advanced filter params (UI-controlled)
        self.morph_size = 3
        self.solidity_max = 1.0
        self.aspect_ratio_max = 5.0

        # Binary model preprocessing
        self.apply_binary_preprocessing = True  # Auto-enabled for Binary: models
        self.cube_wavelengths: Optional[np.ndarray] = None  # Wavelength metadata from HDR

        # Grid search worker
        self.grid_search_worker: Optional[GridSearchWorker] = None
        self.optuna_worker: Optional[OptunaWorker] = None

        # Logging handler for UI
        self.log_handler: Optional[QTextEditLogger] = None

        self._build_ui()
        self._setup_logging()
        self._discover_models()
        self._auto_load_model()
        self._load_cnn_model()  # Load CNN model for classification

    def _setup_logging(self):
        """Setup custom logging handler to capture logs in UI."""
        if hasattr(self, 'log_text_edit'):
            self.log_handler = QTextEditLogger()
            self.log_handler.setFormatter(
                logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                                datefmt='%H:%M:%S')
            )
            self.log_handler.log_signal.connect(self._append_log)

            # Attach to root logger to capture all logs
            root_logger = logging.getLogger()
            root_logger.addHandler(self.log_handler)
            root_logger.setLevel(logging.INFO)

    def _append_log(self, message):
        """Append log message to the log text edit."""
        if hasattr(self, 'log_text_edit'):
            self.log_text_edit.appendPlainText(message)
            # Auto-scroll to bottom
            self.log_text_edit.moveCursor(QTextCursor.End)

    def _build_ui(self):
        """Build the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Top row: 5 image panels ---
        images_row = QHBoxLayout()
        layout.addLayout(images_row)

        # Panel 1: HSI Gray
        self.hsi_band_label = QLabel("1. HSI Gray")
        self.hsi_band_label.setFixedSize(300, 300)
        self.hsi_band_label.setAlignment(Qt.AlignCenter)
        self.hsi_band_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.hsi_band_label)

        # Panel 2: HSI Detection (Before Filters)
        self.hsi_detection_before_label = QLabel("2. Detection (Before Filters)")
        self.hsi_detection_before_label.setFixedSize(300, 300)
        self.hsi_detection_before_label.setAlignment(Qt.AlignCenter)
        self.hsi_detection_before_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.hsi_detection_before_label)

        # Panel 3: HSI Detection (After Filters)
        self.hsi_detection_after_label = QLabel("3. Detection (After Filters)")
        self.hsi_detection_after_label.setFixedSize(300, 300)
        self.hsi_detection_after_label.setAlignment(Qt.AlignCenter)
        self.hsi_detection_after_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.hsi_detection_after_label)

        # Panel 4: RGB with SAM Segments
        self.rgb_sam_label = QLabel("4. RGB + SAM Segments")
        self.rgb_sam_label.setFixedSize(300, 300)
        self.rgb_sam_label.setAlignment(Qt.AlignCenter)
        self.rgb_sam_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.rgb_sam_label)

        # Panel 5: CNN Candidates (BBox)
        self.cnn_candidates_label = QLabel("5. CNN Candidates (BBox)")
        self.cnn_candidates_label.setFixedSize(300, 300)
        self.cnn_candidates_label.setAlignment(Qt.AlignCenter)
        self.cnn_candidates_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.cnn_candidates_label)

        # Panel 6: RGB Image Only
        self.rgb_label = QLabel("6. RGB Image")
        self.rgb_label.setFixedSize(300, 300)
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.rgb_label)

        # --- Navigation row ---
        nav_row = QHBoxLayout()
        layout.addLayout(nav_row)

        load_btn = QPushButton("Load Images")
        load_btn.clicked.connect(self._choose_folder)
        nav_row.addWidget(load_btn)

        self.prev_btn = QPushButton("← Prev Cluster")
        self.prev_btn.clicked.connect(self._load_prev_cluster)
        nav_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next Cluster →")
        self.next_btn.clicked.connect(self._load_next_cluster)
        nav_row.addWidget(self.next_btn)

        self.cluster_label = QLabel("Cluster: N/A")
        self.cluster_label.setFixedWidth(150)
        nav_row.addWidget(self.cluster_label)

        nav_row.addStretch()

        # Model selector
        nav_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(300)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        nav_row.addWidget(self.model_combo)

        # Class selector
        nav_row.addWidget(QLabel("Detect Class:"))
        self.class_combo = QComboBox()
        self.class_combo.setFixedWidth(150)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        nav_row.addWidget(self.class_combo)

        # --- Controls row ---
        controls = QVBoxLayout()
        layout.addLayout(controls)

        # Top row: Band, Cell Size, Patch Thr, Prob Thr, Invert class
        top_row = QHBoxLayout()
        controls.addLayout(top_row)

        top_row.addWidget(QLabel("Band"))
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setValue(0)
        self.band_slider.valueChanged.connect(self._update_band)
        top_row.addWidget(self.band_slider)

        top_row.addWidget(QLabel("Cell Size"))
        self.cell_spin = QSpinBox()
        self.cell_spin.setRange(8, 256)
        self.cell_spin.setSingleStep(8)
        self.cell_spin.setValue(self.cell_size)
        self.cell_spin.valueChanged.connect(self._update_cell)
        top_row.addWidget(self.cell_spin)

        top_row.addWidget(QLabel("Patch Thr"))
        self.patch_thr_combo = QComboBox()
        self.patch_thr_combo.setFixedWidth(80)
        self.patch_thr_combo.setToolTip("Threshold for patch classification (% of cracked pixels in patch)")
        # Add options from 0% to 100% in 5% increments
        for i in range(0, 101, 5):
            self.patch_thr_combo.addItem(f"{i}%", i / 100.0)
        # Set default to 10%
        self.patch_thr_combo.setCurrentIndex(2)  # 10%
        self.patch_thr_value = 0.10
        self.patch_thr_combo.currentIndexChanged.connect(self._update_patch_thr)
        top_row.addWidget(self.patch_thr_combo)

        top_row.addWidget(QLabel("Prob Thr"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.05)
        self.thr_spin.setDecimals(5)
        self.thr_spin.setValue(self.pix_thr)
        self.thr_spin.valueChanged.connect(self._update_thr)
        top_row.addWidget(self.thr_spin)

        self.invert_class_chk = QCheckBox("Invert class")
        self.invert_class_chk.setChecked(False)
        top_row.addWidget(self.invert_class_chk)

        # Binary model preprocessing checkbox (SNV + wavelength filtering)
        self.binary_preprocessing_chk = QCheckBox("Binary Preprocessing (SNV)")
        self.binary_preprocessing_chk.setChecked(self.apply_binary_preprocessing)
        self.binary_preprocessing_chk.setToolTip(
            "Apply SNV normalization + wavelength filtering [450-925nm]\n"
            "Auto-enabled when selecting Binary: models"
        )
        self.binary_preprocessing_chk.stateChanged.connect(
            lambda state: setattr(self, 'apply_binary_preprocessing', state == Qt.Checked)
        )
        top_row.addWidget(self.binary_preprocessing_chk)

        # Filters grid
        filters_group = QGroupBox("Geometric Filters")
        filters_layout = QGridLayout(filters_group)
        controls.addWidget(filters_group)

        # Row 0: Filter types
        filters_layout.addWidget(QLabel("Filter Type"), 0, 0)
        filters_layout.addWidget(QLabel("Border"), 0, 1)
        filters_layout.addWidget(QLabel("Area"), 0, 2)
        filters_layout.addWidget(QLabel("Circularity"), 0, 3)
        filters_layout.addWidget(QLabel("Aspect Ratio"), 0, 4)
        filters_layout.addWidget(QLabel("Solidity"), 0, 5)

        # Row 1: Max labels
        filters_layout.addWidget(QLabel("Max"), 1, 0)
        filters_layout.addWidget(QLabel(""), 1, 1)  # Border has no max
        self.area_max_label = QLabel("Max")
        filters_layout.addWidget(self.area_max_label, 1, 2)
        self.circ_max_label = QLabel("Max")
        filters_layout.addWidget(self.circ_max_label, 1, 3)
        self.aspect_max_label = QLabel("Max")
        filters_layout.addWidget(self.aspect_max_label, 1, 4)
        self.solidity_max_label = QLabel("Max")
        filters_layout.addWidget(self.solidity_max_label, 1, 5)

        # Row 2: Max spinboxes
        filters_layout.addWidget(QLabel(""), 2, 0)  # Empty
        self.border_spin = QSpinBox()
        self.border_spin.setRange(0, 100)
        self.border_spin.setValue(20)
        self.border_spin.setFixedWidth(60)
        filters_layout.addWidget(self.border_spin, 2, 1)

        self.area_max_spin = QSpinBox()
        self.area_max_spin.setRange(1, 10000000)
        self.area_max_spin.setValue(5000)
        self.area_max_spin.setFixedWidth(70)
        filters_layout.addWidget(self.area_max_spin, 2, 2)

        self.circ_max_spin = QDoubleSpinBox()
        self.circ_max_spin.setRange(0.0, 1.0)
        self.circ_max_spin.setSingleStep(0.05)
        self.circ_max_spin.setDecimals(5)
        self.circ_max_spin.setValue(1.0)
        self.circ_max_spin.setFixedWidth(70)
        self.circ_max_spin.setToolTip("Maximum circularity (0.0-1.0)")
        filters_layout.addWidget(self.circ_max_spin, 2, 3)

        self.aspect_ratio_max_spin = QDoubleSpinBox()
        self.aspect_ratio_max_spin.setRange(1.0, 20.0)
        self.aspect_ratio_max_spin.setSingleStep(0.5)
        self.aspect_ratio_max_spin.setValue(self.aspect_ratio_max)
        self.aspect_ratio_max_spin.setFixedWidth(80)
        self.aspect_ratio_max_spin.setToolTip("Maximum aspect ratio (width/height or height/width)")
        filters_layout.addWidget(self.aspect_ratio_max_spin, 2, 4)

        self.solidity_max_spin = QDoubleSpinBox()
        self.solidity_max_spin.setRange(0.0, 1.0)
        self.solidity_max_spin.setSingleStep(0.05)
        self.solidity_max_spin.setDecimals(5)
        self.solidity_max_spin.setValue(self.solidity_max)
        self.solidity_max_spin.setFixedWidth(70)
        self.solidity_max_spin.setToolTip("Maximum solidity (Area / ConvexHullArea)")
        filters_layout.addWidget(self.solidity_max_spin, 2, 5)

        # Row 3: Min labels
        filters_layout.addWidget(QLabel("Min"), 3, 0)
        filters_layout.addWidget(QLabel(""), 3, 1)  # Border has no min
        self.area_min_label = QLabel("Min")
        filters_layout.addWidget(self.area_min_label, 3, 2)
        self.circ_min_label = QLabel("Min")
        filters_layout.addWidget(self.circ_min_label, 3, 3)
        self.aspect_min_label = QLabel("Min")
        filters_layout.addWidget(self.aspect_min_label, 3, 4)
        self.solidity_min_label = QLabel("Min")
        filters_layout.addWidget(self.solidity_min_label, 3, 5)

        # Row 4: Min spinboxes
        filters_layout.addWidget(QLabel(""), 4, 0)  # Empty
        filters_layout.addWidget(QLabel(""), 4, 1)  # Border has no min

        self.area_min_spin = QSpinBox()
        self.area_min_spin.setRange(0, 100000)
        self.area_min_spin.setValue(10)
        self.area_min_spin.setFixedWidth(70)
        filters_layout.addWidget(self.area_min_spin, 4, 2)

        self.circ_min_spin = QDoubleSpinBox()
        self.circ_min_spin.setRange(0.0, 1.0)
        self.circ_min_spin.setSingleStep(0.05)
        self.circ_min_spin.setDecimals(5)
        self.circ_min_spin.setValue(0.0)
        self.circ_min_spin.setFixedWidth(70)
        self.circ_min_spin.setToolTip("Minimum circularity (0.0-1.0)")
        filters_layout.addWidget(self.circ_min_spin, 4, 3)

        self.aspect_ratio_min_spin = QDoubleSpinBox()
        self.aspect_ratio_min_spin.setRange(1.0, 10.0)
        self.aspect_ratio_min_spin.setSingleStep(0.5)
        self.aspect_ratio_min_spin.setValue(1.0)
        self.aspect_ratio_min_spin.setFixedWidth(80)
        self.aspect_ratio_min_spin.setToolTip("Minimum aspect ratio (width/height or height/width)")
        filters_layout.addWidget(self.aspect_ratio_min_spin, 4, 4)

        self.solidity_min_spin = QDoubleSpinBox()
        self.solidity_min_spin.setRange(0.0, 1.0)
        self.solidity_min_spin.setSingleStep(0.05)
        self.solidity_min_spin.setDecimals(5)
        self.solidity_min_spin.setValue(0.0)
        self.solidity_min_spin.setFixedWidth(70)
        self.solidity_min_spin.setToolTip("Minimum solidity (Area / ConvexHullArea)")
        filters_layout.addWidget(self.solidity_min_spin, 4, 5)

        # Row 5: Checkboxes
        filters_layout.addWidget(QLabel("Enable"), 5, 0)
        self.use_border_chk = QCheckBox("Border")
        self.use_border_chk.setChecked(False)
        self.use_border_chk.setToolTip("Remove detections near image edges")
        filters_layout.addWidget(self.use_border_chk, 5, 1)

        self.use_area_chk = QCheckBox("Area")
        self.use_area_chk.setChecked(False)
        self.use_area_chk.setToolTip("Remove detections that are too small or too large")
        filters_layout.addWidget(self.use_area_chk, 5, 2)

        self.use_circ_chk = QCheckBox("Circularity")
        self.use_circ_chk.setChecked(False)
        self.use_circ_chk.setToolTip("Remove detections by circularity (1.0=circle, 0.0=line)")
        filters_layout.addWidget(self.use_circ_chk, 5, 3)

        self.use_ar_chk = QCheckBox("Aspect Ratio")
        self.use_ar_chk.setChecked(False)
        self.use_ar_chk.setToolTip("Remove detections by aspect ratio")
        filters_layout.addWidget(self.use_ar_chk, 5, 4)

        self.use_sol_chk = QCheckBox("Solidity")
        self.use_sol_chk.setChecked(False)
        self.use_sol_chk.setToolTip("Remove detections by solidity")
        filters_layout.addWidget(self.use_sol_chk, 5, 5)

        # Morph Size
        morph_row = QHBoxLayout()
        controls.addLayout(morph_row)
        morph_row.addWidget(QLabel("Morph Size"))
        self.morph_size_spin = QSpinBox()
        self.morph_size_spin.setRange(0, 20)
        self.morph_size_spin.setValue(self.morph_size)
        self.morph_size_spin.setFixedWidth(70)
        self.morph_size_spin.setToolTip("Morphological closing kernel size (0 = disabled)")
        self.morph_size_spin.valueChanged.connect(lambda v: setattr(self, 'morph_size', int(v)))
        morph_row.addWidget(self.morph_size_spin)
        morph_row.addStretch()

        # Cluster Filter checkbox
        cluster_filter_row = QHBoxLayout()
        controls.addLayout(cluster_filter_row)
        self.use_cluster_filter_chk = QCheckBox("Use Cluster Filter (SAM2 ROI)")
        self.use_cluster_filter_chk.setChecked(self.use_cluster_filter)
        self.use_cluster_filter_chk.setToolTip(
            "Use SAM2 to detect the main grape cluster and filter out background.\n"
            "This reduces false positives on tripods, walls, and text.\n"
            "Panel 4 will show the detected cluster boundary in green."
        )
        self.use_cluster_filter_chk.stateChanged.connect(
            lambda state: setattr(self, 'use_cluster_filter', state == Qt.Checked)
        )
        cluster_filter_row.addWidget(self.use_cluster_filter_chk)
        cluster_filter_row.addStretch()

        # --- Action buttons row ---
        actions_row = QHBoxLayout()
        layout.addLayout(actions_row)

        run_analysis_btn = QPushButton("Run Analysis")
        run_analysis_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        run_analysis_btn.clicked.connect(self._run_analysis)
        actions_row.addWidget(run_analysis_btn)

        # SAM Segmentation button - GREEN to stand out
        sam_btn = QPushButton("🎯 Show SAM Segments")
        sam_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #4CAF50; color: white;")
        sam_btn.setToolTip("Segment detected blobs with SAM on RGB image")
        sam_btn.clicked.connect(self._show_sam_segments)
        actions_row.addWidget(sam_btn)

        # Max SAM blobs spinbox
        actions_row.addWidget(QLabel("Max SAM blobs:"))
        self.max_sam_blobs_spin = QSpinBox()
        self.max_sam_blobs_spin.setRange(1, 500)
        self.max_sam_blobs_spin.setValue(self.max_sam_blobs)
        self.max_sam_blobs_spin.setFixedWidth(80)
        self.max_sam_blobs_spin.setToolTip("Maximum number of blobs to segment with SAM (1-500)")
        self.max_sam_blobs_spin.valueChanged.connect(lambda v: setattr(self, 'max_sam_blobs', int(v)))
        actions_row.addWidget(self.max_sam_blobs_spin)

        # Run CNN Debug button - BLUE to stand out
        self.run_cnn_btn = QPushButton("🔍 Run CNN Debug")
        self.run_cnn_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #2196F3; color: white;")
        self.run_cnn_btn.setToolTip("Crop detections and save to debug_crops folder")
        self.run_cnn_btn.clicked.connect(self._debug_cnn_crops)
        actions_row.addWidget(self.run_cnn_btn)

        # Crop mode toggle - Segment only vs Full BBox
        self.crop_segment_only_chk = QCheckBox("Segment Only (Masked)")
        self.crop_segment_only_chk.setChecked(False)  # Default: Full BBox
        self.crop_segment_only_chk.setToolTip(
            "When checked: Show only the segment pixels (masked region)\n"
            "When unchecked: Show full bounding box region"
        )
        actions_row.addWidget(self.crop_segment_only_chk)

        # Show popup before CNN toggle
        self.show_popup_before_cnn_chk = QCheckBox("Show Popup Before CNN")
        self.show_popup_before_cnn_chk.setChecked(True)  # Default: Show popup
        self.show_popup_before_cnn_chk.setToolTip(
            "When checked: Show crop gallery before running CNN classification\n"
            "When unchecked: Run CNN directly without popup"
        )
        self.show_popup_before_cnn_chk.stateChanged.connect(
            lambda state: setattr(self, 'show_popup_before_cnn', state == Qt.Checked)
        )
        actions_row.addWidget(self.show_popup_before_cnn_chk)

        screenshot_btn = QPushButton("Screenshot (All 6 Panels)")
        screenshot_btn.clicked.connect(self._save_screenshot)
        actions_row.addWidget(screenshot_btn)

        actions_row.addStretch()

        # --- Dataset panel ---
        dataset_box = QGroupBox("Dataset Navigator")
        dataset_layout = QVBoxLayout(dataset_box)

        # CSV path row
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Dataset CSV:"))
        self.dataset_csv_edit = QLineEdit()
        self.dataset_csv_edit.setText(DEFAULT_DATASET_CSV)
        self.dataset_csv_edit.setReadOnly(True)
        csv_row.addWidget(self.dataset_csv_edit)

        browse_csv_btn = QPushButton("Browse...")
        browse_csv_btn.clicked.connect(self._browse_dataset_csv)
        csv_row.addWidget(browse_csv_btn)

        load_dataset_btn = QPushButton("Load Dataset")
        load_dataset_btn.setStyleSheet("font-weight: bold;")
        load_dataset_btn.clicked.connect(self._load_dataset)
        csv_row.addWidget(load_dataset_btn)

        # Load Best Grid Result button
        load_best_btn = QPushButton("Load Best Grid Result")
        load_best_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        load_best_btn.setToolTip("Load optimal hyperparameters from test_set_blob_patch_results.csv")
        load_best_btn.clicked.connect(self._load_best_grid_result)
        csv_row.addWidget(load_best_btn)

        dataset_layout.addLayout(csv_row)

        # Navigation row
        nav_dataset_row = QHBoxLayout()
        self.dataset_prev_btn = QPushButton("← Prev Sample")
        self.dataset_prev_btn.clicked.connect(self._dataset_prev)
        self.dataset_prev_btn.setEnabled(False)
        nav_dataset_row.addWidget(self.dataset_prev_btn)

        self.dataset_next_btn = QPushButton("Next Sample →")
        self.dataset_next_btn.clicked.connect(self._dataset_next)
        self.dataset_next_btn.setEnabled(False)
        nav_dataset_row.addWidget(self.dataset_next_btn)

        self.dataset_info_label = QLabel("No dataset loaded")
        nav_dataset_row.addWidget(self.dataset_info_label)
        nav_dataset_row.addStretch()
        dataset_layout.addLayout(nav_dataset_row)

        # Sample table
        self.dataset_table = QTableWidget(0, 4)
        self.dataset_table.setHorizontalHeaderLabels(["grape_id", "row", "label", "image_path"])
        self.dataset_table.setMaximumHeight(150)
        self.dataset_table.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        dataset_layout.addWidget(self.dataset_table)

        layout.addWidget(dataset_box)

        # --- Train & Optimize Section ---
        train_box = QGroupBox("Train & Optimize")
        train_layout = QVBoxLayout(train_box)

        # Grid search configuration row
        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Dataset CSV:"))

        self.grid_dataset_edit = QLineEdit()
        self.grid_dataset_edit.setText(DEFAULT_DATASET_CSV)
        config_row.addWidget(self.grid_dataset_edit)

        browse_grid_btn = QPushButton("Browse...")
        browse_grid_btn.clicked.connect(lambda: self._browse_file(self.grid_dataset_edit, "Dataset CSV"))
        config_row.addWidget(browse_grid_btn)

        config_row.addWidget(QLabel("Model:"))
        self.grid_model_combo = QComboBox()
        for name in self.available_model_names:
            self.grid_model_combo.addItem(name)
        config_row.addWidget(self.grid_model_combo)

        train_layout.addLayout(config_row)

        # Control row
        control_row = QHBoxLayout()

        # Number of Optuna trials
        control_row.addWidget(QLabel("Trials"))
        self.optuna_trials_spin = QSpinBox()
        self.optuna_trials_spin.setRange(10, 1000)
        self.optuna_trials_spin.setValue(100)
        self.optuna_trials_spin.setFixedWidth(90)
        control_row.addWidget(self.optuna_trials_spin)

        self.start_grid_btn = QPushButton("🚀 Start Optuna Optimization")
        self.start_grid_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; font-size: 14px; padding: 10px;"
        )
        self.start_grid_btn.setToolTip("Run Optuna hyperparameter optimization")
        self.start_grid_btn.clicked.connect(self._start_optuna_search)
        control_row.addWidget(self.start_grid_btn)

        self.stop_grid_btn = QPushButton("⏹ Stop")
        self.stop_grid_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 10px;"
        )
        self.stop_grid_btn.setEnabled(False)
        self.stop_grid_btn.clicked.connect(self._stop_grid_search)
        control_row.addWidget(self.stop_grid_btn)

        control_row.addStretch()
        train_layout.addLayout(control_row)

        # Progress bar
        self.grid_progress = QProgressBar()
        self.grid_progress.setVisible(False)
        train_layout.addWidget(self.grid_progress)

        # Log viewer
        log_label = QLabel("Pipeline Logs (Real-time):")
        log_label.setStyleSheet("font-weight: bold;")
        train_layout.addWidget(log_label)

        self.log_text_edit = QPlainTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMaximumBlockCount(10000)  # Limit log size
        self.log_text_edit.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; font-family: 'Consolas', 'Courier New', monospace;"
        )
        train_layout.addWidget(self.log_text_edit)

        layout.addWidget(train_box)

        self.status_bar.showMessage("Load a folder and click 'Run Analysis' to start.")

    def _discover_models(self):
        """Populate model dropdown."""
        self.model_combo.clear()
        for model_name in self.available_model_names:
            self.model_combo.addItem(model_name)

        if self.available_model_paths:
            self.lda_path = self.available_model_paths[0]
            self.model_combo.setCurrentIndex(0)

        logger.info("Loaded %d models into dropdown", len(self.available_model_names))

    def _auto_load_model(self):
        """Auto-load the first model."""
        if self.lda_path:
            self.scaler_path = find_scaler(os.path.dirname(self.lda_path),
                                          [os.path.basename(self.lda_path)])

    def _on_model_changed(self, index: int):
        """Handle model selection change."""
        if 0 <= index < len(self.available_model_paths):
            self.lda_path = self.available_model_paths[index]
            self.scaler_path = find_scaler(os.path.dirname(self.lda_path),
                                          [os.path.basename(self.lda_path)])

            # Auto-enable binary preprocessing for Binary: models
            model_name = self.available_model_names[index]
            is_binary_model = model_name.startswith("Binary:")
            self.apply_binary_preprocessing = is_binary_model
            if hasattr(self, 'binary_preprocessing_chk'):
                self.binary_preprocessing_chk.setChecked(is_binary_model)
            if is_binary_model:
                logger.info("Binary model selected - auto-enabling SNV + wavelength preprocessing")

            try:
                _, _, pos_idx, classes, optimal_threshold = load_model_and_scaler(self.lda_path, self.scaler_path)
                self.model_classes = classes
                self.current_pos_idx = pos_idx

                # Populate class selector
                self.class_combo.blockSignals(True)
                self.class_combo.clear()
                for i, cls in enumerate(classes):
                    self.class_combo.addItem(str(cls))
                self.class_combo.setCurrentIndex(pos_idx)
                self.class_combo.blockSignals(False)

                # Update threshold spinner if model has optimal_threshold
                if optimal_threshold is not None:
                    self.thr_spin.setValue(optimal_threshold)
                    logger.info("Loaded model optimized threshold: %.4f", optimal_threshold)

                logger.info("Model changed: %s (%d classes)",
                           self.available_model_names[index], len(classes))
            except Exception as e:
                logger.warning("Failed to load model classes: %s", e)
                self.model_classes = np.array([])
                self.current_pos_idx = 0

    def _on_class_changed(self, index: int):
        """Handle target class selection change."""
        if 0 <= index < len(self.model_classes):
            self.current_pos_idx = index
            logger.info("Target class changed to: %s (index %d)",
                       self.model_classes[index], index)

    def _discover_available_clusters(self):
        """Find all available clusters in the raw data folder."""
        try:
            folders = [f for f in os.listdir(DEFAULT_SEARCH_FOLDER)
                      if os.path.isdir(os.path.join(DEFAULT_SEARCH_FOLDER, f))]
            import re
            cluster_pattern = re.compile(r'^\d+_\d+$')
            self.available_clusters = sorted([f for f in folders if cluster_pattern.match(f)])
            logger.info("Found %d clusters", len(self.available_clusters))
        except Exception as e:
            logger.warning("Failed to discover clusters: %s", e)
            self.available_clusters = []

    def _load_prev_cluster(self):
        """Load the previous cluster with the same date."""
        if not self.current_cluster_id or not self.current_date:
            self.status_bar.showMessage("Load a folder first to enable navigation.")
            return

        self._discover_available_clusters()
        if not self.available_clusters:
            self.status_bar.showMessage("No clusters available.")
            return

        try:
            current_idx = self.available_clusters.index(self.current_cluster_id)
            if current_idx > 0:
                prev_cluster = self.available_clusters[current_idx - 1]
                prev_folder = os.path.join(DEFAULT_SEARCH_FOLDER, prev_cluster, self.current_date)
                if os.path.exists(prev_folder):
                    self._load_images(prev_folder)
                    # Automatically run analysis after loading
                    self._run_analysis()
                else:
                    self.status_bar.showMessage(f"Date {self.current_date} not found in {prev_cluster}")
            else:
                self.status_bar.showMessage("Already at first cluster.")
        except Exception as e:
            logger.exception("Failed to load previous cluster: %s", e)

    def _load_next_cluster(self):
        """Load the next cluster with the same date."""
        if not self.current_cluster_id or not self.current_date:
            self.status_bar.showMessage("Load a folder first to enable navigation.")
            return

        self._discover_available_clusters()
        if not self.available_clusters:
            self.status_bar.showMessage("No clusters available.")
            return

        try:
            current_idx = self.available_clusters.index(self.current_cluster_id)
            if current_idx < len(self.available_clusters) - 1:
                next_cluster = self.available_clusters[current_idx + 1]
                next_folder = os.path.join(DEFAULT_SEARCH_FOLDER, next_cluster, self.current_date)
                if os.path.exists(next_folder):
                    self._load_images(next_folder)
                    # Automatically run analysis after loading
                    self._run_analysis()
                else:
                    self.status_bar.showMessage(f"Date {self.current_date} not found in {next_cluster}")
            else:
                self.status_bar.showMessage("Already at last cluster.")
        except Exception as e:
            logger.exception("Failed to load next cluster: %s", e)

    def _choose_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", DEFAULT_SEARCH_FOLDER)
        if folder:
            self._load_images(folder)

    def _load_images(self, folder: str):
        """Load all images from the selected folder."""
        try:
            # Extract cluster_id and date
            parts = folder.replace('\\', '/').split('/')
            if len(parts) >= 2:
                self.current_date = parts[-1]
                self.current_cluster_id = parts[-2]
                self.cluster_label.setText(f"Cluster: {self.current_cluster_id}")
                logger.info("Loaded cluster: %s, date: %s", self.current_cluster_id, self.current_date)

            # Load HSI cube
            hs = os.path.join(folder, "HS")
            res = os.path.join(hs, "results")
            hdr_files = [f for f in os.listdir(res) if f.lower().endswith(".hdr")]
            if not hdr_files:
                raise FileNotFoundError("No .hdr file in HS/results")

            self.hdr_path = os.path.join(res, hdr_files[0])
            self.hsi_cube = load_cube(self.hdr_path)

            # Extract wavelengths for binary model preprocessing
            self.cube_wavelengths = get_wavelengths_from_hdr(self.hdr_path)
            self.band_slider.setMaximum(self.hsi_cube.shape[2] - 1)

            # Set default band to 138
            self.current_band = min(138, self.hsi_cube.shape[2] - 1)
            self.band_slider.setValue(self.current_band)

            # Display initial HSI band
            self._update_band()

            # Load RGB image (Canon or any available RGB)
            rgb_path = self._find_rgb_image(folder)
            if rgb_path:
                bgr = cv2.imread(rgb_path)
                if bgr is not None:
                    self.rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # Display RGB as-is (no rotation needed)
                    self._show_image(self.rgb_image, self.rgb_label)
                    logger.info("Loaded RGB image: %s", rgb_path)

            # Load REFLECTANCE_*.png image from HS/results for SAM visualization (Panel 4)
            reflectance_path = self._find_reflectance_image(folder)
            if reflectance_path:
                bgr_refl = cv2.imread(reflectance_path)
                if bgr_refl is not None:
                    self.reflectance_image = cv2.cvtColor(bgr_refl, cv2.COLOR_BGR2RGB)
                    logger.info("Loaded REFLECTANCE image for SAM: %s", reflectance_path)
                else:
                    logger.warning("Failed to read REFLECTANCE image: %s", reflectance_path)
                    self.reflectance_image = None
            else:
                self.reflectance_image = None

            self.status_bar.showMessage(f"Loaded: {self.current_cluster_id} - {self.current_date}")

        except Exception as e:
            logger.exception("Failed to load images: %s", e)
            QMessageBox.critical(self, "Error", f"Failed to load images: {e}")

    def _find_rgb_image(self, folder: str) -> Optional[str]:
        """Find RGB/Canon image in folder."""
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if any(k in f.lower() for k in ("canon", "rgb")):
                        return os.path.join(root, f)
        # Fallback: any image
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(root, f)
        return None

    def _find_reflectance_image(self, folder: str) -> Optional[str]:
        """Find REFLECTANCE_*.png image in HS/results folder for SAM visualization."""
        try:
            hs_results = os.path.join(folder, "HS", "results")
            if not os.path.isdir(hs_results):
                logger.warning(f"HS/results folder not found: {hs_results}")
                return None

            for f in os.listdir(hs_results):
                if f.upper().startswith("REFLECTANCE_") and f.lower().endswith('.png'):
                    reflectance_path = os.path.join(hs_results, f)
                    logger.info(f"Found REFLECTANCE image: {reflectance_path}")
                    return reflectance_path

            logger.warning(f"No REFLECTANCE_*.png found in {hs_results}")
            return None
        except Exception as e:
            logger.warning(f"Failed to find REFLECTANCE image: {e}")
            return None

    def _update_band(self):
        """Update HSI band display."""
        if self.hsi_cube is None:
            return
        self.current_band = self.band_slider.value()
        band = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
        band = cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE)
        self._show_image(band, self.hsi_band_label, is_grayscale=True)
        self.status_bar.showMessage(f"Band {self.current_band}")

    def _update_cell(self, v: int):
        """Update cell size."""
        self.cell_size = int(v)
        logger.info("Cell size changed to %d", self.cell_size)

    def _update_patch_thr(self, index: int):
        """Update patch threshold from dropdown."""
        self.patch_thr_value = self.patch_thr_combo.itemData(index)
        logger.info("Patch threshold changed to %.2f (%d%%)",
                   self.patch_thr_value, int(self.patch_thr_value * 100))

    def _update_thr(self, v: float):
        """Update probability threshold."""
        self.pix_thr = float(v)
        logger.info("Probability threshold changed to %.2f", self.pix_thr)

    def _get_cluster_mask(self) -> Optional[np.ndarray]:
        """
        Generate a cluster ROI mask using SAM2 with center-point prompt.

        This detects the main grape cluster to filter out background objects
        (tripods, walls, text) that cause false positives.

        Returns:
            Binary mask (bool) matching HSI dimensions, or None if detection fails.
        """
        try:
            # Check if RGB image is available
            if self.rgb_image is None:
                logger.warning("Cluster mask: No RGB image available")
                return None

            # Initialize SAM2 if not already done
            if self.sam2_segmenter is None:
                logger.info("Cluster mask: Initializing SAM2 predictor...")
                initial_settings()
                predictor = initialize_sam2_predictor()
                self.sam2_segmenter = create_point_segmenter(predictor)
                logger.info("Cluster mask: SAM2 initialized")

            # Get HSI dimensions (target size)
            if self.hsi_cube is None:
                logger.warning("Cluster mask: No HSI cube loaded")
                return None

            hsi_h, hsi_w = self.hsi_cube.shape[:2]

            # Resize RGB to match HSI dimensions for alignment
            rgb_resized = cv2.resize(self.rgb_image, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)

            # Define prompt point at image center (where grape cluster typically is)
            h, w = rgb_resized.shape[:2]
            center_x, center_y = w // 2, h // 2
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])  # 1 = foreground point

            logger.info(f"Cluster mask: Using center point ({center_x}, {center_y}) on {w}x{h} image")

            # Use SAM2 predictor with multimask_output to get multiple candidates
            masks, scores, logits = self.sam2_segmenter.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )

            # Select mask with highest IoU score (best object match)
            best_idx = np.argmax(scores)
            cluster_mask = masks[best_idx]

            logger.info(f"Cluster mask: Selected mask {best_idx} with score {scores[best_idx]:.4f}")
            logger.info(f"Cluster mask: Coverage {cluster_mask.sum() / cluster_mask.size * 100:.1f}% of image")

            # Cache for visualization
            self.last_cluster_mask = cluster_mask.astype(bool)

            return cluster_mask.astype(bool)

        except Exception as e:
            logger.exception(f"Cluster mask generation failed: {e}")
            return None

    def _run_analysis(self):
        """Run full analysis and update all 4 panels."""
        if self.hsi_cube is None:
            self.status_bar.showMessage("Load an HSI folder first.")
            return
        if not self.lda_path:
            self.status_bar.showMessage("No model configured.")
            return

        try:
            t0 = time.perf_counter()
            logger.info("Running analysis: band=%d, cell=%d, thr=%.2f",
                       self.current_band, self.cell_size, self.pix_thr)

            # Load model
            lda, scaler, _, classes, _ = load_model_and_scaler(self.lda_path, self.scaler_path)
            pos_idx = self.current_pos_idx

            # Compute probability map (with binary preprocessing if enabled)
            prob_map = per_pixel_probs(
                self.hsi_cube, lda, scaler, pos_idx,
                apply_binary_preprocessing=self.apply_binary_preprocessing,
                wavelengths=self.cube_wavelengths
            )

            # Invert if requested
            if self.invert_class_chk.isChecked():
                logger.info("Inverting probabilities")
                prob_map = 1.0 - prob_map

            # =============================================================================
            # STRICT PIPELINE: 4-Step Processing for Solid Blob Visualization
            # =============================================================================
            # Pipeline order (important):
            #   STEP A: Threshold -> create initial binary mask from prob_map
            #   STEP B: Geometric filters (filter_blobs_advanced) -> remove small/noisy blobs
            #   STEP B.5: Optional Cluster ROI filter (SAM2) -> remove background false positives
            #   STEP C: Morphological Closing (CLOSE: dilation followed by erosion) -> merge valid fragments
            #   STEP D: Update state and UI caches
            # Note: Morphological Closing is applied AFTER geometric filtering so we only merge
            #       already-validated fragments (avoids merging noise into large blobs).

            # Cache unfiltered detection mask for Panel 2 (BEFORE any processing)
            self.last_detection_mask_unfiltered = (prob_map >= self.pix_thr)
            logger.info("Pipeline START: Original detection has %d pixels", self.last_detection_mask_unfiltered.sum())

            # -------------------------------------------------------------------------
            # STEP A: Initial Thresholding (Create binary mask from probabilities)
            # -------------------------------------------------------------------------
            initial_binary_mask = (prob_map >= self.pix_thr).astype(bool)
            logger.info("STEP A (Thresholding): Initial mask created with %d pixels", initial_binary_mask.sum())

            # -------------------------------------------------------------------------
            # STEP B: Geometric Filtering (CRUCIAL - Remove noise BEFORE morphology)
            # -------------------------------------------------------------------------
            use_border = self.use_border_chk.isChecked()
            use_area = self.use_area_chk.isChecked()
            use_circ = self.use_circ_chk.isChecked()
            use_ar = self.use_ar_chk.isChecked()
            use_sol = self.use_sol_chk.isChecked()

            filtered_prob_map = prob_map.copy()  # Preserve original for later

            if use_border or use_area or use_circ or use_ar or use_sol:
                filter_params = dict(
                    morph_size=0,  # NO morphology in filter step - done separately in STEP C
                    border_r=self.border_spin.value(),
                    area_min=self.area_min_spin.value(),
                    area_max=self.area_max_spin.value(),
                    circularity_min=(self.circ_min_spin.value() if use_circ else None),
                    circularity_max=(self.circ_max_spin.value() if use_circ else None),
                    aspect_ratio_min=(self.aspect_ratio_min_spin.value() if use_ar else None),
                    aspect_ratio_max=(self.aspect_ratio_max_spin.value() if use_ar else None),
                    solidity_min=(self.solidity_min_spin.value() if use_sol else None),
                    solidity_max=(self.solidity_max_spin.value() if use_sol else None),
                    use_border=use_border,
                    use_area=use_area,
                    use_circularity=use_circ,
                    use_aspect_ratio=use_ar,
                    use_solidity=use_sol,
                )
                logger.info("STEP B (Geometric Filter): Applying filters to remove noise: %s", filter_params)

                # filter_blobs_advanced returns a BOOLEAN mask of valid pixels
                geometric_filter_mask = filter_blobs_advanced(filtered_prob_map, self.pix_thr, **filter_params)

                # Apply filter: Keep only probability values where geometric filter passed
                filtered_prob_map = filtered_prob_map * geometric_filter_mask

                filtered_pixel_count = (filtered_prob_map >= self.pix_thr).sum()
                logger.info("STEP B (Geometric Filter): Complete. Kept %d pixels (removed noise)", filtered_pixel_count)
            else:
                logger.info("STEP B (Geometric Filter): SKIPPED (no filters enabled)")

            # -------------------------------------------------------------------------
            # STEP B.5: Cluster Mask ROI Filter (Remove background false positives)
            # -------------------------------------------------------------------------
            if self.use_cluster_filter:
                logger.info("STEP B.5 (Cluster Filter): Detecting main cluster with SAM2...")
                cluster_mask = self._get_cluster_mask()

                if cluster_mask is not None:
                    # Apply cluster mask: Keep only detections inside the cluster ROI
                    before_cluster_filter = (filtered_prob_map >= self.pix_thr).sum()
                    filtered_prob_map = filtered_prob_map * cluster_mask.astype(float)
                    after_cluster_filter = (filtered_prob_map >= self.pix_thr).sum()

                    removed_pixels = before_cluster_filter - after_cluster_filter
                    logger.info(f"STEP B.5 (Cluster Filter): Complete. Kept {after_cluster_filter} pixels "
                               f"(removed {removed_pixels} background pixels)")
                else:
                    logger.warning("STEP B.5 (Cluster Filter): Failed to generate cluster mask, skipping ROI filter")
            else:
                logger.info("STEP B.5 (Cluster Filter): SKIPPED (not enabled)")

            # -------------------------------------------------------------------------
            # STEP C: Morphological Closing (MERGE VALID FRAGMENTS)
            # -------------------------------------------------------------------------
            # Apply CLOSING to FILTERED data only (noise already removed). Closing is
            # a dilation followed by erosion which fills small holes and connects nearby
            # fragments without introducing as much overgrowth as pure dilation.

            mask_filtered = (filtered_prob_map >= self.pix_thr).astype(bool)

            if self.morph_size > 0:
                logger.info("STEP C (Morphological Closing): Applying closing (kernel=%d)", self.morph_size)

                # Create uint8 mask required by cv2.morphologyEx
                filtered_binary_mask = (mask_filtered.astype('uint8') * 255)

                # Elliptical structuring element for isotropic closing
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size))

                # Apply morphological CLOSE (dilation followed by erosion)
                closed_binary = cv2.morphologyEx(filtered_binary_mask, cv2.MORPH_CLOSE, kernel)

                # Convert back to boolean mask
                mask_final = (closed_binary > 0).astype(bool)

                closed_pixel_count = mask_final.sum()
                logger.info("STEP C (Morphological Closing): Complete. Closed to %d pixels (merged blobs)", closed_pixel_count)
            else:
                mask_final = mask_filtered
                logger.info("STEP C (Morphological Closing): SKIPPED (morph_size = 0)")

            # -------------------------------------------------------------------------
            # STEP D: Update State (Assign closed mask and update prob_map)
            # -------------------------------------------------------------------------
            final_prob_map = np.where(mask_final, np.maximum(filtered_prob_map, 0.5), 0.0)

            # Cache final results for UI display
            self.last_detection_prob_map = final_prob_map
            self.last_detection_mask = mask_final  # Use closed mask directly

            final_pixel_count = self.last_detection_mask.sum()
            logger.info("STEP D (Update State): Pipeline COMPLETE. Final mask: %d pixels for Panel 3",
                       final_pixel_count)
            logger.info("Pipeline END: Transformation: %d → %d → %d pixels (Initial → Filtered → Closed)",
                       self.last_detection_mask_unfiltered.sum(),
                       (filtered_prob_map >= self.pix_thr).sum(),
                       final_pixel_count)

            # Get base band for overlays
            band_img = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)

            # Panel 1: HSI Gray - display grayscale HSI band
            band_display = cv2.rotate(band_img, cv2.ROTATE_90_CLOCKWISE)
            self._show_image(band_display, self.hsi_band_label, is_grayscale=True)

            # Panel 2: HSI Detection (BEFORE filters) - show unfiltered detection
            detection_before_overlay = colorize_binary_mask(band_img, self.last_detection_mask_unfiltered)
            detection_before_overlay = cv2.rotate(detection_before_overlay, cv2.ROTATE_90_CLOCKWISE)
            detection_before_rgb = cv2.cvtColor(detection_before_overlay, cv2.COLOR_BGR2RGB)
            self._show_image(detection_before_rgb, self.hsi_detection_before_label)

            # Panel 3: HSI Detection (AFTER filters) - show filtered detection WITH GRID OVERLAY
            # First, calculate grid stats for the filtered mask
            grid_stats = analyze_grid(final_prob_map, self.cell_size, self.pix_thr)

            # Create grid overlay on the band image to visualize detection areas
            detection_after_overlay = overlay_on_band(band_img, grid_stats, alpha=0.35)

            # Also add the detection mask on top (yellow overlay)
            yellow_color = np.array([0, 215, 255], dtype=np.uint8)  # Yellow in BGR
            alpha_mask = 0.4
            detection_after_overlay[self.last_detection_mask] = (
                alpha_mask * yellow_color + (1 - alpha_mask) * detection_after_overlay[self.last_detection_mask]
            ).astype(np.uint8)

            detection_after_overlay = cv2.rotate(detection_after_overlay, cv2.ROTATE_90_CLOCKWISE)
            detection_after_rgb = cv2.cvtColor(detection_after_overlay, cv2.COLOR_BGR2RGB)
            self._show_image(detection_after_rgb, self.hsi_detection_after_label)

            # Panel 4: REFLECTANCE image with optional Cluster Mask (clean, no LDA detections)
            # Use REFLECTANCE_*.png from HS/results for SAM visualization
            if hasattr(self, 'reflectance_image') and self.reflectance_image is not None:
                # Resize REFLECTANCE to match HSI dimensions for proper alignment
                hsi_h, hsi_w = self.last_detection_mask.shape
                refl_resized = cv2.resize(self.reflectance_image, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)

                # Start with REFLECTANCE as base (clean image, no LDA overlays)
                panel4_viz = refl_resized.copy()

                # Rotate cluster mask to align with the upright REFLECTANCE image (if enabled)
                # HSI panels are shown rotated (90° CW), so masks are in HSI orientation.
                # For Panel 4 we need mask rotated 90° CW to match the upright REFLECTANCE orientation.
                cluster_mask_for_refl = None
                if self.last_cluster_mask is not None:
                    cluster_mask_for_refl = cv2.rotate(self.last_cluster_mask.astype('uint8'), cv2.ROTATE_90_CLOCKWISE).astype(bool)

                # If cluster filter is enabled and cluster mask exists, show it (rotated)
                if self.use_cluster_filter and cluster_mask_for_refl is not None:
                    cluster_overlay = panel4_viz.copy()
                    green_color = np.array([0, 255, 0], dtype=np.uint8)  # Green in RGB
                    cluster_overlay[cluster_mask_for_refl] = green_color
                    panel4_viz = cv2.addWeighted(panel4_viz, 0.7, cluster_overlay, 0.3, 0)
                    logger.info("Panel 4: Added cluster mask visualization (green overlay) on REFLECTANCE image")

                # NO LDA detected pixels overlay - Panel 4 is clean for SAM visualization only
                # DO NOT rotate the final REFLECTANCE viz. It should stay upright to match source image.
                self._show_image(panel4_viz, self.rgb_sam_label)
            elif hasattr(self, 'rgb_image') and self.rgb_image is not None:
                # Fallback: if REFLECTANCE not available, use RGB image
                hsi_h, hsi_w = self.last_detection_mask.shape
                rgb_resized = cv2.resize(self.rgb_image, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)
                panel4_viz = rgb_resized.copy()

                # Show cluster mask only (no LDA detections)
                cluster_mask_for_rgb = None
                if self.last_cluster_mask is not None:
                    cluster_mask_for_rgb = cv2.rotate(self.last_cluster_mask.astype('uint8'), cv2.ROTATE_90_CLOCKWISE).astype(bool)

                if self.use_cluster_filter and cluster_mask_for_rgb is not None:
                    cluster_overlay = panel4_viz.copy()
                    green_color = np.array([0, 255, 0], dtype=np.uint8)
                    cluster_overlay[cluster_mask_for_rgb] = green_color
                    panel4_viz = cv2.addWeighted(panel4_viz, 0.7, cluster_overlay, 0.3, 0)

                self._show_image(panel4_viz, self.rgb_sam_label)
                logger.warning("Panel 4: Using RGB fallback (REFLECTANCE image not found)")

            # Panel 5: CNN Candidates - show CNN candidates with bboxes (or placeholder)
            # Use SAME image source as Panel 4 (REFLECTANCE or RGB fallback)
            if hasattr(self, 'cnn_candidates_image') and self.cnn_candidates_image is not None:
                # If CNN debug has been run, show the result with bboxes
                self._show_image(self.cnn_candidates_image, self.cnn_candidates_label)
            else:
                # Placeholder: show SAME image as Panel 4 (REFLECTANCE or RGB)
                if hasattr(self, 'reflectance_image') and self.reflectance_image is not None:
                    # Use REFLECTANCE image (same as Panel 4)
                    hsi_h, hsi_w = self.last_detection_mask.shape
                    refl_resized = cv2.resize(self.reflectance_image, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)
                    self._show_image(refl_resized, self.cnn_candidates_label)
                    logger.info("Panel 5: Showing REFLECTANCE image (placeholder until CNN debug)")
                elif hasattr(self, 'rgb_image') and self.rgb_image is not None:
                    # Fallback to RGB if REFLECTANCE not available
                    hsi_h, hsi_w = self.last_detection_mask.shape
                    rgb_resized = cv2.resize(self.rgb_image, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)
                    self._show_image(rgb_resized, self.cnn_candidates_label)
                    logger.info("Panel 5: Showing RGB image (placeholder until CNN debug)")

            # Panel 6: RGB Image Only - show plain RGB image
            if hasattr(self, 'rgb_image') and self.rgb_image is not None:
                self._show_image(self.rgb_image, self.rgb_label)

            # grid_stats already calculated earlier for Panel 3 visualization

            elapsed = time.perf_counter() - t0
            blob_count = int(self.last_detection_mask.sum())
            class_name = classes[pos_idx] if pos_idx < len(classes) else pos_idx

            # Save analysis results to Results folder
            self._save_analysis_results(prob_map, grid_stats, class_name, blob_count)

            self.status_bar.showMessage(
                f"Analysis complete: {blob_count} pixels detected for '{class_name}' "
                f"in {elapsed:.2f}s"
            )
            logger.info("Analysis completed in %.2fs", elapsed)

        except Exception as e:
            logger.exception("Analysis failed: %s", e)
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")

    def _save_analysis_results(self, prob_map: np.ndarray, grid_stats: List[Dict],
                               class_name: str, blob_count: int):
        """Save analysis results (probability map and grid stats) to Results folder."""
        try:
            if self.current_cluster_id is None:
                return

            cluster_id = self.current_cluster_id
            results_folder = os.path.join(RESULTS_FOLDER, cluster_id)
            os.makedirs(results_folder, exist_ok=True)

            # Save probability map as .npy
            prob_map_path = os.path.join(results_folder, f"prob_map_{cluster_id}.npy")
            np.save(prob_map_path, prob_map.astype(np.float32))

            # Save grid statistics as CSV
            import csv
            csv_path = os.path.join(results_folder, f"grid_stats_{cluster_id}.csv")
            with open(csv_path, 'w', newline='') as f:
                if grid_stats:
                    writer = csv.DictWriter(f, fieldnames=grid_stats[0].keys())
                    writer.writeheader()
                    writer.writerows(grid_stats)

            # Save summary info
            summary_path = os.path.join(results_folder, f"analysis_summary_{cluster_id}.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Cluster ID: {cluster_id}\n")
                f.write(f"Date: {self.current_date}\n")
                f.write(f"Model: {os.path.basename(self.lda_path)}\n")
                f.write(f"Detected Class: {class_name}\n")
                f.write(f"Pixels Detected: {blob_count}\n")
                f.write(f"Cell Size: {self.cell_size}\n")
                f.write(f"Probability Threshold (Pixel): {self.pix_thr}\n")
                f.write(f"Patch Threshold: {self.patch_thr_value:.2f} ({int(self.patch_thr_value * 100)}%)\n")
                f.write(f"Filters Applied:\n")
                f.write(f"  - Border Filter: {self.use_border_chk.isChecked()}")
                if self.use_border_chk.isChecked():
                    f.write(f" (radius={self.border_spin.value()} px)\n")
                else:
                    f.write(f"\n")
                f.write(f"  - Area Filter: {self.use_area_chk.isChecked()}")
                if self.use_area_chk.isChecked():
                    f.write(f" (min={self.area_min_spin.value()}, max={self.area_max_spin.value()} px²)\n")
                else:
                    f.write(f"\n")
                f.write(f"  - Shape Filter: {self.use_circ_chk.isChecked()}")
                if self.use_circ_chk.isChecked():
                    f.write(f" (circularity: {self.circ_min_spin.value():.2f}-{self.circ_max_spin.value():.2f})\n")
                else:
                    f.write(f"\n")
                f.write(f"\nGrid Statistics:\n")
                f.write(f"  - Total Patches: {len(grid_stats)}\n")
                if grid_stats:
                    percents = [s['percent_cracked'] for s in grid_stats]
                    f.write(f"  - Mean Cracked %: {np.mean(percents):.2f}%\n")
                    f.write(f"  - Max Cracked %: {np.max(percents):.2f}%\n")
                    f.write(f"  - Min Cracked %: {np.min(percents):.2f}%\n")

            logger.info("Analysis results saved to: %s", results_folder)

        except Exception as e:
            logger.warning("Failed to save analysis results: %s", e)

    def _show_sam_segments(self):
        """
        BLOB SEGMENTATION MODE:
        Segment each detected crack/blob (from Panel 3 - after filters) using SAM2.
        Uses the centroid of each filtered blob as a prompt for SAM2 segmentation.

        IMPORTANT: Image stays UPRIGHT (no rotation applied).
        """
        # ========================================================================
        # Require detection mask from Panel 3 (filtered blobs)
        # ========================================================================
        if self.last_detection_mask is None:
            QMessageBox.warning(self, "No Detection", "Run Analysis first to generate detections.")
            return

        if self.hsi_cube is None:
            QMessageBox.warning(self, "No HSI", "No HSI cube loaded.")
            return

        # Use REFLECTANCE image for SAM, fallback to RGB if not available
        source_image = None
        image_type = ""
        if self.reflectance_image is not None:
            source_image = self.reflectance_image
            image_type = "REFLECTANCE"
        elif self.rgb_image is not None:
            source_image = self.rgb_image
            image_type = "RGB"
        else:
            QMessageBox.warning(self, "No Image", "No REFLECTANCE or RGB image loaded.")
            return

        try:
            # Show progress dialog
            progress = QProgressDialog("Testing SAM2 Cluster Detection (Center Point)...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()

            # Initialize SAM if not already done
            if self.sam2_segmenter is None:
                logger.info("Initializing SAM2 predictor...")
                progress.setLabelText("Initializing SAM2 predictor...")
                QApplication.processEvents()
                initial_settings()
                predictor = initialize_sam2_predictor()
                self.sam2_segmenter = create_point_segmenter(predictor)
                logger.info("SAM2 predictor initialized successfully")

            # ========================================================================
            # RESIZE SOURCE IMAGE: Match HSI dimensions (512x512)
            # ========================================================================
            hsi_h, hsi_w = self.hsi_cube.shape[:2]  # Get HSI dimensions (512x512)
            image_original = source_image
            image_resized = cv2.resize(image_original, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)
            logger.info(f"[BLOB SEGMENTATION] Resized {image_type} from {image_original.shape[:2]} to {image_resized.shape[:2]}")

            # ========================================================================
            # EXTRACT BLOB CENTROIDS: From filtered detection mask (Panel 3 - Stage 3)
            # ========================================================================
            progress.setLabelText("Extracting blob centroids from filtered detections...")
            QApplication.processEvents()

            # Extract centroids from the FILTERED detection mask (Panel 3 result)
            max_blobs = self.max_sam_blobs
            blob_centroids_hsi = extract_blob_centroids(self.last_detection_mask, max_blobs=max_blobs)

            if not blob_centroids_hsi:
                progress.close()
                QMessageBox.warning(
                    self,
                    "No Blobs Found",
                    "No blobs detected in filtered mask (Panel 3).\n\n"
                    "Try:\n"
                    "- Lowering the Prob Thr threshold\n"
                    "- Disabling some geometric filters\n"
                    "- Running Analysis again"
                )
                return

            logger.info(f"[BLOB SEGMENTATION] Extracted {len(blob_centroids_hsi)} blob centroids from Panel 3 (after filters)")
            logger.info(f"[BLOB SEGMENTATION] HSI centroids (first 3): {blob_centroids_hsi[:3]}")

            # ========================================================================
            # COORDINATE TRANSFORM: Rotate centroids 90° CW to match REFLECTANCE/RGB
            # ========================================================================
            # Panel 3 uses HSI coordinate space (rotated 90° CW for display)
            # Panel 4 uses REFLECTANCE/RGB coordinate space (upright)
            # Need to transform: (x_hsi, y_hsi) → (x_rgb, y_rgb)
            # Rotation 90° CW: new_x = h - y, new_y = x

            h_hsi, w_hsi = self.last_detection_mask.shape  # HSI dimensions (512x512)
            blob_centroids_rgb = []

            for x_hsi, y_hsi in blob_centroids_hsi:
                # Rotate 90° clockwise to align with upright REFLECTANCE image
                x_rgb = h_hsi - y_hsi - 1
                y_rgb = x_hsi
                blob_centroids_rgb.append((x_rgb, y_rgb))

            logger.info(f"[BLOB SEGMENTATION] Rotated centroids to RGB space (first 3): {blob_centroids_rgb[:3]}")

            progress.setLabelText(f"Segmenting {len(blob_centroids_rgb)} blobs with SAM2...")
            progress.setMaximum(len(blob_centroids_rgb))
            QApplication.processEvents()

            # ========================================================================
            # SAM2 SEGMENTATION: Segment each blob using its centroid as prompt
            # ========================================================================
            # Segment each blob individually using RGB-space centroids
            all_masks = []
            successful_segments = 0

            for i, centroid_rgb in enumerate(blob_centroids_rgb):
                try:
                    # centroid_rgb is (x, y) tuple in RGB space - convert to [x, y] list for API
                    point = [centroid_rgb[0], centroid_rgb[1]]

                    # Segment this blob using RGB-space coordinates
                    _, blob_mask = self.sam2_segmenter.segment_object_from_array(
                        image_resized,
                        [point]  # Single point prompt in RGB coordinate space
                    )

                    all_masks.append(blob_mask)
                    successful_segments += 1

                except Exception as e:
                    logger.warning(f"[BLOB SEGMENTATION] Failed to segment blob {i} at RGB coords {centroid_rgb}: {e}")
                    # Add empty mask as placeholder
                    all_masks.append(np.zeros((hsi_h, hsi_w), dtype=bool))

                # Update progress
                progress.setValue(i + 1)
                QApplication.processEvents()

            logger.info(f"[BLOB SEGMENTATION] Successfully segmented {successful_segments}/{len(blob_centroids_rgb)} blobs")

            # ========================================================================
            # MERGE MASKS: Combine all blob segments into one mask
            # ========================================================================
            combined_mask = np.zeros((hsi_h, hsi_w), dtype=bool)
            for mask in all_masks:
                combined_mask = np.logical_or(combined_mask, mask)

            logger.info(f"[BLOB SEGMENTATION] Combined coverage: {combined_mask.sum() / combined_mask.size * 100:.1f}% of image")

            # ========================================================================
            # VISUALIZATION: Colored overlay on UPRIGHT source image (NO ROTATION)
            # ========================================================================
            progress.setLabelText("Creating colored segment overlay...")
            QApplication.processEvents()

            # Create RGB visualization with colored segments
            # IMPORTANT: Keep image UPRIGHT - do NOT rotate
            visualization = image_resized.copy()

            # Define colors for segments (cycle through if more blobs than colors)
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
                (0, 255, 128),  # Spring Green
                (255, 0, 128),  # Deep Pink
            ]

            # Draw each segment with a different color
            alpha = 0.4
            for i, mask in enumerate(all_masks):
                if mask.sum() == 0:
                    continue  # Skip empty masks

                color = np.array(colors[i % len(colors)], dtype=np.uint8)
                overlay = visualization.copy()
                overlay[mask] = color
                visualization = cv2.addWeighted(visualization, 1 - alpha, overlay, alpha, 0)

            # Draw centroid markers (small red circles) using RGB-space coordinates
            for centroid_rgb in blob_centroids_rgb:
                cv2.circle(visualization, centroid_rgb, 3, (255, 0, 0), -1)  # Red filled dot
                cv2.circle(visualization, centroid_rgb, 5, (255, 255, 255), 1)  # White border

            # ========================================================================
            # DISPLAY: Show UPRIGHT result on Panel 4 (NO ROTATION)
            # ========================================================================
            self._show_image(visualization, self.rgb_sam_label)

            # Cache results for potential reuse
            self.last_sam_segments = all_masks
            self.last_sam_overlay = visualization

            progress.close()

            # ========================================================================
            # STATUS UPDATE: Report results to user
            # ========================================================================
            combined_pixels = int(combined_mask.sum())
            total_pixels = combined_mask.size
            coverage_pct = combined_pixels / total_pixels * 100
            avg_blob_size = combined_pixels / len(blob_centroids_rgb) if blob_centroids_rgb else 0

            logger.info(f"[BLOB SEGMENTATION] ✅ Segmentation Complete:")
            logger.info(f"  - Image Type: {image_type}")
            logger.info(f"  - Blobs Detected: {len(blob_centroids_rgb)}")
            logger.info(f"  - Successfully Segmented: {successful_segments}")
            logger.info(f"  - Combined Coverage: {combined_pixels:,} pixels ({coverage_pct:.1f}%)")
            logger.info(f"  - Avg Blob Size: {avg_blob_size:.1f} pixels")
            logger.info(f"  - Centroids rotated 90° CW (HSI → RGB coordinate transform)")
            logger.info(f"  - Image kept UPRIGHT (no rotation applied)")

            self.status_bar.showMessage(
                f"✅ SAM SEGMENTATION: {successful_segments}/{len(blob_centroids_rgb)} blobs on {image_type} | "
                f"Coverage={coverage_pct:.1f}% | "
                f"Avg={avg_blob_size:.0f}px"
            )

            # Show info dialog with results
            QMessageBox.information(
                self,
                "SAM2 Blob Segmentation Results",
                f"🎯 Blob Segmentation Results:\n\n"
                f"Image Type: {image_type}\n"
                f"Source: Panel 3 (Filtered Blobs)\n"
                f"Blobs Detected: {len(blob_centroids_rgb)}\n"
                f"Successfully Segmented: {successful_segments}\n"
                f"Combined Coverage: {coverage_pct:.1f}%\n"
                f"Combined Pixels: {combined_pixels:,}\n"
                f"Avg Blob Size: {avg_blob_size:.0f} pixels\n\n"
                f"🌈 Colored overlay shows SAM segments\n"
                f"🔴 Red dots show blob centroids (prompts)\n"
                f"🔄 Centroids rotated 90° CW for RGB alignment\n"
                f"📐 Image displayed UPRIGHT (no rotation)"
            )

        except Exception as e:
            logger.exception("[BLOB SEGMENTATION] Failed: %s", e)
            self.status_bar.showMessage(f"❌ SAM segmentation failed: {e}")
            QMessageBox.critical(self, "Error", f"SAM2 Blob Segmentation Failed:\n{str(e)}")

    def _load_cnn_model(self):
        """Load the CNN model for grape/not-grape classification."""
        try:
            logger.info("[CNN] Loading CNN model for classification...")

            # Model path - use grayscale trained model from original_gray results
            model_path = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\models\training_classification_model_cnn_for_grapes_berry\train_model\results\original_gray\best_model.pth")

            logger.info(f"[CNN] Looking for model at: {model_path}")

            if not model_path.exists():
                logger.error(f"[CNN] ❌ Model not found at: {model_path}")
                logger.warning("[CNN] CNN classification will be disabled")
                self.cnn_model = None
                return

            logger.info(f"[CNN] ✅ Model file found")
            logger.info(f"[CNN] Building model architecture...")

            # Build model architecture (EfficientNet-B0 with GRAYSCALE input)
            # This matches the training code which uses get_model_gray
            model = efficientnet_b0(weights=None)

            # Modify first conv for grayscale (1 channel) - SAME AS TRAINING
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                1,  # 1 input channel (grayscale) - matches training
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            logger.info(f"[CNN] Modified first conv for grayscale (1 channel) - matches training")

            # Modify classifier for 2 classes (grape/not-grape)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 2)
            logger.info(f"[CNN] Modified classifier for 2 classes")

            # Load weights
            logger.info(f"[CNN] Loading checkpoint...")
            checkpoint = torch.load(str(model_path), map_location=self.cnn_device)
            logger.info(f"[CNN] Checkpoint loaded, type: {type(checkpoint)}")

            # Load state dict
            model.load_state_dict(checkpoint)
            logger.info(f"[CNN] State dict loaded successfully")

            model.to(self.cnn_device)
            model.eval()
            logger.info(f"[CNN] Model moved to {self.cnn_device} and set to eval mode")

            self.cnn_model = model

            # Setup preprocessing transforms - EXACTLY MATCH TRAINING
            # Training uses: Resize(224,224) -> Grayscale(1) -> ToTensor -> Normalize(0.5, 0.5)
            self.cnn_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),  # 1 channel - matches training
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
            ])
            logger.info(f"[CNN] Preprocessing: Resize(224,224) -> Grayscale(1ch) -> Normalize(0.5)")

            logger.info(f"[CNN] ✅ Model loaded successfully from: {model_path}")
            logger.info(f"[CNN] Device: {self.cnn_device}")
            logger.info(f"[CNN] Architecture: EfficientNet-B0 (grayscale input)")

        except Exception as e:
            logger.exception(f"[CNN] Failed to load model: {e}")
            self.cnn_model = None

    def _preprocess_crop_for_cnn(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess a crop for CNN classification - MATCHES TRAINING PIPELINE.

        Args:
            crop: RGB numpy array (H, W, 3)

        Returns:
            Preprocessed tensor (1, 1, 224, 224) - single channel grayscale
        """
        # Convert numpy (RGB) to PIL Image
        if crop.ndim == 2:
            # Already grayscale
            pil_image = Image.fromarray(crop)
        else:
            # RGB -> convert to PIL
            pil_image = Image.fromarray(crop.astype(np.uint8))

        # Apply transforms (resize, grayscale 1 channel, normalize)
        # This EXACTLY matches the training pipeline
        tensor = self.cnn_transform(pil_image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, 1, 224, 224)

        return tensor

    def _classify_crop(self, crop: np.ndarray) -> Tuple[int, float]:
        """
        Classify a single crop as grape (1) or not-grape (0).

        Args:
            crop: RGB numpy array

        Returns:
            (predicted_class, confidence) where class is 0 (not-grape) or 1 (grape)
        """
        if self.cnn_model is None:
            # Model not loaded, default to grape (1)
            logger.warning("[CNN] Model is None, using default classification")
            return 1, 0.5

        try:
            # Preprocess
            tensor = self._preprocess_crop_for_cnn(crop)
            tensor = tensor.to(self.cnn_device)

            # Inference
            with torch.no_grad():
                outputs = self.cnn_model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            pred_class = predicted.item()
            conf_score = confidence.item()

            # Get both class probabilities for debugging
            prob_not_grape = probs[0, 0].item()
            prob_grape = probs[0, 1].item()

            logger.debug(f"[CNN] Probs: not_grape={prob_not_grape:.3f}, grape={prob_grape:.3f}, pred={pred_class}, conf={conf_score:.3f}")

            return pred_class, conf_score

        except Exception as e:
            logger.exception(f"[CNN] Classification failed: {e}")
            return 1, 0.5  # Default to grape

    def _debug_cnn_crops(self):
        """
        Extract crops from SAM segments and show them in a gallery dialog.
        Shows bounding boxes on Panel 5 (using REFLECTANCE image from HS/results).

        NEW LOGIC:
        1. Uses SAM segments from Panel 4 (self.last_sam_segments)
        2. For each segment mask, finds min/max coordinates (LEFT, RIGHT, UP, DOWN)
        3. Extracts bounding box crop from REFLECTANCE image
        4. Shows crops in popup gallery dialog
        """
        # Check if SAM segments exist
        if not hasattr(self, 'last_sam_segments') or not self.last_sam_segments:
            QMessageBox.warning(
                self,
                "No SAM Segments",
                "Please run 'Show SAM Segments' first to generate segments.\n\n"
                "Steps:\n"
                "1. Load images\n"
                "2. Run Analysis\n"
                "3. Click '🎯 Show SAM Segments'\n"
                "4. Then click '🔍 Run CNN Debug'"
            )
            return

        try:
            # Show progress
            progress = QProgressDialog("Extracting crops from SAM segments...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()

            # ========================================================================
            # USE REFLECTANCE IMAGE FROM HS/results FOLDER (same as Panel 4)
            # ========================================================================
            if not hasattr(self, 'reflectance_image') or self.reflectance_image is None:
                # Try to load REFLECTANCE image
                if self.hdr_path:
                    reflectance_path = self._find_reflectance_image()
                    if reflectance_path and os.path.exists(reflectance_path):
                        bgr = cv2.imread(reflectance_path)
                        if bgr is not None:
                            self.reflectance_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            logger.info(f"[CNN DEBUG] Loaded REFLECTANCE image from: {reflectance_path}")

            # Check if we have REFLECTANCE or need to fallback to RGB
            if hasattr(self, 'reflectance_image') and self.reflectance_image is not None:
                # Use REFLECTANCE image (UPRIGHT, from HS/results/REFLECTANCE_*.png)
                source_image_original = self.reflectance_image
                image_type = "REFLECTANCE"
                logger.info(f"[CNN DEBUG] Using REFLECTANCE image: {source_image_original.shape[:2]}")
            elif hasattr(self, 'rgb_image') and self.rgb_image is not None:
                # Fallback to RGB image
                source_image_original = self.rgb_image
                image_type = "RGB"
                logger.info(f"[CNN DEBUG] Using RGB image (REFLECTANCE not available): {source_image_original.shape[:2]}")
            else:
                progress.close()
                QMessageBox.warning(self, "No Image", "No REFLECTANCE or RGB image available.")
                return

            # Resize to match HSI dimensions (SAM segments are in HSI coordinate space)
            hsi_h, hsi_w = self.last_sam_segments[0].shape
            source_image = cv2.resize(source_image_original, (hsi_w, hsi_h), interpolation=cv2.INTER_LINEAR)
            logger.info(f"[CNN DEBUG] Resized {image_type} from {source_image_original.shape[:2]} to {source_image.shape[:2]}")

            # ========================================================================
            # PROCESS SAM SEGMENTS - Extract BBox from each segment mask
            # SAM segments are already in upright coordinate space (no rotation needed)
            # ========================================================================
            logger.info(f"[CNN DEBUG] Processing {len(self.last_sam_segments)} SAM segments")

            viz_image = source_image.copy()
            crop_list = []

            progress.setMaximum(len(self.last_sam_segments))

            # Process each SAM segment
            for i, segment_mask in enumerate(self.last_sam_segments):
                # Skip empty masks
                if segment_mask.sum() == 0:
                    logger.warning(f"[CNN DEBUG] Skipping empty segment {i}")
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                    continue

                # Find coordinates where mask is True
                coords = np.argwhere(segment_mask)

                if len(coords) == 0:
                    logger.warning(f"[CNN DEBUG] Skipping segment {i} with no coordinates")
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                    continue

                # Get bounding box from mask coordinates
                # coords are (row, col) = (y, x)
                y_coords = coords[:, 0]
                x_coords = coords[:, 1]

                # Find min/max (LEFT, RIGHT, UP, DOWN)
                y_min = int(y_coords.min())  # UP
                y_max = int(y_coords.max())  # DOWN
                x_min = int(x_coords.min())  # LEFT
                x_max = int(x_coords.max())  # RIGHT

                # Clip to image bounds (safety)
                y_min = max(0, y_min)
                x_min = max(0, x_min)
                y_max = min(hsi_h - 1, y_max)
                x_max = min(hsi_w - 1, x_max)

                # Skip if bbox is invalid
                if x_max <= x_min or y_max <= y_min:
                    logger.warning(f"[CNN DEBUG] Skipping invalid bbox {i}: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                    continue

                # ================================================================
                # CROP EXTRACTION - Two Modes:
                # 1. Full BBox: Extract entire bounding box region
                # 2. Segment Only: Extract segment and mask out background
                # ================================================================
                if self.crop_segment_only_chk.isChecked():
                    # MODE: Segment Only (Masked)
                    # Extract bbox region and apply mask to show only segment pixels
                    crop_bbox = source_image[y_min:y_max+1, x_min:x_max+1].copy()
                    segment_bbox_mask = segment_mask[y_min:y_max+1, x_min:x_max+1].copy()

                    # Create masked crop - set background to black (0,0,0)
                    crop = crop_bbox.copy()
                    crop[~segment_bbox_mask] = 0  # Black background where mask is False

                    logger.info(f"[CNN DEBUG] Segment {i}: BBox=[x:{x_min}-{x_max}, y:{y_min}-{y_max}], Mode=SEGMENT_ONLY, Crop size={crop.shape[:2]}")
                else:
                    # MODE: Full BBox (Default)
                    # Extract entire bounding box region without masking
                    crop = source_image[y_min:y_max+1, x_min:x_max+1].copy()

                    logger.info(f"[CNN DEBUG] Segment {i}: BBox=[x:{x_min}-{x_max}, y:{y_min}-{y_max}], Mode=FULL_BBOX, Crop size={crop.shape[:2]}")

                # Skip empty crops
                if crop.size == 0:
                    logger.warning(f"[CNN DEBUG] Skipping empty crop {i}")
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                    continue

                # Store crop with bbox info for later CNN classification
                crop_list.append({
                    'crop': crop,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'segment_idx': i
                })

                # Update progress
                progress.setValue(i + 1)
                QApplication.processEvents()

            progress.close()

            # ========================================================================
            # SHOW POPUP GALLERY (IF ENABLED)
            # ========================================================================
            if crop_list and self.show_popup_before_cnn:
                # Show gallery before CNN classification
                crops_only = [item['crop'] for item in crop_list]
                mode_text = "Segment Only (Masked)" if self.crop_segment_only_chk.isChecked() else "Full BBox"
                gallery = CropGalleryDialog(crops_only, parent=self, title_suffix=f" - {mode_text}")
                gallery.exec_()
                logger.info(f"[CNN DEBUG] Crop gallery closed by user")

            # ========================================================================
            # CNN CLASSIFICATION
            # ========================================================================
            if self.cnn_model is not None:
                logger.info(f"[CNN] Starting classification of {len(crop_list)} crops...")
                progress = QProgressDialog("Running CNN classification...", None, 0, len(crop_list), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                QApplication.processEvents()

                self.cnn_classifications = []

                for idx, crop_data in enumerate(crop_list):
                    crop = crop_data['crop']
                    bbox = crop_data['bbox']
                    segment_idx = crop_data['segment_idx']

                    # Classify crop
                    pred_class, confidence = self._classify_crop(crop)

                    # Store result
                    self.cnn_classifications.append({
                        'segment_idx': segment_idx,
                        'bbox': bbox,
                        'class': pred_class,
                        'confidence': confidence,
                        'class_name': 'grape' if pred_class == 1 else 'not_grape'
                    })

                    logger.info(f"[CNN] Segment {segment_idx}: {['NOT_GRAPE', 'GRAPE'][pred_class]} (conf={confidence:.3f})")

                    progress.setValue(idx + 1)
                    QApplication.processEvents()

                progress.close()
                logger.info(f"[CNN] ✅ Classification complete")

                # Count results
                grape_count = sum(1 for c in self.cnn_classifications if c['class'] == 1)
                not_grape_count = len(self.cnn_classifications) - grape_count
                logger.info(f"[CNN] Results: {grape_count} GRAPE, {not_grape_count} NOT_GRAPE")

            else:
                logger.warning("[CNN] Model not loaded, skipping classification")
                # Default all to grape
                self.cnn_classifications = []
                for idx, crop_data in enumerate(crop_list):
                    self.cnn_classifications.append({
                        'segment_idx': crop_data['segment_idx'],
                        'bbox': crop_data['bbox'],
                        'class': 1,  # Default: grape
                        'confidence': 0.5,
                        'class_name': 'grape'
                    })

            # ========================================================================
            # UPDATE PANEL 5 WITH COLOR-CODED BBOXES
            # GREEN = grape (class 1), RED = not_grape (class 0)
            # ========================================================================
            for result in self.cnn_classifications:
                bbox = result['bbox']
                x_min, y_min, x_max, y_max = bbox
                pred_class = result['class']
                confidence = result['confidence']
                segment_idx = result['segment_idx']

                # Choose color: GREEN for grape, RED for not-grape
                if pred_class == 1:
                    color = (0, 255, 0)  # GREEN (grape)
                else:
                    color = (255, 0, 0)  # RED (not-grape)

                # Draw rectangle
                cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), color, 2)

                # Add label with class and confidence
                label = f"{segment_idx}: {result['class_name']} ({confidence:.2f})"
                cv2.putText(viz_image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, color, 1, cv2.LINE_AA)

            # Update Panel 5
            self.cnn_candidates_image = viz_image
            self._show_image(viz_image, self.cnn_candidates_label)

            # Log results
            crop_mode = "SEGMENT_ONLY" if self.crop_segment_only_chk.isChecked() else "FULL_BBOX"
            logger.info(f"[CNN DEBUG] ✅ Complete:")
            logger.info(f"  - SAM Segments Processed: {len(self.last_sam_segments)}")
            logger.info(f"  - Crops Extracted: {len(crop_list)}")
            logger.info(f"  - Crop Mode: {crop_mode}")
            logger.info(f"  - CNN Classifications: {len(self.cnn_classifications)}")
            logger.info(f"  - Panel 5 updated with color-coded bounding boxes")

            grape_count = sum(1 for c in self.cnn_classifications if c['class'] == 1)
            not_grape_count = len(self.cnn_classifications) - grape_count
            self.status_bar.showMessage(
                f"✅ CNN Complete: {grape_count} GRAPE (green), {not_grape_count} NOT_GRAPE (red)"
            )

        except Exception as e:
            logger.exception("[CNN DEBUG] Failed: %s", e)
            self.status_bar.showMessage(f"❌ CNN Debug failed: {e}")
            QMessageBox.critical(self, "Error", f"CNN Debug Failed:\n{str(e)}")

    def _show_image(self, img: np.ndarray, label: QLabel, is_grayscale: bool = False):
        """Display image in QLabel."""
        if is_grayscale and img.ndim == 2:
            h, w = img.shape
            q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w, ch = img.shape
            q = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q))
        label.setScaledContents(True)

    def _save_screenshot(self):
        """Save screenshot of all 6 panels."""
        try:
            # Ask for prefix
            options = ["new_detect", "new_patch", "old_detect", "old_patch"]
            prefix, ok = QInputDialog.getItem(
                self, "Screenshot Prefix",
                "Select prefix for screenshot name:",
                options, 0, False
            )
            if not ok:
                self.status_bar.showMessage("Screenshot cancelled.")
                return

            # Get pixmaps from all 6 panels
            hsi_band_pm = self.hsi_band_label.pixmap()
            hsi_det_before_pm = self.hsi_detection_before_label.pixmap()
            hsi_det_after_pm = self.hsi_detection_after_label.pixmap()
            rgb_sam_pm = self.rgb_sam_label.pixmap()
            cnn_candidates_pm = self.cnn_candidates_label.pixmap()
            rgb_pm = self.rgb_label.pixmap()

            if not all([hsi_band_pm, hsi_det_before_pm, hsi_det_after_pm, rgb_sam_pm, cnn_candidates_pm, rgb_pm]):
                self.status_bar.showMessage("Cannot screenshot: missing images")
                return

            # Convert to numpy arrays
            def qpixmap_to_np(pm):
                img = pm.toImage()
                img = img.convertToFormat(QImage.Format_RGB888)
                w, h = img.width(), img.height()
                ptr = img.bits()
                ptr.setsize(h * w * 3)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
                return arr.copy()

            target_size = (250, 250)  # Smaller to fit 6 panels
            band_np = cv2.resize(qpixmap_to_np(hsi_band_pm), target_size)
            det_before_np = cv2.resize(qpixmap_to_np(hsi_det_before_pm), target_size)
            det_after_np = cv2.resize(qpixmap_to_np(hsi_det_after_pm), target_size)
            rgb_sam_np = cv2.resize(qpixmap_to_np(rgb_sam_pm), target_size)
            cnn_candidates_np = cv2.resize(qpixmap_to_np(cnn_candidates_pm), target_size)
            rgb_np = cv2.resize(qpixmap_to_np(rgb_pm), target_size)

            # Concatenate horizontally (6 panels)
            combined = cv2.hconcat([band_np, det_before_np, det_after_np, rgb_sam_np, cnn_candidates_np, rgb_np])

            # Save to Results folder
            cluster_id = self.current_cluster_id or 'unknown'
            filename = f"{prefix}_{cluster_id}.png"
            # Use Results folder in the project
            results_folder = os.path.join(RESULTS_FOLDER, cluster_id)
            os.makedirs(results_folder, exist_ok=True)
            save_path = os.path.join(results_folder, filename)
            cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            self.status_bar.showMessage(f"Screenshot saved: {save_path}")
            logger.info("Screenshot saved: %s", save_path)

        except Exception as e:
            logger.exception("Screenshot failed: %s", e)
            QMessageBox.critical(self, "Error", f"Screenshot failed: {e}")

    def _browse_dataset_csv(self):
        """Browse for dataset CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset CSV",
            DEFAULT_DATASET_CSV,
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.dataset_csv_edit.setText(path)

    def _load_dataset(self):
        """Load dataset from CSV file."""
        csv_path = self.dataset_csv_edit.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Dataset", f"Dataset CSV not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            # Normalize column names
            df = df.rename(columns={c: c.strip() for c in df.columns})

            # Ensure required columns
            required_cols = ["grape_id", "row", "image_path", "label"]
            for col in required_cols:
                if col not in df.columns:
                    QMessageBox.warning(self, "Dataset", f"Missing column '{col}' in CSV")
                    return

            # Keep only relevant columns
            df = df[required_cols].copy()
            df["grape_id"] = df["grape_id"].astype(str)
            df["row"] = df["row"].astype(int)
            df["image_path"] = df["image_path"].astype(str)
            df["label"] = df["label"].astype(int)

            # Store dataset
            self.dataset_df = df
            self.dataset_current_index = 0

            # Populate table
            self.dataset_table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.dataset_table.setItem(i, 0, QTableWidgetItem(str(row["grape_id"])))
                self.dataset_table.setItem(i, 1, QTableWidgetItem(str(row["row"])))
                self.dataset_table.setItem(i, 2, QTableWidgetItem(str(row["label"])))
                self.dataset_table.setItem(i, 3, QTableWidgetItem(str(row["image_path"])))

            # Enable navigation buttons
            self.dataset_prev_btn.setEnabled(True)
            self.dataset_next_btn.setEnabled(True)

            # Update info label
            self.dataset_info_label.setText(f"Dataset loaded: {len(df)} samples")

            # Auto-load first sample
            self.dataset_table.selectRow(0)
            self._load_dataset_sample(0)

            logger.info("Loaded dataset: %d samples from %s", len(df), csv_path)
            self.status_bar.showMessage(f"Dataset loaded: {len(df)} samples")

        except Exception as e:
            logger.exception("Failed to load dataset: %s", e)
            QMessageBox.critical(self, "Dataset", f"Failed to load dataset: {e}")

    def _dataset_prev(self):
        """Navigate to previous dataset sample and load it."""
        try:
            if self.dataset_df is None or len(self.dataset_df) == 0:
                return
            idx = max(0, int(self.dataset_current_index) - 1)
            if idx != self.dataset_current_index:
                self.dataset_table.selectRow(idx)
                self._load_dataset_sample(idx)
                # Note: _load_dataset_sample will call _run_analysis() automatically
        except Exception as e:
            logger.exception("_dataset_prev failed: %s", e)

    def _dataset_next(self):
        """Navigate to next dataset sample and load it."""
        try:
            if self.dataset_df is None or len(self.dataset_df) == 0:
                return
            idx = min(len(self.dataset_df) - 1, int(self.dataset_current_index) + 1)
            if idx != self.dataset_current_index:
                self.dataset_table.selectRow(idx)
                self._load_dataset_sample(idx)
                # Note: _load_dataset_sample will call _run_analysis() automatically
        except Exception as e:
            logger.exception("_dataset_next failed: %s", e)

    def _load_dataset_sample(self, index: int):
        """Load one dataset sample row (best-effort): update UI and try to load HSI or RGB if possible.

        The dataset CSV stores an 'image_path' column which may be a folder or image file. We try to
        locate the HS/results/.hdr structure by checking the path and its parents; if found we call
        _load_images(folder). Otherwise we attempt to load an RGB image and display it.
        """
        try:
            if self.dataset_df is None or index < 0 or index >= len(self.dataset_df):
                return
            row = self.dataset_df.iloc[index]
            self.dataset_current_index = int(index)

            grape_id = str(row.get("grape_id", ""))
            img_path = str(row.get("image_path", ""))
            label = row.get("label", "")

            self.dataset_info_label.setText(f"Sample {index+1}/{len(self.dataset_df)} - ID: {grape_id} - Label: {label}")

            # Try loading HSI folder if image_path points to a folder or a file inside the folder structure
            if os.path.isdir(img_path):
                # img_path appears to be a folder; attempt to load images from it
                try:
                    self._load_images(img_path)
                    # Auto-run analysis after loading
                    self._run_analysis()
                    return
                except Exception:
                    pass

            # If path is a file, try to walk up to find HS/results/*.hdr
            if os.path.isfile(img_path):
                candidate = os.path.abspath(os.path.dirname(img_path))
            else:
                candidate = os.path.abspath(img_path) if img_path else None

            found = False
            if candidate:
                for _ in range(4):
                    hs = os.path.join(candidate, "HS")
                    res = os.path.join(hs, "results")
                    if os.path.isdir(res):
                        try:
                            self._load_images(candidate)
                            found = True
                            # Auto-run analysis after loading
                            self._run_analysis()
                            break
                        except Exception:
                            pass
                    parent = os.path.dirname(candidate)
                    if parent == candidate:
                        break
                    candidate = parent

            if found:
                return

            # Otherwise, attempt to load the image as a regular RGB image and display it
            if os.path.isfile(img_path):
                bgr = cv2.imread(img_path)
                if bgr is not None:
                    self.rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    self._show_image(self.rgb_image, self.rgb_label)
                    self.status_bar.showMessage(f"Loaded image: {img_path}")
                    return

            # No image loaded, update status
            self.status_bar.showMessage(f"Sample selected: {grape_id} (no loadable image)" )

        except Exception as e:
            logger.exception("_load_dataset_sample failed: %s", e)
            self.status_bar.showMessage(f"Failed to load sample: {e}")

    def _on_dataset_selection_changed(self):
        """
        Called when the user selects a different row in the dataset table.
        Loads the corresponding sample into the UI.
        """
        try:
            selected = self.dataset_table.selectedItems()
            if not selected:
                return
            # Get the selected row index
            row_idx = self.dataset_table.currentRow()
            if row_idx < 0 or self.dataset_df is None or row_idx >= len(self.dataset_df):
                return
            self._load_dataset_sample(row_idx)
        except Exception as e:
            import logging
            logging.exception("_on_dataset_selection_changed failed: %s", e)

    def _update_grid_progress(self, current: int, total: int, message: str = ""):
        """Update the grid/optuna progress bar and append message to log.

        Accepts (current, total, message) where current and total can be trial counts or percent-like values.
        """
        try:
            # Normalize to percentage
            if total and total > 0:
                pct = int(100.0 * float(current) / float(total))
            else:
                # If the caller used 0..100 convention, clamp
                pct = int(min(max(int(current), 0), 100))

            self.grid_progress.setVisible(True)
            self.grid_progress.setValue(pct)
            if message:
                self._append_log(message)
            self.status_bar.showMessage(message or f"Progress: {pct}%")
        except Exception as e:
            logger.exception("_update_grid_progress failed: %s", e)

    def _load_best_grid_result(self):
        """
        Load best hyperparameters from test_set_blob_patch_results.csv and:
        1. Parse the best parameters
        2. Update all UI controls
        3. Load the dataset and filter to test set only
        4. Start viewing the first test sample
        """
        # Default path for the grid search results
        results_csv = os.path.join(
            os.path.dirname(__file__),
            "test_set_blob_patch_results.csv"
        )

        # Allow user to browse if default doesn't exist
        if not os.path.exists(results_csv):
            results_csv, _ = QFileDialog.getOpenFileName(
                self,
                "Select Grid Search Results CSV",
                os.path.dirname(__file__),
                "CSV Files (*.csv);;All Files (*)"
            )
            if not results_csv:
                return

        try:
            # Read the CSV
            df = pd.read_csv(results_csv)

            # Check if CSV has data
            if len(df) == 0:
                QMessageBox.warning(self, "Load Best Params", "Grid search results CSV is empty!")
                return

            # Get the first row (best result)
            best = df.iloc[0]

            # Extract ALL hyperparameters (including morphology filters)
            prob_thr = float(best.get('prob_thr', 0.90))
            min_blob_size = int(best.get('min_blob_size', 0))
            circularity_min = best.get('circularity_min', None)
            circularity_max = best.get('circularity_max', None)
            aspect_ratio_limit = best.get('aspect_ratio_limit', None)
            solidity_max = best.get('solidity_max', None)
            patch_size = int(best.get('patch_size', 64))
            patch_pixel_ratio = float(best.get('patch_pixel_ratio', 0.10))
            global_threshold = float(best.get('global_threshold', 0.05))

            # Extract metrics if available
            metrics_info = ""
            try:
                f1_score = best.get('f1_score')
                if f1_score is not None:
                    # Convert to scalar if it's a Series/DataFrame
                    if hasattr(f1_score, 'item'):
                        f1_score = f1_score.item()
                    if not pd.isna(f1_score):
                        metrics_info = (
                            f"\n\nTest Set Metrics:"
                            f"\n  F1 Score:  {best.get('f1_score', 0):.4f}"
                            f"\n  Precision: {best.get('precision', 0):.4f}"
                            f"\n  Recall:    {best.get('recall', 0):.4f}"
                            f"\n  Accuracy:  {best.get('accuracy', 0):.4f}"
                        )
            except Exception:
                pass

            # Update UI controls
            logger.info("Applying best parameters: prob_thr=%.3f, min_blob=%.0f, patch_size=%.0f, patch_ratio=%.3f, global_thr=%.3f",
                       prob_thr, min_blob_size, patch_size, patch_pixel_ratio, global_threshold)

            # 1. Set Prob Thr spinbox
            self.thr_spin.setValue(prob_thr)
            self.pix_thr = prob_thr

            # 2. Set Area Min spinbox and enable Area filter checkbox
            self.area_min_spin.setValue(min_blob_size)
            if min_blob_size > 0:
                self.use_area_chk.setChecked(True)

            # 3. Set Cell Size spinbox (patch_size)
            self.cell_spin.setValue(patch_size)
            self.cell_size = patch_size

            # 4. Set Patch Thr combobox (patch_pixel_ratio)
            # Find closest value in combobox (0%, 5%, 10%, etc.)
            target_percent = int(patch_pixel_ratio * 100)
            closest_index = min(range(self.patch_thr_combo.count()),
                               key=lambda i: abs(self.patch_thr_combo.itemData(i) * 100 - target_percent))
            self.patch_thr_combo.setCurrentIndex(closest_index)
            self.patch_thr_value = patch_pixel_ratio

            # 5. Store global_threshold (you might need to add UI control for this if needed)
            # For now, we'll store it as an attribute
            self.best_global_threshold = global_threshold

            # Display status message with ALL hyperparameters
            status_msg = (
                f"✓ Loaded Best Grid Search Parameters:\n\n"
                f"Pixel Classification:\n"
                f"  Prob Thr:            {prob_thr:.4f}\n\n"
                f"Blob Filtering:\n"
                f"  Min Blob Size:       {min_blob_size}\n"
                f"  Circularity Min:     {circularity_min if circularity_min is not None else 'None'}\n"
                f"  Circularity Max:     {circularity_max if circularity_max is not None else 'None'}\n"
                f"  Aspect Ratio Limit:  {aspect_ratio_limit if aspect_ratio_limit is not None else 'None'}\n"
                f"  Solidity Max:        {solidity_max if solidity_max is not None else 'None'}\n\n"
                f"Patch Analysis:\n"
                f"  Patch Size:          {patch_size}\n"
                f"  Patch Pixel Ratio:   {patch_pixel_ratio:.4f}\n\n"
                f"Image-Level Decision:\n"
                f"  Global Threshold:    {global_threshold:.4f}"
                f"{metrics_info}"
            )

            QMessageBox.information(self, "Best Parameters Loaded", status_msg)

            # 6. Load dataset and filter to test set
            csv_path = self.dataset_csv_edit.text().strip()
            if not csv_path or not os.path.exists(csv_path):
                # Use default
                csv_path = DEFAULT_DATASET_CSV
                self.dataset_csv_edit.setText(csv_path)

            if os.path.exists(csv_path):
                # Load full dataset
                df_full = pd.read_csv(csv_path)
                df_full = df_full.rename(columns={c: c.strip() for c in df_full.columns})

                # Filter to test set only (row == 2)
                df_test = df_full[df_full['row'] == 2].copy()

                if len(df_test) == 0:
                    QMessageBox.warning(self, "Test Set", "No test set samples found (row == 2) in dataset!")
                    return

                # Ensure required columns
                required_cols = ["grape_id", "row", "image_path", "label"]
                df_test = df_test[required_cols].copy()
                df_test["grape_id"] = df_test["grape_id"].astype(str)
                df_test["row"] = df_test["row"].astype(int)
                df_test["image_path"] = df_test["image_path"].astype(str)
                df_test["label"] = df_test["label"].astype(int)

                # Reset index
                df_test = df_test.reset_index(drop=True)

                # Store dataset
                self.dataset_df = df_test
                self.dataset_current_index = 0

                # Populate table
                self.dataset_table.setRowCount(len(df_test))
                for i, row in df_test.iterrows():
                    self.dataset_table.setItem(i, 0, QTableWidgetItem(str(row["grape_id"])))
                    self.dataset_table.setItem(i, 1, QTableWidgetItem(str(row["row"])))
                    self.dataset_table.setItem(i, 2, QTableWidgetItem(str(row["label"])))
                    self.dataset_table.setItem(i, 3, QTableWidgetItem(str(row["image_path"])))

                # Enable navigation buttons
                self.dataset_prev_btn.setEnabled(True)
                self.dataset_next_btn.setEnabled(True)

                # Update info label
                self.dataset_info_label.setText(f"Test Set: {len(df_test)} samples (row==2)")

                # Auto-load first test sample
                self.dataset_table.selectRow(0)
                self._load_dataset_sample(0)

                logger.info("Filtered to test set: %d samples", len(df_test))
                self.status_bar.showMessage(f"✓ Best params loaded | Test set: {len(df_test)} samples | Use Next/Prev to navigate")
            else:
                self.status_bar.showMessage(f"✓ Best params loaded: Thr={prob_thr:.2f}, Blob={min_blob_size}, Patch={patch_size}")

        except Exception as e:
            logger.exception("Failed to load best grid result: %s", e)
            QMessageBox.critical(self, "Load Best Params", f"Failed to load grid search results:\n{e}")

    def _start_optuna_search(self):
        """
        Start Optuna hyperparameter optimization from the UI controls.
        """
        # Disable start button, enable stop button
        self.start_grid_btn.setEnabled(False)
        self.stop_grid_btn.setEnabled(True)
        self.grid_progress.setVisible(True)
        self.grid_progress.setValue(0)
        self.log_text_edit.clear()
        self.status_bar.showMessage("Optuna optimization started...")

        # Get parameters from UI
        dataset_csv = self.grid_dataset_edit.text().strip()
        model_idx = self.grid_model_combo.currentIndex()
        model_path = self.available_model_paths[model_idx] if 0 <= model_idx < len(self.available_model_paths) else None
        n_trials = self.optuna_trials_spin.value()

        # Create and start OptunaWorker
        self.optuna_worker = OptunaWorker(dataset_csv, model_path, n_trials)
        self.optuna_worker.log_signal.connect(self._append_log)
        self.optuna_worker.progress_signal.connect(self._update_optuna_progress)
        self.optuna_worker.finished_signal.connect(self._on_optuna_finished)
        self.optuna_worker.start()

    def _update_optuna_progress(self, cur_trial):
        """
        Update progress bar for Optuna trials.
        """
        total = self.optuna_trials_spin.value()
        pct = int(100.0 * float(cur_trial) / float(total))
        self.grid_progress.setValue(pct)
        self.status_bar.showMessage(f"Optuna progress: Trial {cur_trial}/{total} ({pct}%)")

    def _on_optuna_finished(self, success, result):
        """
        Handle Optuna completion: update UI, show best params, re-enable buttons.
        """
        self.start_grid_btn.setEnabled(True)
        self.stop_grid_btn.setEnabled(False)
        self.grid_progress.setVisible(False)
        if success and isinstance(result, dict):
            best_params = result.get("best_params", {})
            best_value = result.get("best_value", None)
            # Populate UI controls with best parameters
            if "prob_thr" in best_params:
                self.thr_spin.setValue(float(best_params["prob_thr"]))
            if "min_blob_size" in best_params:
                self.area_min_spin.setValue(int(best_params["min_blob_size"]))
                self.use_area_chk.setChecked(True)
            if "patch_size" in best_params:
                self.cell_spin.setValue(int(best_params["patch_size"]))
            if "patch_pixel_ratio" in best_params:
                target_percent = int(float(best_params["patch_pixel_ratio"]) * 100)
                closest_index = min(range(self.patch_thr_combo.count()),
                                   key=lambda i: abs(self.patch_thr_combo.itemData(i) * 100 - target_percent))
                self.patch_thr_combo.setCurrentIndex(closest_index)
            if "global_threshold" in best_params:
                self.best_global_threshold = float(best_params["global_threshold"])
            if "morph_size" in best_params:
                self.morph_size_spin.setValue(int(best_params["morph_size"]))

            # Circularity parameters
            if "circularity_min" in best_params:
                self.circ_min_spin.setValue(float(best_params["circularity_min"]))
                self.use_circ_chk.setChecked(True)
            if "circularity_max" in best_params:
                self.circ_max_spin.setValue(float(best_params["circularity_max"]))
                self.use_circ_chk.setChecked(True)

            # Aspect ratio parameters
            if "aspect_ratio_min" in best_params:
                self.aspect_ratio_min_spin.setValue(float(best_params["aspect_ratio_min"]))
                self.use_ar_chk.setChecked(True)
            if "aspect_ratio_limit" in best_params:
                self.aspect_ratio_max_spin.setValue(float(best_params["aspect_ratio_limit"]))
                self.use_ar_chk.setChecked(True)

            # Solidity parameters
            if "solidity_min" in best_params:
                self.solidity_min_spin.setValue(float(best_params["solidity_min"]))
                self.use_sol_chk.setChecked(True)
            if "solidity_max" in best_params:
                self.solidity_max_spin.setValue(float(best_params["solidity_max"]))
                self.use_sol_chk.setChecked(True)

            # Show message box
            msg = f"Optimization Complete! Best F2-Score: {best_value}\nParameters loaded into UI."
            QMessageBox.information(self, "Optuna Optimization", msg)
            self.status_bar.showMessage(msg)
        else:
            err = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
            QMessageBox.critical(self, "Optuna Optimization", f"Optimization failed: {err}")
            self.status_bar.showMessage(f"Optuna failed: {err}")

    def _stop_grid_search(self):
        """
        Stop the running Optuna optimization.
        """
        if hasattr(self, 'optuna_worker') and self.optuna_worker is not None:
            self.optuna_worker.stop()
            self.status_bar.showMessage("Optuna optimization stop requested...")
        else:
            self.status_bar.showMessage("No optimization running to stop.")

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    viewer = HSILateDetectionViewer()
    viewer.show()
    sys.exit(app.exec_())

