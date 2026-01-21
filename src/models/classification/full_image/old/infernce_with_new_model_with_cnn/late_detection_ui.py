"""
Late Detection UI - PyQt5 Viewer Application

This module contains the PyQt5 GUI application for interactive visualization
of late detection results with 4-panel layout:
1. HSI Band (default: band 138)
2. HSI Detection (binary mask overlay)
3. RGB Image (corresponding to HSI, from RGB folder)
4. RGB Image (Canon/standard RGB)
"""

import os
import sys
import logging
import time
from typing import Optional, List, Dict, Tuple

import logger
import numpy as np
import pandas as pd
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QStatusBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QInputDialog,
    QMessageBox, QLineEdit, QGroupBox, QTableWidget, QTableWidgetItem,
    QPlainTextEdit, QProgressBar, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QTextCursor

# Import core functions
try:
    from late_detection_core import (
        load_cube, load_model_and_scaler, per_pixel_probs, analyze_grid,
        filter_blobs_by_shape, find_band_index_for_wavelength, find_scaler,
        GRID_COLOR_BUCKETS
    )
    # Try to import CNN function, make it optional
    try:
        from late_detection_core import filter_blobs_with_cnn
    except ImportError:
        def filter_blobs_with_cnn(binary_mask, verifier, **kwargs):
            """Placeholder for CNN filtering"""
            logger.warning("filter_blobs_with_cnn not available, returning unfiltered mask")
            return binary_mask
except ImportError as e:
    logger.error(f"Failed to import core functions: {e}")
    raise

# Import CNN verifier
try:
    from cnn_verifier import BlobVerifier
    CNN_AVAILABLE = True
except ImportError:
    BlobVerifier = None
    CNN_AVAILABLE = False
    logger.warning("CNN verifier not available (cnn_verifier.py or dependencies not found)")

# Import grid search functions
try:
    from grid_search_blob_patch import GridSearchConfig
    GRID_SEARCH_AVAILABLE = True
    
    # Create placeholder functions for missing imports
    def load_probability_maps_from_results(df, row_filter=None):
        """Placeholder - implement if needed"""
        logger.warning("load_probability_maps_from_results not implemented")
        return {}, {}
    
    def run_grid_search(prob_maps, labels, config, n_jobs=1, progress_callback=None):
        """Placeholder - implement if needed"""
        logger.warning("run_grid_search not implemented")
        return []
    
    def evaluate_single_combination(prob_maps, labels, params):
        """Placeholder - implement if needed"""
        logger.warning("evaluate_single_combination not implemented")
        return {}
        
except ImportError:
    GridSearchConfig = None
    GRID_SEARCH_AVAILABLE = False
    logger.warning("Grid search not fully available")
    
    def load_probability_maps_from_results(df, row_filter=None):
        return {}, {}
    def run_grid_search(prob_maps, labels, config, n_jobs=1, progress_callback=None):
        return []
    def evaluate_single_combination(prob_maps, labels, params):
        return {}

try:
    from run_late_detection_inference import prepare_and_run_inference
except ImportError:
    prepare_and_run_inference = None
    logger.warning("prepare_and_run_inference not available")

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
    "OLD LDA [1=CRACK, 0=regular]": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\pixel_level\simple_classification_leave_one_out\comare_all_models\models\LDA_Balanced.pkl",
}
DEFAULT_SEARCH_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"
DEFAULT_DATASET_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset\hole_image\late_detection\late_detection_dataset.csv"
RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_cnn\Results"


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
        # Fallback to single band
        band = cv2.normalize(cube[:, :, band_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(band, cv2.COLOR_GRAY2RGB)


def get_rgb_path_from_hsi(hsi_path: str) -> Optional[str]:
    """
    Convert HSI path to corresponding RGB path.

    Logic:
    - HSI file: .../HS/results/REFLECTANCE_<date>_<id>.hdr
    - RGB file: .../HS/<date>_<id>.png

    Example:
    Input:  C:\\...\\1_04\\25.09.24\\HS\\results\\REFLECTANCE_2024-09-25_004.hdr
    Output: C:\\...\\1_04\\25.09.24\\HS\\2024-09-25_004.png

    Args:
        hsi_path: Path to HSI file (.hdr, .png, .tif, etc.)

    Returns:
        Path to RGB file if found, None otherwise
    """
    try:
        # Normalize path separators
        hsi_path = hsi_path.replace('\\', '/')

        # Get filename from HSI path
        filename = os.path.basename(hsi_path)

        # Extract date and ID from REFLECTANCE_<date>_<id>
        if filename.startswith('REFLECTANCE_'):
            # Remove 'REFLECTANCE_' prefix and extension
            suffix = filename.replace('REFLECTANCE_', '')  # e.g., "2024-09-25_004.hdr"
            suffix_noext = os.path.splitext(suffix)[0]  # e.g., "2024-09-25_004"

            # Replace /HS/results/ with /HS/ to go up one directory
            if '/HS/results/' in hsi_path:
                hs_dir = hsi_path.split('/HS/results/')[0] + '/HS'
            elif '/HS/' in hsi_path:
                # If path is already in HS folder (not results subfolder)
                hs_dir = os.path.dirname(hsi_path)
            else:
                logger.warning(f"Path doesn't contain '/HS/': {hsi_path}")
                return None

            # Try different extensions
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                candidate = os.path.join(hs_dir, f"{suffix_noext}{ext}").replace('/', '\\')
                if os.path.exists(candidate):
                    logger.info(f"Found RGB image: {candidate}")
                    return candidate

        logger.warning(f"RGB image not found for HSI: {hsi_path}")
        return None

    except Exception as e:
        logger.error(f"Error finding RGB path: {e}")
        return None


def draw_cnn_bboxes_on_rgb(rgb_image: np.ndarray, cnn_results: List[Dict], hsi_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Draw bounding boxes with CNN classifications on RGB image.

    Args:
        rgb_image: RGB image (H, W, 3)
        cnn_results: List of CNN prediction results
            [{'bbox': (x,y,w,h), 'prob_grape': 0.95, 'is_grape': True}, ...]
        hsi_shape: Optional (H, W) shape of HSI to scale bboxes if sizes don't match

    Returns:
        RGB image with bounding boxes drawn
    """
    # Create a copy to avoid modifying original
    img_with_boxes = rgb_image.copy()

    rgb_h, rgb_w = rgb_image.shape[:2]

    # Calculate scaling factors if HSI and RGB have different sizes
    scale_x = 1.0
    scale_y = 1.0
    if hsi_shape is not None:
        hsi_h, hsi_w = hsi_shape
        scale_x = rgb_w / hsi_w
        scale_y = rgb_h / hsi_h
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            logger.info(f"Scaling bboxes: HSI {hsi_shape} -> RGB ({rgb_h}, {rgb_w}), scale=({scale_x:.2f}, {scale_y:.2f})")
    else:
        logger.info(f"No scaling applied - bboxes already in RGB coordinates")

    for result in cnn_results:
        bbox = result['bbox']
        prob_grape = result['prob_grape']
        is_grape = result['is_grape']

        x, y, w, h = bbox

        # Scale bbox coordinates if needed
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        w_scaled = int(w * scale_x)
        h_scaled = int(h * scale_y)

        # Choose color based on classification
        if is_grape:
            color = (0, 255, 0)  # GREEN for grapes (kept)
            label = f"{prob_grape*100:.0f}%"
        else:
            color = (255, 0, 0)  # RED for noise (rejected)
            label = f"{prob_grape*100:.0f}%"

        # Draw rectangle
        thickness = 2
        cv2.rectangle(img_with_boxes, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, thickness)

        # Draw probability text above the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # Position text above the box (with background) - use scaled coordinates
        text_x = x_scaled
        text_y = y_scaled - 5 if y_scaled > 20 else y_scaled + h_scaled + 15

        # Draw text background (semi-transparent rectangle)
        bg_padding = 2
        cv2.rectangle(img_with_boxes,
                     (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                     (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                     color, -1)

        # Draw text
        cv2.putText(img_with_boxes, label, (text_x, text_y),
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return img_with_boxes


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
                prob_thr_candidates=np.arange(0.96, 1.00, 0.05).tolist(),
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
                        aspect_ratio_limit=best.get('aspect_ratio_limit', None),
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


# ===== Main UI Class =====

class HSILateDetectionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Late Detection Viewer - 4 Panel Layout")
        self.setGeometry(100, 100, 1600, 800)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # State variables
        self.hsi_cube: Optional[np.ndarray] = None
        self.hdr_path: Optional[str] = None
        self.rgb_image: Optional[np.ndarray] = None
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

        # Dataset state
        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_current_index: int = -1

        # Runtime params
        self.cell_size = 64
        self.pix_thr = 0.96  # Default threshold
        self.best_global_threshold = 0.05  # Global threshold from grid search

        # Grid search worker
        self.grid_search_worker: Optional[GridSearchWorker] = None

        # Logging handler for UI
        self.log_handler: Optional[QTextEditLogger] = None

        self._build_ui()
        self._setup_logging()
        self._discover_models()
        self._auto_load_model()

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

        # --- Top row: 4 image panels ---
        images_row = QHBoxLayout()
        layout.addLayout(images_row)

        # Panel 1: HSI Band (default: band 138)
        self.hsi_band_label = QLabel("HSI Band (Band 138)")
        self.hsi_band_label.setFixedSize(380, 380)
        self.hsi_band_label.setAlignment(Qt.AlignCenter)
        self.hsi_band_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.hsi_band_label)

        # Panel 2: HSI Detection
        self.hsi_detection_label = QLabel("HSI Detection")
        self.hsi_detection_label.setFixedSize(380, 380)
        self.hsi_detection_label.setAlignment(Qt.AlignCenter)
        self.hsi_detection_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.hsi_detection_label)

        # Panel 3: RGB Image (corresponding to HSI)
        self.hsi_grid_label = QLabel("RGB Image (HSI)")
        self.hsi_grid_label.setFixedSize(380, 380)
        self.hsi_grid_label.setAlignment(Qt.AlignCenter)
        self.hsi_grid_label.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self.hsi_grid_label)

        # Panel 4: RGB Image
        self.rgb_label = QLabel("RGB Image")
        self.rgb_label.setFixedSize(380, 380)
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
        controls = QHBoxLayout()
        layout.addLayout(controls)

        # Band slider
        controls.addWidget(QLabel("Band"))
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setValue(0)
        self.band_slider.valueChanged.connect(self._update_band)
        controls.addWidget(self.band_slider)

        # Cell size
        controls.addWidget(QLabel("Cell Size"))
        self.cell_spin = QSpinBox()
        self.cell_spin.setRange(8, 256)
        self.cell_spin.setSingleStep(8)
        self.cell_spin.setValue(self.cell_size)
        self.cell_spin.valueChanged.connect(self._update_cell)
        controls.addWidget(self.cell_spin)

        # Patch threshold dropdown (5% increments)
        controls.addWidget(QLabel("Patch Thr"))
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
        controls.addWidget(self.patch_thr_combo)

        # Probability threshold
        controls.addWidget(QLabel("Prob Thr"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.05)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(self.pix_thr)
        self.thr_spin.valueChanged.connect(self._update_thr)
        controls.addWidget(self.thr_spin)

        # Invert class
        self.invert_class_chk = QCheckBox("Invert class")
        self.invert_class_chk.setChecked(False)
        controls.addWidget(self.invert_class_chk)

        # Border filter (independent)
        self.use_border_chk = QCheckBox("Border filter")
        self.use_border_chk.setChecked(False)
        self.use_border_chk.setToolTip("Remove detections near image edges")
        controls.addWidget(self.use_border_chk)

        controls.addWidget(QLabel("Border R"))
        self.border_spin = QSpinBox()
        self.border_spin.setRange(0, 100)
        self.border_spin.setValue(20)
        self.border_spin.setFixedWidth(60)
        controls.addWidget(self.border_spin)

        # Area filter (independent)
        self.use_area_chk = QCheckBox("Area filter")
        self.use_area_chk.setChecked(False)
        self.use_area_chk.setToolTip("Remove detections that are too small or too large")
        controls.addWidget(self.use_area_chk)

        controls.addWidget(QLabel("Min"))
        self.area_min_spin = QSpinBox()
        self.area_min_spin.setRange(0, 100000)
        self.area_min_spin.setValue(10)
        self.area_min_spin.setFixedWidth(70)
        controls.addWidget(self.area_min_spin)

        controls.addWidget(QLabel("Max"))
        self.area_max_spin = QSpinBox()
        self.area_max_spin.setRange(1, 10000000)
        self.area_max_spin.setValue(5000)
        self.area_max_spin.setFixedWidth(70)
        controls.addWidget(self.area_max_spin)

        # Shape filter (independent) - Circularity/Compactness
        self.use_shape_chk = QCheckBox("Shape filter")
        self.use_shape_chk.setChecked(False)
        self.use_shape_chk.setToolTip("Remove detections by circularity (1.0=circle, 0.0=line)")
        controls.addWidget(self.use_shape_chk)

        controls.addWidget(QLabel("Min C"))
        self.circ_min_spin = QDoubleSpinBox()
        self.circ_min_spin.setRange(0.0, 1.0)
        self.circ_min_spin.setSingleStep(0.05)
        self.circ_min_spin.setDecimals(2)
        self.circ_min_spin.setValue(0.0)
        self.circ_min_spin.setFixedWidth(70)
        self.circ_min_spin.setToolTip("Minimum circularity (0.0-1.0)")
        controls.addWidget(self.circ_min_spin)

        controls.addWidget(QLabel("Max C"))
        self.circ_max_spin = QDoubleSpinBox()
        self.circ_max_spin.setRange(0.0, 1.0)
        self.circ_max_spin.setSingleStep(0.05)
        self.circ_max_spin.setDecimals(2)
        self.circ_max_spin.setValue(1.0)
        self.circ_max_spin.setFixedWidth(70)
        self.circ_max_spin.setToolTip("Maximum circularity (0.0-1.0)")
        controls.addWidget(self.circ_max_spin)

        # --- CNN Filter (for False Positive removal) ---
        controls.addWidget(QLabel("─" * 20))  # Separator

        self.use_cnn_chk = QCheckBox("Enable CNN Filter")
        self.use_cnn_chk.setChecked(False)
        self.use_cnn_chk.setToolTip("Use CNN to verify blobs and remove false positives")
        controls.addWidget(self.use_cnn_chk)

        self.show_cnn_patches_chk = QCheckBox("Show CNN Patches")
        self.show_cnn_patches_chk.setChecked(False)
        self.show_cnn_patches_chk.setToolTip("Display first 5 blob patches before CNN verification")
        controls.addWidget(self.show_cnn_patches_chk)

        controls.addWidget(QLabel("CNN Model:"))

        cnn_model_row = QHBoxLayout()
        self.cnn_model_edit = QLineEdit()
        default_cnn_path = r"C:\Users\yovel\Desktop\Grape_Project\src\models\training_classification_model_cnn_for_grapes_berry\model_weights\best_model_original.pth"
        self.cnn_model_edit.setText(default_cnn_path)
        self.cnn_model_edit.setPlaceholderText("Path to CNN model (.pth)")
        cnn_model_row.addWidget(self.cnn_model_edit)

        browse_cnn_btn = QPushButton("Browse")
        browse_cnn_btn.setFixedWidth(70)
        browse_cnn_btn.clicked.connect(self._browse_cnn_model)
        cnn_model_row.addWidget(browse_cnn_btn)

        controls.addLayout(cnn_model_row)

        # --- Action buttons row ---
        actions_row = QHBoxLayout()
        layout.addLayout(actions_row)

        run_analysis_btn = QPushButton("Run Analysis")
        run_analysis_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        run_analysis_btn.clicked.connect(self._run_analysis)
        actions_row.addWidget(run_analysis_btn)

        screenshot_btn = QPushButton("Screenshot (All 4 Panels)")
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

        self.start_grid_btn = QPushButton("🚀 Start Grid Search Pipeline")
        self.start_grid_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; font-size: 14px; padding: 10px;"
        )
        self.start_grid_btn.setToolTip("Run complete hyperparameter optimization pipeline")
        self.start_grid_btn.clicked.connect(self._start_grid_search)
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

            try:
                _, _, pos_idx, classes = load_model_and_scaler(self.lda_path, self.scaler_path)
                self.model_classes = classes
                self.current_pos_idx = pos_idx

                # Populate class selector
                self.class_combo.blockSignals(True)
                self.class_combo.clear()
                for i, cls in enumerate(classes):
                    self.class_combo.addItem(str(cls))
                self.class_combo.setCurrentIndex(pos_idx)
                self.class_combo.blockSignals(False)

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
        self.status_bar.showMessage(f"Patch threshold: {int(self.patch_thr_value * 100)}%")

    def _update_thr(self, v: float):
        """Update probability threshold."""
        self.pix_thr = float(v)

    def _show_cnn_patches(self, rgb_image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], padding: int = 10):
        """Display first 5 blob patches in a popup dialog."""
        if len(bboxes) == 0:
            return
        
        h_img, w_img = rgb_image.shape[:2]
        patches_to_show = min(5, len(bboxes))
        patch_images = []
        
        for i, (x, y, w, h) in enumerate(bboxes[:patches_to_show]):
            # Add padding for context
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            # Extract patch (no annotations)
            patch = rgb_image[y1:y2, x1:x2].copy()
            patch_images.append(patch)
        
        # Resize all patches to same height for display
        max_height = 150
        resized_patches = []
        for patch in patch_images:
            h_p, w_p = patch.shape[:2]
            scale = max_height / h_p
            new_w = int(w_p * scale)
            resized = cv2.resize(patch, (new_w, max_height))
            resized_patches.append(resized)
        
        # Concatenate patches horizontally
        combined_patches = cv2.hconcat(resized_patches)
        
        # Create a dialog to show patches
        msg = QMessageBox(self)
        msg.setWindowTitle("CNN Input Patches (First 5)")
        msg.setText(f"Showing {patches_to_show} of {len(bboxes)} blob patches\nPadding: {padding} pixels")
        
        # Convert to QPixmap and display
        h_c, w_c = combined_patches.shape[:2]
        q_img = QImage(combined_patches.data, w_c, h_c, 3 * w_c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        msg.setIconPixmap(pixmap.scaled(1200, 200, Qt.KeepAspectRatio))
        msg.exec_()

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
            lda, scaler, _, classes = load_model_and_scaler(self.lda_path, self.scaler_path)
            pos_idx = self.current_pos_idx

            # Compute probability map
            prob_map = per_pixel_probs(self.hsi_cube, lda, scaler, pos_idx)

            # Invert if requested
            if self.invert_class_chk.isChecked():
                logger.info("Inverting probabilities")
                prob_map = 1.0 - prob_map

            # Apply filters if any are enabled (all 3 filters are independent)
            use_border = self.use_border_chk.isChecked()
            use_area = self.use_area_chk.isChecked()
            use_shape = self.use_shape_chk.isChecked()

            if use_border or use_area or use_shape:
                filter_params = dict(
                    border_r=self.border_spin.value(),
                    area_min=self.area_min_spin.value(),
                    area_max=self.area_max_spin.value(),
                    circularity_min=self.circ_min_spin.value(),
                    circularity_max=self.circ_max_spin.value(),
                    use_border=use_border,
                    use_area=use_area,
                    use_shape=use_shape,
                )
                logger.info("Applying filters: %s", filter_params)
                mask_filtered = filter_blobs_by_shape(prob_map, self.pix_thr, **filter_params)
                prob_map = prob_map * mask_filtered

            # Apply CNN filter if enabled
            use_cnn = self.use_cnn_chk.isChecked()
            cnn_results = []  # Store CNN results for visualization
            rgb_for_display = None  # Store RGB image for panel 3
            mask_before_cnn = None  # Store mask BEFORE CNN filtering (to show all blobs)

            if use_cnn and CNN_AVAILABLE:
                cnn_model_path = self.cnn_model_edit.text().strip()

                if not cnn_model_path or not os.path.exists(cnn_model_path):
                    logger.warning(f"CNN model not found: {cnn_model_path}")
                    self.status_bar.showMessage(f"CNN model not found: {cnn_model_path}")
                else:
                    # Get RGB image for CNN verification
                    rgb_path = get_rgb_path_from_hsi(self.hdr_path)

                    if rgb_path and os.path.exists(rgb_path):
                        try:
                            # Load RGB image
                            bgr = cv2.imread(rgb_path)
                            if bgr is not None:
                                rgb_for_cnn = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                rgb_for_display = rgb_for_cnn.copy()  # Keep for visualization in RGB format

                                # Initialize CNN verifier
                                logger.info(f"Loading CNN verifier from: {cnn_model_path}")
                                verifier = BlobVerifier(cnn_model_path)

                                # IMPORTANT: Save mask BEFORE CNN filtering (in HSI orientation)
                                binary_mask_hsi = (prob_map >= self.pix_thr)
                                mask_before_cnn = binary_mask_hsi.copy()  # Save for Panel 2 (HSI orientation)

                                # Rotate binary mask to match RGB orientation (90° clockwise)
                                # HSI is "sideways", RGB is "upright"
                                from cnn_verifier import extract_bboxes_from_mask, filter_mask_by_bboxes
                                
                                binary_mask_rgb = cv2.rotate(binary_mask_hsi.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)
                                logger.info(f"Rotated mask from HSI orientation {binary_mask_hsi.shape} to RGB orientation {binary_mask_rgb.shape}")
                                
                                # Extract bounding boxes from ROTATED mask (now aligned with RGB)
                                bboxes = extract_bboxes_from_mask(binary_mask_rgb)
                                
                                logger.info(f"Found {len(bboxes)} blobs to verify with CNN (in RGB coordinates)")

                                # Show CNN patches if toggle is enabled
                                if self.show_cnn_patches_chk.isChecked():
                                    self._show_cnn_patches(rgb_for_cnn, bboxes, padding=10)

                                # Apply CNN verification using RGB image and RGB-oriented bboxes
                                cnn_results = verifier.predict_blobs(
                                    rgb_for_cnn,
                                    bboxes,
                                    padding=10
                                )

                                # Filter the RGB-oriented mask to keep only verified blobs
                                keep_flags = [r['is_grape'] for r in cnn_results]
                                filtered_mask_rgb = filter_mask_by_bboxes(binary_mask_rgb, bboxes, keep_flags)
                                
                                # Rotate filtered mask BACK to HSI orientation (90° counter-clockwise)
                                filtered_mask_hsi = cv2.rotate(filtered_mask_rgb.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
                                logger.info(f"Rotated filtered mask back from RGB {filtered_mask_rgb.shape} to HSI orientation {filtered_mask_hsi.shape}")
                                
                                # Update probability map to keep only CNN-verified blobs (in HSI orientation)
                                prob_map = prob_map * filtered_mask_hsi

                                logger.info("CNN filter applied successfully")
                            else:
                                logger.warning("Failed to load RGB image for CNN")
                        except Exception as e:
                            logger.error(f"CNN filtering failed: {e}", exc_info=True)
                            self.status_bar.showMessage(f"CNN filter error: {e}")
                    else:
                        logger.warning(f"RGB image not found for CNN filtering: {rgb_path}")
                        self.status_bar.showMessage("CNN filter: RGB image not found")
            elif use_cnn and not CNN_AVAILABLE:
                logger.warning("CNN filter enabled but not available")
                self.status_bar.showMessage("CNN filter not available (check dependencies)")

            # Cache results
            self.last_detection_prob_map = prob_map
            self.last_detection_mask = (prob_map >= self.pix_thr)

            # Get base band for overlays
            band_img = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)

            # Panel 1: HSI Band (already displayed via _update_band)
            band_display = cv2.rotate(band_img, cv2.ROTATE_90_CLOCKWISE)
            self._show_image(band_display, self.hsi_band_label, is_grayscale=True)

            # Panel 2: HSI Detection (binary mask with yellow overlay)
            # Show ALL blobs (before CNN filtering) when CNN is used, so you can see both kept and rejected
            if mask_before_cnn is not None:
                # CNN was used - show ALL candidates (before CNN filtering)
                detection_overlay = colorize_binary_mask(band_img, mask_before_cnn)
            else:
                # No CNN - show final result
                detection_overlay = colorize_binary_mask(band_img, self.last_detection_mask)
            
            # Rotate to match the upright HSI display orientation
            detection_overlay = cv2.rotate(detection_overlay, cv2.ROTATE_90_CLOCKWISE)
            detection_rgb = cv2.cvtColor(detection_overlay, cv2.COLOR_BGR2RGB)
            self._show_image(detection_rgb, self.hsi_detection_label)

            # Panel 3: RGB Image with CNN verification results
            # This shows ALL candidates with color-coded boxes (Red=rejected, Green=kept)
            if rgb_for_display is not None and cnn_results is not None and len(cnn_results) > 0:
                # Draw CNN bounding boxes on RGB image
                # No need to pass HSI shape - bboxes are already in RGB coordinates
                logger.info(f"Drawing {len(cnn_results)} CNN bounding boxes on RGB image")
                logger.info(f"RGB shape: {rgb_for_display.shape[:2]}")
                rgb_with_boxes = draw_cnn_bboxes_on_rgb(rgb_for_display, cnn_results, hsi_shape=None)
                self._show_image(rgb_with_boxes, self.hsi_grid_label)
                logger.info("Displayed RGB with CNN annotations in panel 3")
                
                # Log summary of CNN results
                grape_count = sum(1 for r in cnn_results if r['is_grape'])
                noise_count = len(cnn_results) - grape_count
                logger.info(f"CNN Summary: {grape_count} GREEN (grape), {noise_count} RED (noise)")
            else:
                # Fallback: show plain RGB if CNN not used or no results
                rgb_path = get_rgb_path_from_hsi(self.hdr_path)
                if rgb_path and os.path.exists(rgb_path):
                    try:
                        bgr = cv2.imread(rgb_path)
                        if bgr is not None:
                            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            # Display as-is, no rotation
                            self._show_image(rgb_img, self.hsi_grid_label)
                            logger.info("Displayed plain RGB image in panel 3: %s", rgb_path)
                        else:
                            # Show placeholder if load fails
                            self.hsi_grid_label.setText("RGB Image Load Failed")
                            logger.warning("Failed to load RGB image: %s", rgb_path)
                    except Exception as e:
                        self.hsi_grid_label.setText(f"RGB Error: {str(e)}")
                        logger.error("Error displaying RGB: %s", e)
                else:
                    # Show placeholder if not found
                    self.hsi_grid_label.setText("RGB Not Found")
                    logger.warning("RGB image not found for: %s", self.hdr_path)

            # Panel 4: RGB Image (already loaded)
            # No update needed if already displayed

            # Compute grid stats for analysis results (not displayed in UI)
            grid_stats = analyze_grid(prob_map, self.cell_size, self.pix_thr)

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
                f.write(f"  - Shape Filter: {self.use_shape_chk.isChecked()}")
                if self.use_shape_chk.isChecked():
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
        """Save screenshot of all 4 panels."""
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

            # Get pixmaps from all 4 panels
            hsi_band_pm = self.hsi_band_label.pixmap()
            hsi_det_pm = self.hsi_detection_label.pixmap()
            hsi_grid_pm = self.hsi_grid_label.pixmap()
            rgb_pm = self.rgb_label.pixmap()

            if not all([hsi_band_pm, hsi_det_pm, hsi_grid_pm, rgb_pm]):
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

            target_size = (380, 380)
            band_np = cv2.resize(qpixmap_to_np(hsi_band_pm), target_size)
            det_np = cv2.resize(qpixmap_to_np(hsi_det_pm), target_size)
            grid_np = cv2.resize(qpixmap_to_np(hsi_grid_pm), target_size)
            rgb_np = cv2.resize(qpixmap_to_np(rgb_pm), target_size)

            # Concatenate horizontally
            combined = cv2.hconcat([band_np, det_np, grid_np, rgb_np])

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

    def _browse_cnn_model(self):
        """Browse for CNN model file."""
        current_path = self.cnn_model_edit.text()
        start_dir = os.path.dirname(current_path) if current_path else os.path.dirname(__file__)

        path, _ = QFileDialog.getOpenFileName(
            self, "Select CNN Model",
            start_dir,
            "PyTorch Models (*.pth *.pt);;All Files (*)"
        )
        if path:
            self.cnn_model_edit.setText(path)
            logger.info("CNN model selected: %s", path)

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

    def _on_dataset_selection_changed(self):
        """Handle table row selection."""
        selected = self.dataset_table.selectedIndexes()
        if selected and self.dataset_df is not None:
            row_idx = selected[0].row()
            self.dataset_current_index = row_idx
            self._load_dataset_sample(row_idx)
            # Automatically run analysis after loading
            self._run_analysis()

    def _browse_file(self, line_edit: QLineEdit, title: str):
        """Browse for a file and update line edit."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {title}",
            os.path.dirname(line_edit.text()) if line_edit.text() else "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            line_edit.setText(path)

    def _start_grid_search(self):
        """Start the grid search pipeline in background thread."""
        # Validate inputs
        dataset_csv = self.grid_dataset_edit.text().strip()
        if not dataset_csv or not os.path.exists(dataset_csv):
            QMessageBox.warning(
                self,
                "Grid Search",
                f"Dataset CSV not found: {dataset_csv}\nPlease select a valid dataset file."
            )
            return

        # Get selected model
        model_idx = self.grid_model_combo.currentIndex()
        if model_idx < 0 or model_idx >= len(self.available_model_paths):
            QMessageBox.warning(self, "Grid Search", "Please select a model.")
            return

        model_path = self.available_model_paths[model_idx]
        if not os.path.exists(model_path):
            QMessageBox.warning(
                self,
                "Grid Search",
                f"Model not found: {model_path}"
            )
            return

        # Clear log
        self.log_text_edit.clear()
        self.log_text_edit.appendPlainText("="*80)
        self.log_text_edit.appendPlainText("INITIALIZING GRID SEARCH PIPELINE")
        self.log_text_edit.appendPlainText("="*80)
        self.log_text_edit.appendPlainText(f"Dataset: {dataset_csv}")
        self.log_text_edit.appendPlainText(f"Model: {model_path}")
        self.log_text_edit.appendPlainText("")

        # Create and configure worker
        self.grid_search_worker = GridSearchWorker(
            dataset_csv=dataset_csv,
            model_path=model_path,
            output_dir=os.path.dirname(__file__)
        )

        # Connect signals
        self.grid_search_worker.log_signal.connect(self._append_log)
        self.grid_search_worker.progress_signal.connect(self._update_grid_progress)
        self.grid_search_worker.finished_signal.connect(self._on_grid_search_finished)

        # Update UI
        self.start_grid_btn.setEnabled(False)
        self.stop_grid_btn.setEnabled(True)
        self.grid_progress.setVisible(True)
        self.grid_progress.setValue(0)
        self.status_bar.showMessage("Grid search running...")

        # Start worker
        self.grid_search_worker.start()

        logger.info("Grid search pipeline started")

    def _stop_grid_search(self):
        """Stop the running grid search."""
        if self.grid_search_worker and self.grid_search_worker.isRunning():
            self.log_text_edit.appendPlainText("")
            self.log_text_edit.appendPlainText("="*80)
            self.log_text_edit.appendPlainText("STOPPING GRID SEARCH...")
            self.log_text_edit.appendPlainText("="*80)
            self.grid_search_worker.stop()
            self.grid_search_worker.wait(5000)  # Wait up to 5 seconds

            self.start_grid_btn.setEnabled(True)
            self.stop_grid_btn.setEnabled(False)
            self.status_bar.showMessage("Grid search stopped by user")

    def _update_grid_progress(self, current: int, total: int, message: str):
        """Update progress bar and status."""
        if total > 0:
            self.grid_progress.setMaximum(total)
            self.grid_progress.setValue(current)
        self.status_bar.showMessage(message)

    def _on_grid_search_finished(self, success: bool, message: str):
        """Handle grid search completion."""
        self.start_grid_btn.setEnabled(True)
        self.stop_grid_btn.setEnabled(False)
        self.grid_progress.setVisible(False)

        if success:
            self.status_bar.showMessage("✓ " + message)

            # Show success dialog
            reply = QMessageBox.information(
                self,
                "Grid Search Complete",
                f"{message}\n\nWould you like to load the best parameters now?",
                QMessageBox.Yes | QMessageBox.No
            )

            # Auto-load best parameters if user agrees
            if reply == QMessageBox.Yes:
                self._load_best_grid_result()
        else:
            self.status_bar.showMessage("✗ " + message)
            QMessageBox.warning(self, "Grid Search Failed", message)

    def _load_dataset_sample(self, index: int):
        """Load a specific sample from the dataset."""
        if self.dataset_df is None or index < 0 or index >= len(self.dataset_df):
            return

        try:
            row = self.dataset_df.iloc[index]
            grape_id = row["grape_id"]
            image_path = row["image_path"]
            label = row["label"]

            # Update info label
            self.dataset_info_label.setText(
                f"Sample {index+1}/{len(self.dataset_df)}: {grape_id} (label={label})"
            )

            # Load images from this path
            if os.path.exists(image_path):
                self._load_images(image_path)
                logger.info("Loaded dataset sample: %s (index=%d)", grape_id, index)
            else:
                logger.warning("Image path not found: %s", image_path)
                self.status_bar.showMessage(f"Warning: Path not found: {image_path}")

        except Exception as e:
            logger.exception("Failed to load dataset sample %d: %s", index, e)
            QMessageBox.warning(self, "Dataset", f"Failed to load sample: {e}")

    def _dataset_next(self):
        """Navigate to next dataset sample."""
        if self.dataset_df is None:
            return

        if self.dataset_current_index < len(self.dataset_df) - 1:
            self.dataset_current_index += 1
            self.dataset_table.selectRow(self.dataset_current_index)
            self._load_dataset_sample(self.dataset_current_index)
            # Automatically run analysis after loading
            self._run_analysis()
        else:
            self.status_bar.showMessage("Already at last sample")

    def _dataset_prev(self):
        """Navigate to previous dataset sample."""
        if self.dataset_df is None:
            return

        if self.dataset_current_index > 0:
            self.dataset_current_index -= 1
            self.dataset_table.selectRow(self.dataset_current_index)
            self._load_dataset_sample(self.dataset_current_index)
            # Automatically run analysis after loading
            self._run_analysis()
        else:
            self.status_bar.showMessage("Already at first sample")


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("HSI.Patch.UI")
    try:
        logger.info("Starting HSI Late Detection UI...")
        app = QApplication(sys.argv)
        window = HSILateDetectionViewer()
        window.show()
        logger.info("UI started. Entering event loop.")
        sys.exit(app.exec_())
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        print(f"Fatal error: {e}")

