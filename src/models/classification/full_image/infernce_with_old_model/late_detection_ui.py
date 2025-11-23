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
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QStatusBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QInputDialog,
    QMessageBox, QLineEdit, QGroupBox, QTableWidget, QTableWidgetItem,
    QPlainTextEdit, QProgressBar, QTabWidget, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QTextCursor

# Import core functions
from late_detection_core import (
    load_cube, load_model_and_scaler, per_pixel_probs, analyze_grid,
    filter_blobs_by_shape, filter_blobs_advanced, find_band_index_for_wavelength, find_scaler,
    GRID_COLOR_BUCKETS
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
from pathlib import Path

def find_project_root(start_path: Optional[str] = None) -> str:
    """Try to locate the project root by walking up from the current file.

    Heuristics: a folder that contains 'requirements.txt', a 'data' folder, or a '.git' folder.
    Falls back to a fixed number of parents if heuristics fail.
    """
    try:
        p = Path(start_path or __file__).resolve()
    except Exception:
        p = Path(__file__).resolve()

    for parent in [p] + list(p.parents):
        if (parent / 'requirements.txt').exists() or (parent / 'data').exists() or (parent / '.git').exists():
            return str(parent)

    # Best effort fallback: go up 5 levels from this file (suits this repo layout)
    try:
        return str(p.parents[5])
    except Exception:
        return str(p.parent)

# Determine project root (allow override through env var)
PROJECT_ROOT = os.environ.get('GRAPE_PROJECT_ROOT', find_project_root())

AVAILABLE_MODELS = {
    "OLD LDA [1=CRACK, 0=regular]": os.path.join(PROJECT_ROOT, "src", "models", "classification", "pixel_level", "simple_classification_leave_one_out", "comare_all_models", "models", "LDA_Balanced.pkl"),
    # "NEW LDA Multi-class": os.path.join(PROJECT_ROOT, "src", "models", "classification", "full_image", "classification_by_pixel", "Train", "LDA", "lda_model_multi_class.joblib"),  # Empty - needs retraining
}
DEFAULT_SEARCH_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_DATASET_CSV = os.path.join(PROJECT_ROOT, "src", "preprocessing","prepare_dataset", "hole_image", "late_detection", "late_detection_dataset.csv")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "Results")


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
            def objective(trial: optuna.Trial) -> float:
                # Suggest parameters
                prob_thr = trial.suggest_float("prob_thr", 0.80, 0.99)
                morph_size = trial.suggest_categorical("morph_size", [0] + list(range(1, 16, 2)))
                min_blob_size = trial.suggest_int("min_blob_size", 10, 1000)

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
                    aspect_ratio_min = trial.suggest_float("aspect_ratio_min", 1.0, 3.0)
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

            # When finished, emit best params/value
            best = study.best_trial
            best_info = {
                "best_value": float(best.value) if best.value is not None else None,
                "best_params": dict(best.params)
            }
            self.log(f"Optuna complete: best F2={best_info['best_value']}")
            self.finished_signal.emit(True, best_info)

        except Exception as e:
            self.log(f"OptunaWorker failed: {e}")
            self.finished_signal.emit(False, {"error": str(e)})


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
        # Advanced filter params (UI-controlled)
        self.morph_size = 3
        self.solidity_max = 1.0
        self.aspect_ratio_max = 5.0

        # Grid search worker
        self.grid_search_worker: Optional[GridSearchWorker] = None
        self.optuna_worker: Optional[OptunaWorker] = None

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

        # Panel 3: HSI Patch Grid
        self.hsi_grid_label = QLabel("HSI Patch Grid")
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
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(self.pix_thr)
        self.thr_spin.valueChanged.connect(self._update_thr)
        top_row.addWidget(self.thr_spin)

        self.invert_class_chk = QCheckBox("Invert class")
        self.invert_class_chk.setChecked(False)
        top_row.addWidget(self.invert_class_chk)

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
        self.circ_max_spin.setDecimals(2)
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
        self.solidity_max_spin.setDecimals(2)
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
        self.circ_min_spin.setDecimals(2)
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
        self.solidity_min_spin.setDecimals(2)
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

            # Apply filters if any are enabled (all filters are independent)
            use_border = self.use_border_chk.isChecked()
            use_area = self.use_area_chk.isChecked()
            use_circ = self.use_circ_chk.isChecked()
            use_ar = self.use_ar_chk.isChecked()
            use_sol = self.use_sol_chk.isChecked()

            if use_border or use_area or use_circ or use_ar or use_sol or (getattr(self, 'morph_size', 0) > 0):
                filter_params = dict(
                    morph_size=int(getattr(self, 'morph_size', 0)),
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
                logger.info("Applying advanced filters: %s", filter_params)
                # Use the advanced filter which supports morphological closing and geometric tests
                mask_filtered = filter_blobs_advanced(prob_map, self.pix_thr, **filter_params)
                prob_map = prob_map * mask_filtered

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
            detection_overlay = colorize_binary_mask(band_img, self.last_detection_mask)
            detection_overlay = cv2.rotate(detection_overlay, cv2.ROTATE_90_CLOCKWISE)
            detection_rgb = cv2.cvtColor(detection_overlay, cv2.COLOR_BGR2RGB)
            self._show_image(detection_rgb, self.hsi_detection_label)

            # Panel 3: HSI Patch Grid
            grid_stats = analyze_grid(prob_map, self.cell_size, self.pix_thr)
            grid_overlay = overlay_on_band(band_img, grid_stats, alpha=0.45)
            grid_overlay = cv2.rotate(grid_overlay, cv2.ROTATE_90_CLOCKWISE)
            grid_rgb = cv2.cvtColor(grid_overlay, cv2.COLOR_BGR2RGB)
            self._show_image(grid_rgb, self.hsi_grid_label)

            # Panel 4: RGB Image (already loaded)
            # No update needed if already displayed

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

