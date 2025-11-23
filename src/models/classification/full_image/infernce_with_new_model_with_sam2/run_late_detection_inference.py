import os, sys, csv, logging, time
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import argparse
from pathlib import Path

import numpy as np
import joblib
import cv2
import spectral as spy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QStatusBar,
    QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QComboBox, QInputDialog,
    QGroupBox, QPlainTextEdit, QTableWidget, QTableWidgetItem, QMessageBox,
    QProgressDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QObject, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

# New imports for batch processing and metrics
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import glob
import itertools
from joblib import Parallel, delayed

# SAM2 imports for segmentation
project_path = Path(__file__).resolve().parents[4]
sys.path.append(str(project_path))
from src.preprocessing.MaskGenerator.segment_object_module import create_point_segmenter
from src.preprocessing.MaskGenerator.mask_generator_module import (
    initial_settings,
    initialize_sam2_predictor,
)

# ===== Hyperparameter search config =====
# Ranges for PLA hyperparameters (grid-based aggregation)
GRID_SIZE_CANDIDATES = [16, 32, 48, 64]
GRID_CRACK_RATIO_THR_CANDIDATES = [0.05, 0.10, 0.15, 0.20]  # per-patch crack_fraction threshold
CLUSTER_CRACK_RATIO_THR_CANDIDATES = [0.02, 0.05, 0.10, 0.15, 0.20]  # threshold on cluster-level crack_ratio
METRICS_OUTPUT_CSV = "late_detection_hparam_metrics.csv"

# ===== Dataset Runner config =====
DEFAULT_LATE_DETECTION_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset\hole_image\late_detection\late_detection_dataset.csv"
DATASET_RESULTS_CSV = "late_detection_ui_results.csv"
DATASET_METRICS_CSV = "late_detection_ui_metrics.csv"

# Default pixel threshold (used to compute initial crack_ratio per cluster)
DEFAULT_CELL_SIZE = 64
DEFAULT_PIX_THR = 0.5

# ===== Model shims (for pickles that reference __main__.ModelClass) =====
class LDAModel:
    """Shim for legacy pickles. Delegates predict_proba to an inner estimator."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)
    def _inner(self):
        for attr in ("model", "clf", "estimator_", "lda", "classifier", "classifier_"):
            est = getattr(self, attr, None)
            if est is not None:
                return est
        return None
    def predict_proba(self, X):
        est = self._inner()
        if est is None or not hasattr(est, "predict_proba"):
            raise AttributeError("LDAModel shim: no inner estimator with predict_proba")
        return est.predict_proba(X)

# Shim for legacy DecisionTreeModel pickles (some models use a custom thin wrapper)
class DecisionTreeModel:
    """Minimal shim: restore state and delegate predict_proba to inner estimator if present."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        # store state as attributes so downstream code can access underlying estimator
        self.__dict__.update(state)
    def _inner(self):
        for attr in ("model", "clf", "estimator_", "tree", "classifier", "classifier_"):
            est = getattr(self, attr, None)
            if est is not None:
                return est
        return None
    def predict_proba(self, X):
        est = self._inner()
        if est is None or not hasattr(est, "predict_proba"):
            raise AttributeError("DecisionTreeModel shim: no inner estimator with predict_proba")
        return est.predict_proba(X)


# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("HSI.Patch")
if os.environ.get("HSI_DEBUG") == "1": logger.setLevel(logging.DEBUG)

# ===== Config =====
AVAILABLE_MODELS = {
    "NEW LDA Multi-class": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\lda_model_multi_class.joblib",
    "OLD LDA  [1=CRACK, 0=regular]": r"C:\Users\yovel\Desktop\Grape_Project\src\models\classification\pixel_level\simple_classification_leave_one_out\comare_all_models\models\LDA_Balanced.pkl",
}
DEFAULT_LDA_PATH = list(AVAILABLE_MODELS.values())[0]
DEFAULT_SEARCH_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"
GRID_COLOR_BUCKETS = [(50, (0, 0, 180)), (40, (0, 30, 200)), (30, (0, 120, 255)), (20, (0, 200, 255))]

# ===== Dynamic palette generation =====
def build_dynamic_palette(n: int) -> List[Tuple[int, int, int]]:
    if n <= 0: return []
    base_palette_bgr = [
        (0, 255, 255), (0, 230, 255), (0, 215, 255), (0, 200, 255),
        (0, 140, 255), (0, 90, 255), (0, 50, 240), (0, 30, 200), (0, 0, 180)
    ]
    if n <= len(base_palette_bgr):
        if n == 1: return [base_palette_bgr[-1]]
        indices = np.linspace(0, len(base_palette_bgr) - 1, n, dtype=int)
        return [base_palette_bgr[i] for i in indices]
    else:
        logger.warning(f"Building palette for {n} thresholds (>9 recommended max). Colors may look similar.")
        result = []
        for i in range(n):
            pos = i * 8.0 / (n - 1)
            idx_low = int(np.floor(pos))
            idx_high = min(idx_low + 1, len(base_palette_bgr) - 1)
            frac = pos - idx_low
            color_low = base_palette_bgr[idx_low]
            color_high = base_palette_bgr[idx_high]
            b = int(color_low[0] * (1 - frac) + color_high[0] * frac)
            g = int(color_low[1] * (1 - frac) + color_high[1] * frac)
            r = int(color_low[2] * (1 - frac) + color_high[2] * frac)
            result.append((b, g, r))
        return result

# ===== Small helpers =====

def _predict_proba_any(model, X):
    """Try common attributes to find a classifier with predict_proba."""
    for attr in (None, "estimator_", "clf", "model", "lda", "tree", "classifier", "classifier_"):
        m = getattr(model, attr, model) if attr else model
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)
    raise AttributeError("No predict_proba on provided model")


def _classes_any(model):
    # helper: robustly fetch class labels from the model/pipeline
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    # If wrapped in a pipeline, try to find final estimator
    est = getattr(model, "named_steps", {}).get("model", None) or getattr(model, "model", None)
    if est is not None and hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    # safe fallback
    return np.asarray([0, 1])

def _get_model_expected_features(model) -> Optional[int]:
    """Try to infer number of input features the model expects.
    Look for n_features_in_ on possible underlying estimators or coef_ shape.
    Returns None if unknown.
    """
    for attr in (None, "estimator_", "clf", "model", "lda", "tree", "classifier", "classifier_"):
        m = getattr(model, attr, model) if attr else model
        if hasattr(m, "n_features_in_"):
            try:
                return int(m.n_features_in_)
            except Exception:
                pass
        if hasattr(m, "coef_"):
            try:
                coef = getattr(m, "coef_")
                # coef_ may be (n_classes-1, n_features) or (n_features,) depending on model
                if hasattr(coef, "shape") and len(coef.shape) >= 1:
                    return int(coef.shape[-1])
            except Exception:
                pass
    return None


def find_scaler(path_dir: str, exclude: Optional[List[str]] = None) -> Optional[str]:
    exclude = set(exclude or [])
    cands = [f for f in os.listdir(path_dir) if f.lower().endswith((".joblib", ".pkl")) and f not in exclude]
    for name in cands:  # prefer names that look like a scaler
        if any(k in name.lower() for k in ("scaler", "standard", "std")):
            p = os.path.join(path_dir, name)
            try:
                if hasattr(joblib.load(p), "transform"): return p
            except Exception:
                pass
    # fallback: any obj with transform
    for name in cands:
        p = os.path.join(path_dir, name)
        try:
            if hasattr(joblib.load(p), "transform"): return p
        except Exception:
            pass
    return None


def load_model_and_scaler(lda_path: str, scaler_path: Optional[str] = None) -> Tuple[object, Optional[object], int, np.ndarray]:
    logger.info("Loading model: %s", lda_path)
    try:
        import types
        main_mod = sys.modules.get("__main__")
        if main_mod is None:
            main_mod = types.ModuleType("__main__")
            sys.modules["__main__"] = main_mod
        # Inject shims so joblib/pickle can resolve custom classes referencing __main__
        for cls in (LDAModel, DecisionTreeModel):
            setattr(main_mod, cls.__name__, cls)
    except Exception:
        logger.debug("Failed to inject shim classes into __main__ (non-fatal)")

    try:
        lda = joblib.load(lda_path)
    except AttributeError as e:
        if "Can't get attribute" in str(e):
            model_name = os.path.basename(lda_path)
            logger.error(f"Failed to load {model_name}: Model was trained with a custom class not available here.")
            logger.error(f"This model is not compatible. Please use one of the whitelisted LDA models.")
            raise ValueError(f"Model {model_name} is not compatible - use LDA models only") from e
        raise

    if scaler_path is None:
        scaler_path = find_scaler(os.path.dirname(lda_path), [os.path.basename(lda_path)])
    scaler = joblib.load(scaler_path) if scaler_path else None
    classes = _classes_any(lda)

    pos_idx = 0
    if len(classes) > 0 and isinstance(classes[0], str):
        crack_candidates = ['CRACK', 'crack', 'Crack', 'DECAY', 'decay']
        for i, cls in enumerate(classes):
            if cls in crack_candidates:
                pos_idx = i
                break

    logger.info("Model loaded. classes=%s | pos_idx=%d (CRACK/decay class) | scaler=%s", classes.tolist(), pos_idx, bool(scaler))
    expected = _get_model_expected_features(lda)
    logger.info("Model expected input features: %s", expected)
    return lda, scaler, pos_idx, classes


def load_cube(hdr_path: str) -> np.ndarray:
    t0 = time.perf_counter()
    dat_path = hdr_path.replace(".hdr", ".dat")
    logger.info(f"[DEBUG] Loading HSI cube: hdr_path={hdr_path}, dat_path={dat_path}")
    cube = np.array(spy.envi.open(hdr_path, dat_path).load())
    logger.info(f"[DEBUG] Loaded cube shape: {cube.shape}, dtype: {cube.dtype}")
    if cube.ndim != 3 or cube.shape[2] != 204:
        logger.error(f"[ERROR] Loaded cube shape is {cube.shape}, expected (H, W, 204). Check your .hdr/.dat files and data source.")
    logger.info("Loaded cube: %s (%.2fs) shape=%s", hdr_path, time.perf_counter() - t0, tuple(cube.shape))
    return cube


def per_pixel_probs(cube: np.ndarray, lda: object, scaler: Optional[object], pos_idx: int) -> np.ndarray:
    t0 = time.perf_counter()
    H, W, C = cube.shape
    X_raw = np.nan_to_num(cube.reshape(-1, C), copy=False)
    logger.info(f"[DEBUG] Using C bands for classification: C={C}, X.shape={X_raw.shape}")

    # Prepare candidate feature preps and test which one the model accepts using a small sample
    candidates = [("raw", X_raw[:min(200, X_raw.shape[0])])]
    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            X_scaled_sample = scaler.transform(X_raw[:min(200, X_raw.shape[0])])
            candidates.append(("scaled", X_scaled_sample))
            logger.info(f"[DEBUG] Scaler transform sample shape: {X_scaled_sample.shape}")
        except Exception as e:
            logger.warning(f"[DEBUG] Scaler transform failed on sample: {e}. Not using scaler sample.")
    else:
        logger.info(f"[DEBUG] No scaler available.")

    chosen = None
    chosen_name = None
    for name, sample_mat in candidates:
        try:
            # Try to predict on a small sample; if it doesn't raise, assume this prep works
            _predict_proba_any(lda, sample_mat)
            chosen = name
            chosen_name = name
            logger.info(f"[DEBUG] Using feature prep: {chosen}")
            break
        except Exception as e:
            logger.debug(f"[DEBUG] Candidate '{name}' rejected: {e}")

    if chosen is None:
        logger.warning("No compatible feature preparation found. Falling back to raw features and attempting prediction.")
        chosen = "raw"

    # Apply the chosen preparation to the full dataset
    X = X_raw
    if chosen == "scaled":
        try:
            X = scaler.transform(X_raw)
            logger.info(f"[DEBUG] Scaler applied. X.shape after scaling: {X.shape}")
        except Exception as e:
            logger.warning(f"[DEBUG] Failed to apply scaler to full X: {e}. Falling back to raw features.")
            X = X_raw

    # Finally predict probabilities (robust to different model wrappers)
    try:
        probs = _predict_proba_any(lda, X)
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=-1)
        if pos_idx >= probs.shape[1]:
            logger.warning(f"pos_idx {pos_idx} out of bounds for probs shape {probs.shape}, using last index.")
            pos_idx = probs.shape[1] - 1
        probs = probs[:, pos_idx].reshape(H, W)
    except Exception as e:
        logger.error(f"Error in per_pixel_probs: {e}")
        probs = np.zeros((H, W), dtype=np.float32)
    logger.info("Per-pixel probs: time=%.2fs | min=%.4f max=%.4f mean=%.4f", time.perf_counter()-t0, float(probs.min()), float(probs.max()), float(probs.mean()))
    return probs


def iter_grid(H: int, W: int, cell: int):
    for r0 in range(0, H, cell):
        r1 = min(r0 + cell, H)
        for c0 in range(0, W, cell):
            c1 = min(c0 + cell, W)
            yield r0, c0, r1, c1


def analyze_grid(prob_map: np.ndarray, cell: int, pix_thr: float) -> List[Dict]:
    H, W = prob_map.shape
    logger.info("Analyzing grid: H=%d W=%d cell=%d thr=%.2f", H, W, cell, pix_thr)
    out = []
    for r0, c0, r1, c1 in iter_grid(H, W, cell):
        patch = prob_map[r0:r1, c0:c1]
        n = patch.size
        k = int((patch >= pix_thr).sum())
        pct = 100.0 * k / max(1, n)
        out.append({
            "row0": r0, "col0": c0, "row1": r1, "col1": c1,
            "cell": cell, "area": n, "count_cracked": k,
            "percent_cracked": pct, "pix_thr": pix_thr
        })
    percents = [c["percent_cracked"] for c in out]
    mean_p = float(np.mean(percents)) if percents else 0.0
    p50 = float(np.median(percents)) if percents else 0.0
    p90 = float(np.percentile(percents, 90)) if percents else 0.0
    logger.info("Grid summary: patches=%d | mean=%.2f%% p50=%.2f%% p90=%.2f%%", len(out), mean_p, p50, p90)
    return out


def color_for_percent(p: float):
    for thr, col in GRID_COLOR_BUCKETS:
        if p >= thr:
            return col
    return None


def overlay_on_band(band_img: np.ndarray, grid_stats: List[Dict], alpha: float = 0.35) -> np.ndarray:
    # Convert grayscale band to RGB background
    rgb = cv2.cvtColor(band_img, cv2.COLOR_GRAY2RGB)
    over = rgb.copy()
    # Ensure buckets are in descending order (highest ratio -> first)
    # color_for_percent already expects GRID_COLOR_BUCKETS descending
    for c in grid_stats:
        col = color_for_percent(c["percent_cracked"])
        if col:
            r0, c0, r1, c1 = c["row0"], c["col0"], c["row1"], c["col1"]
            # Fill patch with bucket color
            cv2.rectangle(over, (c0, r0), (c1 - 1, r1 - 1), col, -1)
            # Draw a thin border so patches remain visible over background
            cv2.rectangle(over, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 0), 1)
    # Blend overlay on top of background. Increase alpha for clearer visual weight.
    merged = cv2.addWeighted(over, max(alpha, 0.45), rgb, 1 - max(alpha, 0.45), 0)
    return merged

# ===== Legend rendering (outside the image) =====

def render_legend_image(buckets: List[Tuple[int, Tuple[int, int, int]]]) -> np.ndarray:
    # Create a compact vertical legend image (RGB). Minimal width/height to save space.
    w, h, pad = 120, 80, 5
    img = np.full((h, w, 3), 245, np.uint8)
    x, y = 6, 14
    box_w, box_h = 18, 11
    # Ensure buckets are sorted descending (highest first)
    buckets_sorted = sorted(buckets, key=lambda x: x[0], reverse=True)
    for thr, bgr in buckets_sorted:
        rgb = (bgr[2], bgr[1], bgr[0])
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), rgb, -1)
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), 1)
        label = f">={thr}%"
        cv2.putText(img, label, (x + box_w + 3, y + box_h - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
        y += box_h + pad
    # Title: minimal
    title = "Crack %"
    cv2.putText(img, title, (6, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
    return img

# ===== CSV export =====

def export_grid_csv(grid_stats: List[Dict], context: Dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = f"grid_percent_{context['folder_main']}_{context['folder_date']}_b{context['band']}_c{context['cell']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    fp = os.path.join(out_dir, fn)
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id","folder_main","folder_date","band","cell","pix_thr","row0","col0","row1","col1","area","count_cracked","percent_cracked"])
        for c in grid_stats:
            w.writerow([context.get("image_id",""),context.get("folder_main",""),context.get("folder_date",""),context.get("band",""),c["cell"],c["pix_thr"],c["row0"],c["col0"],c["row1"],c["col1"],c["area"],c["count_cracked"],c["percent_cracked"]])
    logger.info("CSV exported: %s (rows=%d)", fp, len(grid_stats))
    return fp

# ============================
# SAM2 Multi-Point Segmentation Functions
# ============================

def extract_blob_centroids(binary_mask: np.ndarray, max_blobs: int = 100) -> List[Tuple[int, int]]:
    """
    Extract centroid coordinates from all connected components in a binary mask.

    Args:
        binary_mask: Binary detection mask (0 or 255/True/False)
        max_blobs: Maximum number of blobs to process (largest by area)

    Returns:
        List of (x, y) tuples representing blob centroids
    """
    # Ensure binary mask is uint8
    if binary_mask.dtype == bool:
        mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    else:
        mask_uint8 = binary_mask.astype(np.uint8)

    # Find connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    # Skip background (label 0)
    if num_labels <= 1:
        return []

    # Extract areas and sort by size (descending)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    sorted_indices = np.argsort(-areas)  # Descending order

    # Limit to max_blobs
    sorted_indices = sorted_indices[:max_blobs]

    # Extract centroids (x, y) for selected blobs
    centroid_points = []
    for idx in sorted_indices:
        label_idx = idx + 1  # Offset for background
        cx, cy = centroids[label_idx]
        centroid_points.append((int(cx), int(cy)))

    logger.info(f"Extracted {len(centroid_points)} blob centroids from {num_labels-1} total blobs")
    return centroid_points


def segment_multiple_objects_with_sam(
    rgb_image: np.ndarray,
    points: List[Tuple[int, int]],
    sam2_segmenter
) -> List[np.ndarray]:
    """
    Segment multiple objects using SAM with point prompts.

    Args:
        rgb_image: RGB image array (H, W, 3)
        points: List of (x, y) tuples for object centroids
        sam2_segmenter: Initialized PointSegmenter instance

    Returns:
        List of boolean masks, one per input point
    """
    if not points:
        return []

    masks = []
    # Note: Current PointSegmenter expects image_path, so we need to save temp image
    # Alternative: modify to accept numpy array directly
    # For now, segment each point individually
    for i, point in enumerate(points):
        try:
            # SAM expects points as list even for single point
            _, mask_bool = sam2_segmenter.segment_object_from_array(rgb_image, [point])
            masks.append(mask_bool)
        except Exception as e:
            logger.warning(f"Failed to segment point {i} at {point}: {e}")
            # Add empty mask as placeholder
            masks.append(np.zeros(rgb_image.shape[:2], dtype=bool))

    logger.info(f"Successfully segmented {len(masks)} objects with SAM")
    return masks


def create_sam_segment_overlay(
    base_image: np.ndarray,
    masks: List[np.ndarray],
    prob_map: Optional[np.ndarray] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create colored overlay showing SAM segments on base image.

    Args:
        base_image: Grayscale base image (H, W) 0-255
        masks: List of boolean masks from SAM segmentation
        prob_map: Optional probability map to color segments by intensity
        alpha: Overlay transparency (0-1)

    Returns:
        RGB overlay image with colored segments
    """
    # Convert base to RGB
    rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB) if len(base_image.shape) == 2 else base_image.copy()
    overlay = rgb.copy()

    # Color palette for segments (cycling through distinct colors)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Light green
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light blue
    ]

    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue

        # Choose color based on index (cycle through palette)
        color = np.array(colors[i % len(colors)], dtype=np.uint8)

        # If prob_map provided, modulate color intensity by average probability
        if prob_map is not None:
            avg_prob = np.mean(prob_map[mask])
            intensity = np.clip(avg_prob, 0.3, 1.0)  # Min 30% intensity
            color = (color * intensity).astype(np.uint8)

        # Apply color to mask region
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

        # Draw contour for better visibility
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

    return overlay.astype(np.uint8)


# ============================
# add: compact blob-shape filter with configurable operators
# ============================
# ===== Compact border+area-only blob filter =====
def filter_blobs_by_shape(proba, thr, *, border_r=20, area_min=10, area_max=5000,
                          use_border=True, use_area=True):
    import cv2, numpy as np
    H, W = proba.shape
    mask = ((proba >= thr).astype("uint8") * 255)
    out = np.zeros_like(mask)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        A = cv2.contourArea(c)
        if use_area and (A < area_min or A > area_max):
            continue
        x, y, w, h = cv2.boundingRect(c)
        near_border = (x <= border_r or y <= border_r or (x + w) >= (W - 1 - border_r) or (y + h) >= (H - 1 - border_r))
        if use_border and near_border:
            continue
        cv2.drawContours(out, [c], -1, 255, -1)
    return (out > 0)

# ===== Pipeline =====

def run_patch_analysis(
    hdr_path: str,
    lda_path: str,
    scaler_path: Optional[str],
    band: int = 0,
    cell_size: int = 64,
    pix_thr: float = 0.5,
    out_dir: str = "grid_results",
    pos_idx: Optional[int] = None,
    reuse_prob_map: Optional[np.ndarray] = None,
) -> Tuple[str, str]:
    logger.info("Params -> band=%d cell=%d thr=%.2f", band, cell_size, pix_thr)
    lda, scaler, default_pos_idx, classes = load_model_and_scaler(lda_path, scaler_path)

    # Use provided pos_idx if available, otherwise use default
    if pos_idx is None:
        pos_idx = default_pos_idx

    cube = load_cube(hdr_path)

    # Decide whether to reuse prob_map or compute fresh
    if reuse_prob_map is not None:
        logger.info("Patch analysis: reusing provided prob_map from detection-only")
        prob_map = reuse_prob_map.copy()
    else:
        prob_map = per_pixel_probs(cube, lda, scaler, pos_idx)

    # Invert class if requested via filter_params
    f = getattr(run_patch_analysis, "filter_params", None)
    if f and f.get("invert", False):
        logger.info("Inverting probabilities (invert flag set)")
        prob_map = 1.0 - prob_map

    # optional border+area filtering - ONLY if shape_enabled is True
    if f and f.get("shape_enabled", False):
        try:
            logger.info("Applying shape filter -> border_r=%d area_min=%d area_max=%d", f.get("border_r",20), f.get("area_min",10), f.get("area_max",5000))
            mask = filter_blobs_by_shape(prob_map, pix_thr, border_r=int(f.get("border_r",20)), area_min=int(f.get("area_min",10)), area_max=int(f.get("area_max",5000)), use_border=bool(f.get("use_border", True)), use_area=bool(f.get("use_area", True)))
            prob_map = prob_map * mask
        except Exception:
            logger.exception("Shape filter failed; continuing without it")
    else:
        logger.info("Shape filter not enabled, skipping border/area filtering")

    grid_stats = analyze_grid(prob_map, cell_size, pix_thr)

    band_img = cv2.normalize(cube[:, :, band], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay_img = overlay_on_band(band_img, grid_stats)

    os.makedirs(out_dir, exist_ok=True)
    overlay_path = os.path.join(out_dir, "overlay.png")
    cv2.imwrite(overlay_path, overlay_img)
    logger.info("Overlay saved: %s", overlay_path)

    folder_main = os.path.basename(os.path.dirname(hdr_path))
    folder_date = os.path.basename(os.path.dirname(os.path.dirname(hdr_path)))
    ctx = {"image_id": os.path.splitext(os.path.basename(hdr_path))[0], "folder_main": folder_main, "folder_date": folder_date, "band": band, "cell": cell_size, "pix_thr": pix_thr}
    csv_path = export_grid_csv(grid_stats, ctx, out_dir)
    return overlay_path, csv_path

# ===== Threaded worker =====
class AnalysisThread(QThread):
    finished_signal = pyqtSignal(object, str)  # (overlay_rgb_array or None, csv_path)
    def __init__(self, hdr_path, lda_path, scaler_path, band, cell_size, pix_thr, out_dir, pos_idx=None):
        super().__init__()
        self.hdr_path, self.lda_path, self.scaler_path = hdr_path, lda_path, scaler_path
        self.band, self.cell_size, self.pix_thr, self.out_dir = band, cell_size, pix_thr, out_dir
        self.pos_idx = pos_idx
    def run(self):
        try:
            t0 = time.perf_counter()
            overlay_path, csv_path = run_patch_analysis(self.hdr_path, self.lda_path, self.scaler_path, self.band, self.cell_size, self.pix_thr, self.out_dir, self.pos_idx)
            bgr = cv2.imread(overlay_path)
            if bgr is None:
                logger.warning("Overlay not found after analysis: %s", overlay_path)
                self.finished_signal.emit(None, csv_path); return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            logger.info("Analysis done in %.2fs", time.perf_counter() - t0)
            self.finished_signal.emit(rgb, csv_path)
        except Exception as e:
            logger.exception("AnalysisThread failed: %s", e)
            self.finished_signal.emit(None, "")

# ===== Small helpers for detection-only visualization =====
def colorize_binary_mask(gray, mask):
    import numpy as np, cv2
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = np.array([0, 215, 255], dtype=np.uint8)  # yellow
    alpha = 0.35
    rgb[mask] = (alpha*color + (1-alpha)*rgb[mask]).astype(np.uint8)
    return rgb

# ===== UI =====
class HSILDAViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI LDA Patch Classifier (All-in-One + UI Legend)")
        self.setGeometry(100, 100, 1250, 640)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)

        self.rgb_image: Optional[np.ndarray] = None
        self.hsi_cube: Optional[np.ndarray] = None
        # ...existing code...
        self.current_band: int = 0
        self.folder_path: str = ""
        self.hdr_path: Optional[str] = None

        # For navigation
        self.current_cluster_id: Optional[str] = None
        self.current_date: Optional[str] = None
        self.available_clusters: List[str] = []

        self.lda_path: Optional[str] = DEFAULT_LDA_PATH if os.path.exists(DEFAULT_LDA_PATH) else None
        self.scaler_path: Optional[str] = None

        # Available models - direct paths (no directory selection needed)
        self.available_model_names: List[str] = list(AVAILABLE_MODELS.keys())
        self.available_model_paths: List[str] = list(AVAILABLE_MODELS.values())

        # Multi-class support
        self.current_pos_idx: int = 0  # Current positive class index
        self.model_classes: np.ndarray = np.array([])  # Classes from loaded model

        # Store latest detection results for grid visualization
        self.last_detection_prob_map: Optional[np.ndarray] = None
        self.last_detection_mask: Optional[np.ndarray] = None
        self.last_params: Dict = {}  # comprehensive params dict for reuse detection

        # SAM2 segmentation infrastructure
        self.sam2_segmenter = None
        self.last_sam_segments: List[np.ndarray] = []  # Cache SAM segment masks
        self.last_sam_overlay: Optional[np.ndarray] = None  # Cache SAM overlay

        # Dataset runner state
        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_current_index: int = -1
        self.dataset_running: bool = False
        self.dataset_results: Optional[pd.DataFrame] = None
        self.dataset_subset: str = "both"  # 'dev','test','both'
        self.dataset_auto_display: bool = True

        # runtime params
        self.cell_size = 64
        self.pix_thr = 0.98  # Default threshold set to 0.90

        self._build_ui()

        # Attach Qt log handler to module logger so logs appear in the UI
        try:
            self.log_handler = QtLogHandler(self.log_text)
            self.log_handler.setLevel(logging.INFO)
            self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(self.log_handler)
            logger.setLevel(logging.INFO)
        except Exception:
            logger.debug("Failed to attach QtLogHandler (UI logging may be disabled)")

        self._discover_models()
        self._auto_load_model()
        self._refresh_legend()

    def _current_effective_params(self) -> Dict:
        """Capture current effective parameters for reuse detection."""
        return {
            "model_path": self.lda_path,
            "pos_idx": int(self.current_pos_idx),
            "band": int(self.current_band),
            "thr": float(self.pix_thr),
            "invert": bool(self.invert_class_chk.isChecked()),
            "shape_enabled": bool(getattr(self, "filter_enable", None) and self.filter_enable.isChecked()),
            "border_r": int(self.border_spin.value()),
            "area_min": int(self.area_min_spin.value()),
            "area_max": int(self.area_max_spin.value()),
            "use_border": bool(self.use_border_chk.isChecked()),
            "use_area": bool(self.use_area_chk.isChecked()),
            "cell": int(self.cell_size),  # grid-only param
        }

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        L = QVBoxLayout(c)

        # --- Top row: images + legend on the far right ---
        imgs = QHBoxLayout(); L.addLayout(imgs)
        # HSI Band display (left)
        self.hsi_label = QLabel("HSI Band")
        self.hsi_label.setFixedSize(512, 512)
        self.hsi_label.setAlignment(Qt.AlignCenter)
        imgs.addWidget(self.hsi_label)
        # HSI RGB composite (middle)
        self.rgb_label = QLabel("HSI RGB")
        self.rgb_label.setFixedSize(512, 512)
        self.rgb_label.setAlignment(Qt.AlignCenter)
        imgs.addWidget(self.rgb_label)
        # Canon RGB (right)
        self.canon_label = QLabel("Canon RGB")
        self.canon_label.setFixedSize(512, 512)
        self.canon_label.setAlignment(Qt.AlignCenter)
        imgs.addWidget(self.canon_label)
        # Legend (far right, compact)
        self.legend_label = QLabel("Legend")
        self.legend_label.setFixedSize(120, 80)
        self.legend_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        imgs.addWidget(self.legend_label)

        # --- Navigation row ---
        nav_row = QHBoxLayout(); L.addLayout(nav_row)
        btn = QPushButton("Load Images"); btn.clicked.connect(self._choose_folder); nav_row.addWidget(btn)

        # Prev/Next navigation buttons
        self.prev_btn = QPushButton("← Prev Cluster"); self.prev_btn.clicked.connect(self._load_prev_cluster); nav_row.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next Cluster →"); self.next_btn.clicked.connect(self._load_next_cluster); nav_row.addWidget(self.next_btn)
        self.cluster_label = QLabel("Cluster: N/A"); self.cluster_label.setFixedWidth(150); nav_row.addWidget(self.cluster_label)

        nav_row.addStretch()

        # Model selector (direct model selection)
        nav_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox(); self.model_combo.setFixedWidth(300)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        nav_row.addWidget(self.model_combo)

        # Class selector (for multi-class models)
        nav_row.addWidget(QLabel("Detect Class:"))
        self.class_combo = QComboBox(); self.class_combo.setFixedWidth(150)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        nav_row.addWidget(self.class_combo)

        # --- Controls row ---
        controls = QHBoxLayout(); L.addLayout(controls)

        # Band slider
        self.band_slider = QSlider(Qt.Horizontal); self.band_slider.setMinimum(0); self.band_slider.setValue(0); self.band_slider.valueChanged.connect(self._update_band); controls.addWidget(QLabel("Band")); controls.addWidget(self.band_slider)

        # Cell size spin
        controls.addWidget(QLabel("Cell"))
        self.cell_spin = QSpinBox(); self.cell_spin.setRange(8, 256); self.cell_spin.setSingleStep(8); self.cell_spin.setValue(self.cell_size); self.cell_spin.valueChanged.connect(self._update_cell); controls.addWidget(self.cell_spin)

        # Pixel threshold spin
        controls.addWidget(QLabel("Prob thr"))
        self.thr_spin = QDoubleSpinBox(); self.thr_spin.setRange(0.0, 1.0); self.thr_spin.setSingleStep(0.05); self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(self.pix_thr)  # Ensure UI reflects default threshold
        self.thr_spin.valueChanged.connect(self._update_thr); controls.addWidget(self.thr_spin)

        # Invert class checkbox
        self.invert_class_chk = QCheckBox("Invert pos class")
        self.invert_class_chk.setChecked(False)  # Default unchecked
        controls.addWidget(self.invert_class_chk)

        # Shape-filter enable - DISABLED by default
        self.filter_enable = QCheckBox("Enable shape filter"); self.filter_enable.setChecked(False); controls.addWidget(self.filter_enable)

        # Border filter toggle + control - DISABLED by default
        self.use_border_chk = QCheckBox("Use border filter"); self.use_border_chk.setChecked(False); self.use_border_chk.setToolTip("Remove blobs within R pixels from image edges"); controls.addWidget(self.use_border_chk)
        controls.addWidget(QLabel("Border Radius (px)"))
        self.border_spin = QSpinBox(); self.border_spin.setRange(0, 100); self.border_spin.setValue(20); self.border_spin.setFixedWidth(80); controls.addWidget(self.border_spin)

        # Area filter toggle + controls - DISABLED by default
        self.use_area_chk = QCheckBox("Use area filter"); self.use_area_chk.setChecked(False); self.use_area_chk.setToolTip("Remove blobs smaller than Min or larger than Max"); controls.addWidget(self.use_area_chk)
        controls.addWidget(QLabel("Min Area (px²)"))
        self.area_min_spin = QSpinBox(); self.area_min_spin.setRange(0, 100000); self.area_min_spin.setValue(10); self.area_min_spin.setFixedWidth(90); controls.addWidget(self.area_min_spin)
        controls.addWidget(QLabel("Max Area (px²)"))
        self.area_max_spin = QSpinBox(); self.area_max_spin.setRange(1, 10000000); self.area_max_spin.setValue(5000); self.area_max_spin.setFixedWidth(90); controls.addWidget(self.area_max_spin)

        # Grid size display
        controls.addWidget(QLabel("Grid"))
        self.grid_size_label = QLabel("N/A"); self.grid_size_label.setFixedWidth(140); controls.addWidget(self.grid_size_label)

        # Max SAM segments control
        controls.addWidget(QLabel("Max SAM blobs"))
        self.max_sam_segments_spin = QSpinBox(); self.max_sam_segments_spin.setRange(1, 500); self.max_sam_segments_spin.setValue(50); self.max_sam_segments_spin.setFixedWidth(80); controls.addWidget(self.max_sam_segments_spin)

        # Threshold buckets editor
        controls.addWidget(QLabel("Legend % (CSV)"))
        self.bucket_edit = QLineEdit("10,20,30,40"); self.bucket_edit.setFixedWidth(120); controls.addWidget(self.bucket_edit)
        apply_btn = QPushButton("Apply legend"); apply_btn.clicked.connect(self._apply_legend_from_text); controls.addWidget(apply_btn)

        # Run buttons
        run_btn = QPushButton("Run Patch Analysis"); run_btn.clicked.connect(self._run_patches); controls.addWidget(run_btn)
        detect_btn = QPushButton("Run Detection Only"); detect_btn.clicked.connect(self._run_detection_only); controls.addWidget(detect_btn)
        show_grid_btn = QPushButton("Show Grid"); show_grid_btn.clicked.connect(self._show_detection_grid); controls.addWidget(show_grid_btn)
        show_sam_btn = QPushButton("Show SAM Segments"); show_sam_btn.clicked.connect(self._show_sam_segments); controls.addWidget(show_sam_btn)

        # --- Screenshot button ---
        self.screenshot_btn = QPushButton("Screenshot (all images)")
        self.screenshot_btn.clicked.connect(self._save_screenshot)
        nav_row.addWidget(self.screenshot_btn)

        # --- Dataset Runner and Logging panel ---
        ds_box = QGroupBox("Dataset Runner")
        ds_layout = QVBoxLayout(ds_box)
        # CSV path + browse
        path_row = QHBoxLayout()
        self.dataset_csv_edit = QLineEdit()
        self.dataset_csv_edit.setText(DEFAULT_LATE_DETECTION_CSV)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dataset)
        path_row.addWidget(self.dataset_csv_edit)
        path_row.addWidget(browse_btn)
        ds_layout.addLayout(path_row)

        # subset selector + auto-display
        opts_row = QHBoxLayout()
        self.dataset_subset_combo = QComboBox(); self.dataset_subset_combo.addItems(["Dev only", "Test only", "Dev + Test"])
        opts_row.addWidget(QLabel("Subset:")); opts_row.addWidget(self.dataset_subset_combo)
        self.dataset_auto_display_chk = QCheckBox("Auto-display in viewer")
        self.dataset_auto_display_chk.setChecked(True)
        self.dataset_auto_display_chk.stateChanged.connect(lambda s: setattr(self, 'dataset_auto_display', bool(s)))
        opts_row.addWidget(self.dataset_auto_display_chk)
        ds_layout.addLayout(opts_row)

        # control buttons
        btn_row = QHBoxLayout()
        self.dataset_load_button = QPushButton("Load Dataset"); self.dataset_load_button.clicked.connect(self._dataset_load)
        self.dataset_start_button = QPushButton("Start"); self.dataset_start_button.clicked.connect(self._dataset_start)
        self.dataset_stop_button = QPushButton("Stop"); self.dataset_stop_button.clicked.connect(self._dataset_stop)
        self.dataset_prev_button = QPushButton("Prev Sample"); self.dataset_prev_button.clicked.connect(self._dataset_prev)
        self.dataset_next_button = QPushButton("Next Sample"); self.dataset_next_button.clicked.connect(self._dataset_next)
        btn_row.addWidget(self.dataset_load_button)
        btn_row.addWidget(self.dataset_start_button)
        btn_row.addWidget(self.dataset_stop_button)
        btn_row.addStretch(); btn_row.addWidget(self.dataset_prev_button); btn_row.addWidget(self.dataset_next_button)
        ds_layout.addLayout(btn_row)

        # results table
        self.dataset_table = QTableWidget(0, 6)
        self.dataset_table.setHorizontalHeaderLabels(["grape_id", "row", "label", "crack_ratio", "pred_label", "status"])
        ds_layout.addWidget(self.dataset_table)

        L.addWidget(ds_box)

        # Logging panel (QPlainTextEdit)
        self.log_panel = QGroupBox("Log")
        self.log_layout = QVBoxLayout(self.log_panel)
        self.log_text = QPlainTextEdit(); self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        L.addWidget(self.log_panel)

        self.status_bar.showMessage("Load a folder, set params, then Run Patch Analysis.")

    def _discover_models(self):
        """Populate model dropdown from AVAILABLE_MODELS dictionary."""
        self.model_combo.clear()

        for model_name in self.available_model_names:
            self.model_combo.addItem(model_name)

        # Set default model
        if self.lda_path:
            # Find index of current model path
            try:
                idx = self.available_model_paths.index(self.lda_path)
                self.model_combo.setCurrentIndex(idx)
            except ValueError:
                # Default not in list, select first
                if self.available_model_paths:
                    self.lda_path = self.available_model_paths[0]
                    self.model_combo.setCurrentIndex(0)
        elif self.available_model_paths:
            self.lda_path = self.available_model_paths[0]
            self.model_combo.setCurrentIndex(0)

        logger.info("Loaded %d models into dropdown", len(self.available_model_names))


    def _on_model_changed(self, index: int):
        """Handle model selection change."""
        if index >= 0 and index < len(self.available_model_paths):
            self.lda_path = self.available_model_paths[index]
            # Find scaler for this model
            cand = find_scaler(os.path.dirname(self.lda_path), [os.path.basename(self.lda_path)])
            self.scaler_path = cand
            model_display_name = self.available_model_names[index]

            # Load model to get classes
            try:
                _, _, pos_idx, classes = load_model_and_scaler(self.lda_path, self.scaler_path)
                self.model_classes = classes
                self.current_pos_idx = pos_idx

                # Populate class selector
                self.class_combo.blockSignals(True)  # Prevent triggering change event
                self.class_combo.clear()
                for i, cls in enumerate(classes):
                    self.class_combo.addItem(str(cls))
                self.class_combo.setCurrentIndex(pos_idx)
                self.class_combo.blockSignals(False)

                # Show/hide class selector based on number of classes
                if len(classes) > 2:
                    self.class_combo.setEnabled(True)
                    logger.info("Multi-class model detected (%d classes). Class selector enabled.", len(classes))
                else:
                    self.class_combo.setEnabled(True)  # Keep enabled for visibility
                    logger.info("Binary model detected (2 classes).")

            except Exception as e:
                logger.warning("Failed to load model classes: %s", e)
                self.model_classes = np.array([])
                self.current_pos_idx = 0

            logger.info("Model changed to: %s (scaler: %s)", model_display_name, os.path.basename(self.scaler_path) if self.scaler_path else "None")
            self.status_bar.showMessage(f"Model: {model_display_name}")

    def _on_class_changed(self, index: int):
        """Handle target class selection change."""
        if index >= 0 and index < len(self.model_classes):
            self.current_pos_idx = index
            logger.info("Target class changed to: %s (index %d)", self.model_classes[index], index)
            self.status_bar.showMessage(f"Detecting class: {self.model_classes[index]}")



    def _auto_load_model(self):
        if not self.lda_path:
            self.status_bar.showMessage("Default LDA not found. Set DEFAULT_LDA_PATH or pick another model."); return
        cand = find_scaler(os.path.dirname(self.lda_path), [os.path.basename(self.lda_path)])
        self.scaler_path = cand

    def _discover_available_clusters(self):
        """Find all available clusters (e.g., 1_01, 1_02, ...) in the raw data folder."""
        try:
            folders = [f for f in os.listdir(DEFAULT_SEARCH_FOLDER) if os.path.isdir(os.path.join(DEFAULT_SEARCH_FOLDER, f))]
            # Filter for cluster pattern (e.g., 1_01, 2_15, etc.)
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
            self.status_bar.showMessage("No clusters available for navigation.")
            return

        try:
            current_idx = self.available_clusters.index(self.current_cluster_id)
            if current_idx > 0:
                prev_cluster = self.available_clusters[current_idx - 1]
                # Check if the same date exists in the previous cluster
                prev_folder = os.path.join(DEFAULT_SEARCH_FOLDER, prev_cluster, self.current_date)
                if os.path.exists(prev_folder):
                    self._load_images(prev_folder)
                else:
                    self.status_bar.showMessage(f"Date {self.current_date} not found in cluster {prev_cluster}")
            else:
                self.status_bar.showMessage("Already at first cluster.")
        except ValueError:
            self.status_bar.showMessage("Current cluster not in list.")
        except Exception as e:
            logger.exception("Failed to load previous cluster: %s", e)
            self.status_bar.showMessage(f"Error loading previous cluster: {e}")

    def _load_next_cluster(self):
        """Load the next cluster with the same date."""
        if not self.current_cluster_id or not self.current_date:
            self.status_bar.showMessage("Load a folder first to enable navigation.")
            return

        self._discover_available_clusters()
        if not self.available_clusters:
            self.status_bar.showMessage("No clusters available for navigation.")
            return

        try:
            current_idx = self.available_clusters.index(self.current_cluster_id)
            if current_idx < len(self.available_clusters) - 1:
                next_cluster = self.available_clusters[current_idx + 1]
                # Check if the same date exists in the next cluster
                next_folder = os.path.join(DEFAULT_SEARCH_FOLDER, next_cluster, self.current_date)
                if os.path.exists(next_folder):
                    self._load_images(next_folder)
                else:
                    self.status_bar.showMessage(f"Date {self.current_date} not found in cluster {next_cluster}")
            else:
                self.status_bar.showMessage("Already at last cluster.")
        except ValueError:
            self.status_bar.showMessage("Current cluster not in list.")
        except Exception as e:
            logger.exception("Failed to load next cluster: %s", e)
            self.status_bar.showMessage(f"Error loading next cluster: {e}")


    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", DEFAULT_SEARCH_FOLDER)
        if not folder:
            self.status_bar.showMessage("No folder selected."); return
        self.folder_path = folder
        self._load_images(folder)

    def _load_images(self, folder: str):
        # Extract cluster_id and date from folder path
        # Expected path structure: .../raw/1_14/25.09.24
        parts = folder.replace('\\', '/').split('/')
        if len(parts) >= 2:
            self.current_date = parts[-1]  # e.g., "25.09.24"
            self.current_cluster_id = parts[-2]  # e.g., "1_14"
            self.cluster_label.setText(f"Cluster: {self.current_cluster_id}")
            logger.info("Loaded cluster: %s, date: %s", self.current_cluster_id, self.current_date)

        hs = os.path.join(folder, "HS")
        rgb_files = [f for f in os.listdir(hs) if f.lower().endswith(".png")]
        if not rgb_files: raise FileNotFoundError("No RGB image in HS folder")
        rgb = cv2.cvtColor(cv2.imread(os.path.join(hs, rgb_files[0])), cv2.COLOR_BGR2RGB)
        # Show HS-provided RGB in the left label (no rotation)
        self._show_image(rgb, self.rgb_label)

        # Try to find a Canon/RGB camera image somewhere under the folder (recursive search)
        canon_img_path = None
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if 'canon' in f.lower() or 'canon' in os.path.basename(root).lower() or 'rgb' in os.path.basename(root).lower():
                        canon_img_path = os.path.join(root, f)
                        break
            if canon_img_path:
                break
        if canon_img_path is None:
            # fallback: any image in the folder (excluding HS/results .png used above)
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        p = os.path.join(root, f)
                        if os.path.abspath(p) != os.path.abspath(os.path.join(hs, rgb_files[0])):
                            canon_img_path = p
                            break
                if canon_img_path:
                    break

        if canon_img_path:
            try:
                canon_bgr = cv2.imread(canon_img_path)
                if canon_bgr is not None:
                    canon_rgb = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2RGB)
                    # Display canon RGB without rotation
                    self._show_image(canon_rgb, self.canon_label)
                    logger.info("Loaded Canon image: %s", canon_img_path)
            except Exception as e:
                logger.warning("Failed to load Canon image: %s", e)

        res = os.path.join(hs, "results")
        hdr_files = [f for f in os.listdir(res) if f.lower().endswith(".hdr")]
        if not hdr_files: raise FileNotFoundError("No .hdr in HS/results")
        self.hdr_path = os.path.join(res, hdr_files[0])
        self.hsi_cube = load_cube(self.hdr_path)
        self.band_slider.setMaximum(self.hsi_cube.shape[2] - 1)
        self._update_band()
        # update grid label when cube is loaded
        self._update_grid_label()
        self.status_bar.showMessage(f"Loaded images from {folder}")

    def _update_band(self):
        if self.hsi_cube is None: return
        self.current_band = self.band_slider.value()
        band = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # keep rotation for HSI display so overlay aligns correctly
        band = cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE)
        self._show_image(band, self.hsi_label, True)
        self.status_bar.showMessage(f"Band {self.current_band}")

    def _update_cell(self, v: int):
        self.cell_size = int(v)
        # update and log grid size
        self._update_grid_label()
        logger.info("Cell size changed -> %d", self.cell_size)
        self.status_bar.showMessage(f"Cell size set to {self.cell_size}")

    def _update_thr(self, v: float):
        self.pix_thr = float(v)

    def _show_image(self, img: np.ndarray, label: QLabel, is_hsi: bool = False):
        if is_hsi and img.ndim == 2:
            h, w = img.shape; q = QImage(img.data, w, h, QImage.Format_Grayscale8)
        else:
            h, w, ch = img.shape; q = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q)); label.setScaledContents(True)

    def _refresh_legend(self):
        img = render_legend_image(GRID_COLOR_BUCKETS)
        h, w, _ = img.shape
        q = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        self.legend_label.setPixmap(QPixmap.fromImage(q))
        self.legend_label.setScaledContents(True)

    def _apply_legend_from_text(self):
        text = self.bucket_edit.text().strip()
        try:
            # parse CSV like '20,30,40,50' -> ascending list [20,30,40,50]
            vals = sorted({int(x) for x in text.split(',') if x.strip() != ''})
            if not vals:
                raise ValueError("No thresholds parsed from CSV")

            n = len(vals)
            if n > 9:
                logger.warning(f"Large number of thresholds ({n}). Recommended max is 9 for clarity.")
                self.status_bar.showMessage(f"Warning: {n} thresholds (>9 recommended). Colors may be hard to distinguish.")

            # Build dynamic palette: N colors from light-yellow to dark-red (BGR format)
            palette = build_dynamic_palette(n)

            # Build descending list for GRID_COLOR_BUCKETS: highest threshold first
            # Map: highest val (index n-1) -> darkest color (palette[n-1])
            #      lowest val (index 0) -> lightest color (palette[0])
            new_buckets = []
            for i in range(n - 1, -1, -1):  # iterate from highest val to lowest
                thr = vals[i]  # e.g., 50, 40, 30, 20
                color = palette[i]  # corresponding color from palette
                new_buckets.append((thr, color))

            # Update the global GRID_COLOR_BUCKETS in module scope
            globals()['GRID_COLOR_BUCKETS'] = new_buckets
            logger.info("Updated legend thresholds (desc, highest first): %s", new_buckets)
            logger.info("Palette colors (BGR): %s", palette)
            self._refresh_legend()
        except Exception as e:
            logger.warning("Error parsing legend CSV: %s", e)
            self.status_bar.showMessage("Invalid legend CSV. Use numbers like 20,30,40,50")

    def _run_patches(self):
        if not self.hdr_path:
            self.status_bar.showMessage("Load an HSI folder first."); return
        if not self.lda_path:
            self.status_bar.showMessage("No LDA model configured."); return
        band = self.current_band; cell = self.cell_size; thr = self.pix_thr
        # log grid rows/cols before running
        if self.hsi_cube is not None:
            H, W, _ = self.hsi_cube.shape
            rows = (H + cell - 1) // cell
            cols = (W + cell - 1) // cell
            logger.info("Running analysis -> band=%d cell=%d thr=%.2f | grid=%dx%d", band, cell, thr, rows, cols)

        # Build filter_params ONLY with currently active settings
        filter_params = {"invert": self.invert_class_chk.isChecked()}
        shape_filter_enabled = getattr(self, "filter_enable", None) and self.filter_enable.isChecked()

        if shape_filter_enabled:
            # Add shape filter params ONLY if enabled
            filter_params.update({
                "border_r": int(self.border_spin.value()),
                "area_min": int(self.area_min_spin.value()),
                "area_max": int(self.area_max_spin.value()),
                "use_border": bool(self.use_border_chk.isChecked()),
                "use_area": bool(self.use_area_chk.isChecked()),
            })
            logger.info("Shape filter ENABLED -> %s", filter_params)
        else:
            logger.info("Shape filter DISABLED (invert=%s)", filter_params.get("invert"))
            # Explicitly set shape_enabled=False so run_patch_analysis knows not to apply filters
            filter_params["shape_enabled"] = False

        # Attach filter params for run_patch_analysis to pick up
        run_patch_analysis.filter_params = filter_params

        # Decide whether we can reuse last detection map
        current_params = self._current_effective_params()
        # Exclude "cell" from reuse decision (grid display only; doesn't affect prob_map)
        current_params_for_comparison = {k: v for k, v in current_params.items() if k != "cell"}
        last_params_for_comparison = {k: v for k, v in self.last_params.items() if k != "cell"}

        can_reuse = (
            self.last_detection_prob_map is not None
            and self.last_params
            and current_params_for_comparison == last_params_for_comparison
        )

        if can_reuse:
            logger.info("Patch analysis can reuse detection prob_map (params match exactly)")
        else:
            logger.info("Patch analysis will recompute prob_map (params differ or no cached map)")
            if self.last_params:
                # Show what differs
                for key in current_params_for_comparison:
                    if current_params_for_comparison.get(key) != last_params_for_comparison.get(key):
                        logger.debug("  Param mismatch: %s: current=%s vs last=%s", key, current_params_for_comparison.get(key), last_params_for_comparison.get(key))

        model_name = os.path.splitext(os.path.basename(self.lda_path))[0]
        out_dir = os.path.join(os.path.dirname(self.hdr_path), f"grid_results_{model_name}")
        os.makedirs(out_dir, exist_ok=True)

        self.worker = AnalysisThread(self.hdr_path, self.lda_path, self.scaler_path, band, cell, thr, out_dir, self.current_pos_idx)

        # Monkey-patch the thread to pass reuse_prob_map into run_patch_analysis
        orig_run = self.worker.run
        def _run_with_reuse():
            try:
                t0 = time.perf_counter()
                overlay_path, csv_path = run_patch_analysis(
                    self.hdr_path, self.lda_path, self.scaler_path,
                    band, cell, thr, out_dir, self.current_pos_idx,
                    reuse_prob_map=(self.last_detection_prob_map if can_reuse else None)
                )
                bgr = cv2.imread(overlay_path)
                if bgr is None:
                    logger.warning("Overlay not found after analysis: %s", overlay_path)
                    self.worker.finished_signal.emit(None, csv_path); return
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                logger.info("Analysis done in %.2fs (reuse=%s)", time.perf_counter() - t0, can_reuse)
                self.worker.finished_signal.emit(rgb, csv_path)
            except Exception as e:
                logger.exception("AnalysisThread failed: %s", e)
                self.worker.finished_signal.emit(None, "")

        self.worker.run = _run_with_reuse
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()
        self.status_bar.showMessage(f"Running analysis… (reuse cached map: {can_reuse})")

    def _run_detection_only(self):
        if not self.hdr_path:
            self.status_bar.showMessage("Load an HSI folder first."); return
        if not self.lda_path:
            self.status_bar.showMessage("No LDA model configured."); return

        try:
            logger.info("Running detection-only -> band=%d thr=%.2f", self.current_band, self.pix_thr)
            logger.info(f"HSI cube shape: {None if self.hsi_cube is None else self.hsi_cube.shape}")
            # Load model if needed
            lda, scaler, default_pos_idx, classes = load_model_and_scaler(self.lda_path, self.scaler_path)

            # Use current_pos_idx (selected from dropdown)
            pos_idx = self.current_pos_idx

            # Compute probabilities
            prob_map = per_pixel_probs(self.hsi_cube, lda, scaler, pos_idx)

            # Invert class if checkbox is checked
            if self.invert_class_chk.isChecked():
                logger.info("Inverting probabilities (invert checkbox checked)")
                prob_map = 1.0 - prob_map

            # Apply shape filter if enabled
            if getattr(self, "filter_enable", None) and self.filter_enable.isChecked():
                filter_params = dict(
                    border_r=self.border_spin.value(),
                    area_min=self.area_min_spin.value(),
                    area_max=self.area_max_spin.value(),
                    use_border=self.use_border_chk.isChecked(),
                    use_area=self.use_area_chk.isChecked(),
                )
                logger.info("Detection filter -> %s", filter_params)
                mask = filter_blobs_by_shape(prob_map, self.pix_thr, **filter_params)
            else:
                mask = (prob_map >= self.pix_thr)

            # Store the latest detection probability map for grid visualization
            self.last_detection_prob_map = prob_map
            self.last_detection_mask = mask

            # Store detection parameters for reuse detection
            self.last_params = self._current_effective_params()
            logger.info("Stored detection params: %s", self.last_params)

            # Render detection overlay
            base_band = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            overlay = colorize_binary_mask(base_band, mask)
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)

            # Convert to RGB for display
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            self._show_image(overlay_rgb, self.hsi_label)

            blob_count = int(mask.sum())
            class_name = classes[pos_idx] if pos_idx < len(classes) else pos_idx
            self.status_bar.showMessage(f"Detection only: {blob_count} pixels detected for class '{class_name}' — Click 'Show Grid' to visualize patch ratios")
            logger.info("Detection visualization updated: %d pixels for class '%s'", blob_count, class_name)

        except Exception as e:
            logger.exception("Detection-only failed: %s", e)
            self.status_bar.showMessage(f"Detection failed: {e}")

    def _show_detection_grid(self):
        """Generate and display patch-level grid heatmap from the latest detection-only results."""
        if self.last_detection_prob_map is None:
            self.status_bar.showMessage("Run Detection Only first to generate grid visualization")
            return
        if self.hsi_cube is None:
            self.status_bar.showMessage("No HSI cube loaded")
            return

        try:
            logger.info("Generating patch grid from detection results -> cell=%d thr=%.2f", self.cell_size, self.pix_thr)

            # Analyze grid based on stored probability map
            grid_stats = analyze_grid(self.last_detection_prob_map, self.cell_size, self.pix_thr)

            # Get the band for the background
            band_img = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Overlay grid patches on the band image
            overlay_img = overlay_on_band(band_img, grid_stats, alpha=0.45)

            # Rotate to match display orientation
            overlay_rotated = cv2.rotate(overlay_img, cv2.ROTATE_90_CLOCKWISE)

            # Convert to RGB for display
            overlay_rgb = cv2.cvtColor(overlay_rotated, cv2.COLOR_BGR2RGB)
            self._show_image(overlay_rgb, self.hsi_label)

            # Log grid statistics
            percents = [c["percent_cracked"] for c in grid_stats]
            if percents:
                mean_pct = float(np.mean(percents))
                max_pct = float(np.max(percents))
                logger.info("Grid visualization: %d patches | mean=%.2f%% max=%.2f%%", len(grid_stats), mean_pct, max_pct)
                self.status_bar.showMessage(f"Grid visualization: {len(grid_stats)} patches | mean cracked={mean_pct:.1f}% | max={max_pct:.1f}%")
            else:
                self.status_bar.showMessage("Grid visualization complete (no cracked pixels detected)")

        except Exception as e:
            logger.exception("Failed to show detection grid: %s", e)
            self.status_bar.showMessage(f"Grid visualization failed: {e}")

    def _show_sam_segments(self):
        """Generate and display SAM segmentation for detected blobs."""
        if self.last_detection_mask is None:
            self.status_bar.showMessage("Run Detection Only first to generate SAM segments")
            return
        if self.hsi_cube is None:
            self.status_bar.showMessage("No HSI cube loaded")
            return
        if self.rgb_image is None:
            self.status_bar.showMessage("No RGB image loaded")
            return

        try:
            # Show progress dialog
            progress = QProgressDialog("Initializing SAM and segmenting blobs...", None, 0, 0, self)
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

            # Extract blob centroids from detection mask
            max_blobs = self.max_sam_segments_spin.value()
            logger.info(f"Extracting centroids from detection mask (max_blobs={max_blobs})...")
            progress.setLabelText(f"Extracting up to {max_blobs} blob centroids...")
            QApplication.processEvents()

            centroids = extract_blob_centroids(self.last_detection_mask, max_blobs=max_blobs)

            if not centroids:
                progress.close()
                self.status_bar.showMessage("No blobs found in detection mask")
                QMessageBox.warning(self, "No Blobs", "No blobs found in detection mask. Try adjusting threshold or filters.")
                return

            logger.info(f"Found {len(centroids)} blob centroids, starting SAM segmentation...")
            progress.setLabelText(f"Segmenting {len(centroids)} blobs with SAM...")
            progress.setMaximum(len(centroids))
            QApplication.processEvents()

            # Segment each blob with SAM
            masks = []
            for i, point in enumerate(centroids):
                try:
                    _, mask_bool = self.sam2_segmenter.segment_object_from_array(self.rgb_image, [point])
                    masks.append(mask_bool)
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                except Exception as e:
                    logger.warning(f"Failed to segment point {i} at {point}: {e}")
                    masks.append(np.zeros(self.rgb_image.shape[:2], dtype=bool))

            progress.setLabelText("Creating overlay visualization...")
            QApplication.processEvents()

            # Create colored overlay on HSI band
            band_img = cv2.normalize(
                self.hsi_cube[:, :, self.current_band],
                None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            overlay = create_sam_segment_overlay(
                band_img,
                masks,
                prob_map=self.last_detection_prob_map,
                alpha=0.4
            )

            # Rotate to match display orientation
            overlay_rotated = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)

            # Convert to RGB for display
            overlay_rgb = cv2.cvtColor(overlay_rotated, cv2.COLOR_BGR2RGB)
            self._show_image(overlay_rgb, self.hsi_label)

            # Cache results
            self.last_sam_segments = masks
            self.last_sam_overlay = overlay_rgb

            # Calculate statistics
            total_segment_pixels = sum(mask.sum() for mask in masks)
            avg_segment_size = total_segment_pixels / len(masks) if masks else 0

            progress.close()
            logger.info(f"SAM segmentation complete: {len(masks)} segments | avg size={avg_segment_size:.1f}px")
            self.status_bar.showMessage(
                f"SAM segmentation: {len(masks)} segments | "
                f"avg size={avg_segment_size:.0f}px | "
                f"total={total_segment_pixels}px"
            )

        except Exception as e:
            logger.exception("Failed to show SAM segments: %s", e)
            self.status_bar.showMessage(f"SAM segmentation failed: {e}")
            QMessageBox.critical(self, "Error", f"SAM segmentation failed:\n{str(e)}")

    # New helper: update grid size label
    def _update_grid_label(self):
        if self.hsi_cube is None:
            self.grid_size_label.setText("N/A")
            return
        H, W, _ = self.hsi_cube.shape
        cell = max(1, int(self.cell_size))
        rows = (H + cell - 1) // cell
        cols = (W + cell - 1) // cell
        self.grid_size_label.setText(f"{rows} x {cols} (cell={cell})")
        logger.info("Grid size updated: rows=%d cols=%d cell=%d", rows, cols, cell)

    def _on_finished(self, overlay_rgb: Optional[np.ndarray], csv_path: str):
        if overlay_rgb is not None:
            # overlay_rgb here is already rotated in the worker to match HSI orientation
            self._show_image(overlay_rgb, self.hsi_label, True)
            reuse_note = " (same prob map as 'Detection Only')" if self.last_detection_prob_map is not None else ""
            self.status_bar.showMessage(f"Done. CSV: {os.path.basename(csv_path)}{reuse_note}")
        else:
            self.status_bar.showMessage("Analysis finished (no overlay).")

    def _save_screenshot(self):
        try:
            # Ask user to select prefix
            options = ["new_detect", "new_patch", "old_detect", "old_patch"]
            prefix, ok = QInputDialog.getItem(
                self,
                "Screenshot Prefix",
                "Select prefix for screenshot name:",
                options,
                0,
                False
            )
            if not ok:
                self.status_bar.showMessage("Screenshot cancelled.")
                return

            # Grab pixmaps from the three main image labels only (no legend)
            hsi_pixmap = self.hsi_label.pixmap()
            rgb_pixmap = self.rgb_label.pixmap()
            canon_pixmap = self.canon_label.pixmap()

            if not (hsi_pixmap and rgb_pixmap and canon_pixmap):
                self.status_bar.showMessage("Cannot screenshot: missing image(s)")
                return
            # Convert to QImage
            hsi_img = hsi_pixmap.toImage()
            rgb_img = rgb_pixmap.toImage()
            canon_img = canon_pixmap.toImage()

            # Convert QImage to numpy array and resize to 512x512
            def qimage_to_np(qimg):
                qimg = qimg.convertToFormat(QImage.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                ptr.setsize(h * w * 3)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
                return arr.copy()

            target_size = (512, 512)
            hsi_np = cv2.resize(qimage_to_np(hsi_img), target_size)
            rgb_np = cv2.resize(qimage_to_np(rgb_img), target_size)
            canon_np = cv2.resize(qimage_to_np(canon_img), target_size)
            # Concatenate horizontally: HSI (left), RGB (middle), Canon (right)
            combined = cv2.hconcat([hsi_np, rgb_np, canon_np])
            # Build filename with timestamp


            # Save to thesis folder with cluster_id subfolder
            cluster_id = self.current_cluster_id if hasattr(self, 'current_cluster_id') and self.current_cluster_id else 'unknown'
            filename = f"{prefix}_{cluster_id}.png"
            thesis_folder = os.path.join(r"C:\Users\yovel\Desktop\images_for_thesis\full_image_detection", cluster_id)
            os.makedirs(thesis_folder, exist_ok=True)
            save_path = os.path.join(thesis_folder, filename)
            cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            self.status_bar.showMessage(f"Screenshot saved: {save_path}")
            logger.info("Screenshot saved: %s", save_path)
        except Exception as e:
            logger.exception("Screenshot failed: %s", e)
            self.status_bar.showMessage(f"Screenshot failed: {e}")

    # ===== Dataset runner methods =====
    def _browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset CSV", DEFAULT_LATE_DETECTION_CSV, "CSV Files (*.csv);;All Files (*)")
        if path:
            self.dataset_csv_edit.setText(path)

    def _dataset_load(self):
        path = self.dataset_csv_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Dataset", f"Dataset CSV not found: {path}")
            return
        try:
            df = pd.read_csv(path)
            # normalize columns
            df = df.rename(columns={c: c.strip() for c in df.columns})
            # Ensure required cols
            for col in ("grape_id", "row", "image_path", "label"):
                if col not in df.columns:
                    QMessageBox.warning(self, "Dataset", f"Missing column '{col}' in CSV")
                    return
            df = df[["grape_id", "row", "image_path", "label"]].copy()
            df["grape_id"] = df["grape_id"].astype(str)
            df["row"] = df["row"].astype(int)
            df["image_path"] = df["image_path"].astype(str)
            df["label"] = df["label"].astype(int)

            # subset
            subset = self.dataset_subset_combo.currentText()
            if subset == "Dev only":
                df = df[df["row"] == 1].reset_index(drop=True)
            elif subset == "Test only":
                df = df[df["row"] == 2].reset_index(drop=True)
            else:
                df = df[df["row"].isin([1,2])].reset_index(drop=True)

            self.dataset_df = df
            # prepare results df
            res = df.copy()
            res["crack_ratio"] = float('nan')
            res["pred_label"] = float('nan')
            res["status"] = "pending"
            self.dataset_results = res
            self.dataset_current_index = -1
            self.dataset_running = False

            # populate table
            self.dataset_table.setRowCount(len(df))
            for i, row in df.iterrows():
                ii = int(i)
                self.dataset_table.setItem(ii, 0, QTableWidgetItem(str(row["grape_id"])))
                self.dataset_table.setItem(ii, 1, QTableWidgetItem(str(int(row["row"]))))
                self.dataset_table.setItem(ii, 2, QTableWidgetItem(str(int(row["label"])) ))
                self.dataset_table.setItem(ii, 3, QTableWidgetItem(""))
                self.dataset_table.setItem(ii, 4, QTableWidgetItem(""))
                self.dataset_table.setItem(ii, 5, QTableWidgetItem("pending"))
            logger.info("Loaded dataset: %d rows (subset=%s)", len(df), subset)
        except Exception as e:
            logger.exception("Failed to load dataset: %s", e)
            QMessageBox.critical(self, "Dataset", f"Failed to load dataset: {e}")

    def _dataset_start(self):
        if self.dataset_df is None or self.dataset_df.empty:
            QMessageBox.warning(self, "Dataset", "No dataset loaded")
            return
        # ensure model loaded
        try:
            if not self.lda_path:
                QMessageBox.warning(self, "Model", "No LDA model configured")
                return
            self.lda_model_for_dataset = load_model_and_scaler(self.lda_path, self.scaler_path)
        except Exception as e:
            logger.exception("Failed to load model for dataset run: %s", e)
            QMessageBox.critical(self, "Model", f"Failed to load model: {e}")
            return

        self.dataset_running = True
        if self.dataset_current_index < 0:
            self.dataset_current_index = 0
        logger.info("Starting dataset run (rows=%d) from index %d", len(self.dataset_df), self.dataset_current_index)
        QTimer.singleShot(10, self._dataset_run_step)

    def _dataset_stop(self):
        self.dataset_running = False
        logger.info("Dataset run stopped by user")

    def _dataset_next(self):
        if self.dataset_df is None: return
        if self.dataset_current_index < len(self.dataset_df) - 1:
            self.dataset_current_index += 1
            self._dataset_display_index(self.dataset_current_index)

    def _dataset_prev(self):
        if self.dataset_df is None: return
        if self.dataset_current_index > 0:
            self.dataset_current_index -= 1
            self._dataset_display_index(self.dataset_current_index)

    def _dataset_display_index(self, idx: int):
        # highlight table row
        self.dataset_table.selectRow(idx)
        # optionally display visualization for this sample
        if not self.dataset_auto_display:
            return
        try:
            r = self.dataset_df.loc[idx]
            image_path = r["image_path"]
            hdr_glob = glob.glob(os.path.join(image_path, "HS", "results", "*.hdr"))
            if not hdr_glob:
                logger.warning("No .hdr for display: %s", image_path)
                return
            hdr_path = hdr_glob[0]
            cube = load_cube(hdr_path)
            # reuse model loaded earlier
            lda, scaler, pos_idx, classes = load_model_and_scaler(self.lda_path, self.scaler_path)
            prob_map = per_pixel_probs(cube, lda, scaler, pos_idx)
            mask = (prob_map >= DEFAULT_PIX_THR)
            band_img = cv2.normalize(cube[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            overlay = colorize_binary_mask(band_img, mask)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            self._show_image(overlay_rgb, self.hsi_label)
        except Exception as e:
            logger.exception("Failed to display sample %d: %s", idx, e)

    def _dataset_run_step(self):
        if not self.dataset_running:
            return
        if self.dataset_df is None:
            self.dataset_running = False
            return
        if self.dataset_current_index >= len(self.dataset_df):
            # finished
            self.dataset_running = False
            logger.info("Dataset run completed")
            self._dataset_finish()
            return
        idx = int(self.dataset_current_index)
        try:
            self._dataset_process_row(idx)
        except Exception as e:
            logger.exception("Processing row %d failed: %s", idx, e)
            # mark failed
            self.dataset_results.at[idx, 'status'] = 'error'
            self.dataset_table.setItem(idx, 5, QTableWidgetItem('error'))
        # advance
        self.dataset_current_index += 1
        QTimer.singleShot(10, self._dataset_run_step)

    def _dataset_process_row(self, idx: int):
        if self.dataset_df is None: return
        r = self.dataset_df.loc[idx]
        grape_id = str(r['grape_id'])
        image_path = str(r['image_path'])
        label = int(r['label']) if not pd.isna(r['label']) else None
        status = 'ok'
        crack_ratio = float('nan')
        pred_label = float('nan')
        try:
            # locate hdr
            hdr_glob = glob.glob(os.path.join(image_path, 'HS', 'results', '*.hdr'))
            if not hdr_glob:
                status = 'no_hdr'
                logger.warning('No .hdr found for %s', grape_id)
            else:
                hdr_path = hdr_glob[0]
                cube = load_cube(hdr_path)
                lda, scaler, pos_idx, classes = load_model_and_scaler(self.lda_path, self.scaler_path)
                prob_map = per_pixel_probs(cube, lda, scaler, pos_idx)
                mask = (prob_map >= DEFAULT_PIX_THR)
                H, W = prob_map.shape
                total = H * W
                cracked = int(mask.sum())
                crack_ratio = cracked / max(1, total)
                # cluster decision threshold (default 0.05) - can be adjusted
                cluster_thr = CLUSTER_CRACK_RATIO_THR_CANDIDATES[1] if len(CLUSTER_CRACK_RATIO_THR_CANDIDATES) > 1 else CLUSTER_CRACK_RATIO_THR_CANDIDATES[0]
                pred_label = 1 if crack_ratio >= cluster_thr else 0
                status = 'ok'
                logger.info("Processed %s row=%s label=%s -> crack_ratio=%.4f pred=%d", grape_id, r['row'], label, crack_ratio, pred_label)
                # update visual if requested
                if self.dataset_auto_display:
                    band_img = cv2.normalize(cube[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    overlay = colorize_binary_mask(band_img, mask)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    self._show_image(overlay_rgb, self.hsi_label)
        except Exception as e:
            logger.exception("Error processing dataset row %s: %s", grape_id, e)
            status = 'error'

        # write results to table and dataset_results
        try:
            self.dataset_results.at[idx, 'crack_ratio'] = crack_ratio
            self.dataset_results.at[idx, 'pred_label'] = int(pred_label) if not pd.isna(pred_label) else float('nan')
            self.dataset_results.at[idx, 'status'] = status
            # update table columns: crack_ratio(3), pred_label(4), status(5)
            self.dataset_table.setItem(idx, 3, QTableWidgetItem(f"{crack_ratio:.4f}" if not pd.isna(crack_ratio) else ""))
            self.dataset_table.setItem(idx, 4, QTableWidgetItem(str(int(pred_label)) if not pd.isna(pred_label) else ""))
            self.dataset_table.setItem(idx, 5, QTableWidgetItem(status))
        except Exception:
            pass

    def _dataset_finish(self):
        # compute metrics and save results
        if self.dataset_results is None:
            return
        try:
            self.dataset_results.to_csv(DATASET_RESULTS_CSV, index=False)
            logger.info("Saved dataset per-sample results to %s", DATASET_RESULTS_CSV)
        except Exception as e:
            logger.exception("Failed to save dataset results: %s", e)

        # compute metrics for dev/test
        valid = self.dataset_results[self.dataset_results['status'] == 'ok'].copy()
        if valid.empty:
            logger.info("No valid rows to compute metrics")
            return
        valid['y_true'] = valid['label'].astype(int)
        valid['y_pred'] = valid['pred_label'].astype(int)

        metrics = []
        for split_name, split_df in (('dev', valid[valid['row'] == 1]), ('test', valid[valid['row'] == 2])):
            if split_df.empty:
                metrics.append({'split': split_name, 'acc': float('nan'), 'prec': float('nan'), 'rec': float('nan'), 'f1': float('nan')})
                continue
            y_true = split_df['y_true'].values
            y_pred = split_df['y_pred'].values
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics.append({'split': split_name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
            logger.info("Metrics [%s]: acc=%.4f prec=%.4f rec=%.4f f1=%.4f", split_name, acc, prec, rec, f1)

        # save metrics CSV
        try:
            with open(DATASET_METRICS_CSV, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['split','acc','prec','rec','f1'])
                w.writeheader()
                w.writerows(metrics)
            logger.info("Saved dataset metrics to %s", DATASET_METRICS_CSV)
            QMessageBox.information(self, "Dataset Run", f"Dataset run completed. Metrics saved to {DATASET_METRICS_CSV}")
        except Exception as e:
            logger.exception("Failed to save metrics: %s", e)
            QMessageBox.warning(self, "Dataset Run", f"Completed but failed to save metrics: {e}")


# ===== Qt logging helper (defined early so UI can attach handler) =====
class _LogEmitter(QObject):
    log_record = pyqtSignal(str)

class QtLogHandler(logging.Handler):
    """Logging handler that emits records to a QPlainTextEdit via a Qt signal."""
    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self.widget = widget
        self.emitter = _LogEmitter()
        try:
            self.emitter.log_record.connect(self._append)
        except Exception:
            # in case Qt not fully initialized at import
            pass

    @pyqtSlot(str)
    def _append(self, text: str):
        try:
            self.widget.appendPlainText(text)
            self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())
        except Exception:
            pass

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.emitter.log_record.emit(msg)
        except Exception:
            pass

# New batch processing functions
def compute_cluster_patch_crack_ratio(prob_map: np.ndarray, patch_size: int, patch_crack_ratio_thr: float, pix_thr: float = DEFAULT_PIX_THR) -> float:
    """Patch-based cluster crack ratio.

    A patch is cracked if fraction of pixels with prob>=pix_thr > patch_crack_ratio_thr.
    cluster_crack_ratio = cracked_patches / total_patches.
    """
    H, W = prob_map.shape
    cracked_patches = 0
    total_patches = 0
    for r0, c0, r1, c1 in iter_grid(H, W, patch_size):
        patch = prob_map[r0:r1, c0:c1]
        n = patch.size
        if n == 0:
            continue
        cnt = int((patch >= pix_thr).sum())
        frac = cnt / max(1, n)
        total_patches += 1
        if frac > patch_crack_ratio_thr:
            cracked_patches += 1
    if total_patches == 0:
        return 0.0
    return cracked_patches / total_patches


def _run_single_cluster_prob_map(lda, scaler, pos_idx, row: pd.Series) -> dict:
    """Helper to compute prob_map once per cluster; returns dict row for results_df.
    This is used by prepare_and_run_inference; patch-based aggregation is done later.
    """
    grape_id = row.get("grape_id")
    rownum = int(row.get("row", -1))
    image_path = row.get("image_path")
    label = row.get("label")
    status = "ok"
    prob_map_path = ""
    crack_ratio_pix = float("nan")
    try:
        if not isinstance(image_path, str) or not os.path.isdir(image_path):
            # try to tolerate if image_path points inside the cluster
            if os.path.isdir(os.path.dirname(image_path)):
                image_path = os.path.dirname(image_path)
            else:
                status = "missing_folder"
                logger.warning("Missing image folder for grape_id=%s path=%s", grape_id, image_path)
                return {"grape_id": grape_id, "row": rownum, "image_path": image_path, "label": label, "crack_ratio": crack_ratio_pix, "status": status, "prob_map_path": prob_map_path}
        hdr_glob = glob.glob(os.path.join(image_path, "HS", "results", "*.hdr"))
        if not hdr_glob:
            status = "no_hdr"
            logger.warning("No .hdr found for %s (path=%s)", grape_id, image_path)
            return {"grape_id": grape_id, "row": rownum, "image_path": image_path, "label": label, "crack_ratio": crack_ratio_pix, "status": status, "prob_map_path": prob_map_path}
        hdr_path = hdr_glob[0]
        cube = load_cube(hdr_path)
        prob_map = per_pixel_probs(cube, lda, scaler, pos_idx)
        # Save prob map
        prob_dir = os.path.join(os.getcwd(), "prob_maps_late_detection")
        os.makedirs(prob_dir, exist_ok=True)
        safe_grape = str(grape_id).replace(os.sep, "_").replace(" ", "_")
        prob_map_fname = f"prob_{safe_grape}_{rownum}_{row.name}.npy"
        prob_map_path = os.path.join(prob_dir, prob_map_fname)
        np.save(prob_map_path, prob_map.astype(np.float32))
        # simple pixel-based crack ratio (for backward compatibility / debugging)
        mask = (prob_map >= DEFAULT_PIX_THR)
        H, W = prob_map.shape
        crack_ratio_pix = float(mask.sum()) / float(max(1, H * W))
        logger.info("Cluster %s: saved prob_map=%s pixel_crack_ratio=%.4f", grape_id, prob_map_path, crack_ratio_pix)
    except Exception as e:
        logger.exception("Error processing cluster %s: %s", grape_id, e)
        status = "error"
    return {"grape_id": grape_id, "row": rownum, "image_path": image_path, "label": label, "crack_ratio": crack_ratio_pix, "status": status, "prob_map_path": prob_map_path}


def prepare_and_run_inference(input_csv: str, lda_path: str, output_csv: str = "late_detection_with_crack_ratio.csv") -> pd.DataFrame:
    """Batch mode stage 1: compute and cache per-pixel probability maps for all dev+test clusters.

    Returns a DataFrame with columns including prob_map_path; crack_ratio column is the
    legacy pixel-based ratio and is not used for tuning.
    """
    logger.info("Preparing and running per-cluster probability inference on dataset: %s", input_csv)
    df = pd.read_csv(input_csv)
    df = df[df["row"].isin([1, 2])].copy()
    if df.empty:
        logger.warning("No dev/test rows in input CSV")
        return df

    lda, scaler, pos_idx, _ = load_model_and_scaler(lda_path, None)

    # Parallelize over clusters (joblib); fall back to serial if issues
    try:
        rows = [row for _, row in df.iterrows()]
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_run_single_cluster_prob_map)(lda, scaler, pos_idx, r) for r in rows
        )
    except Exception as e:
        logger.exception("Parallel prob_map computation failed, falling back to serial: %s", e)
        results = [_run_single_cluster_prob_map(lda, scaler, pos_idx, r) for _, r in df.iterrows()]

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    logger.info("Saved base inference results (with prob_map_path) to %s (rows=%d)", output_csv, len(results_df))
    logger.info("Dev rows: %d | Test rows: %d", (results_df["row"] == 1).sum(), (results_df["row"] == 2).sum())
    return results_df


def run_grid_search(results_df: pd.DataFrame) -> tuple[Optional[tuple[int, float, float]], list[dict]]:
    """3-parameter grid search on dev set.

    Parameters searched:
    - patch_size in GRID_SIZE_CANDIDATES
    - patch_crack_ratio_thr in GRID_CRACK_RATIO_THR_CANDIDATES
    - cluster_crack_ratio_thr in CLUSTER_CRACK_RATIO_THR_CANDIDATES

    Uses only row==1 (dev) and status=='ok' rows with existing prob_map_path.
    Returns (best_params, metrics_list) where best_params=(patch_size, patch_thr, cluster_thr).
    """
    dev = results_df[(results_df["row"] == 1) & (results_df["status"] == "ok") & results_df["prob_map_path"].notna() & (results_df["prob_map_path"] != "")].copy()
    if dev.empty:
        logger.warning("No valid dev rows for grid search")
        return None, []

    metrics_list: list[dict] = []

    for gs, pthr, cthr in itertools.product(GRID_SIZE_CANDIDATES, GRID_CRACK_RATIO_THR_CANDIDATES, CLUSTER_CRACK_RATIO_THR_CANDIDATES):
        y_true: list[int] = []
        y_pred: list[int] = []
        y_scores: list[float] = []  # For ROC AUC calculation
        for _, r in dev.iterrows():
            prob_path = r["prob_map_path"]
            if not isinstance(prob_path, str) or not os.path.exists(prob_path):
                continue
            try:
                prob_map = np.load(prob_path)
                cluster_ratio = compute_cluster_patch_crack_ratio(prob_map, int(gs), float(pthr))
                pred = 1 if cluster_ratio >= float(cthr) else 0
                y_pred.append(pred)
                y_true.append(int(r["label"]))
                y_scores.append(cluster_ratio)
            except Exception as e:
                logger.debug("Grid search: failed on %s: %s", prob_path, e)
        if not y_true:
            dev_acc = dev_prec = dev_rec = dev_f1 = dev_roc_auc = float("nan")
        else:
            dev_acc = accuracy_score(y_true, y_pred)
            dev_prec = precision_score(y_true, y_pred, zero_division=0)
            dev_rec = recall_score(y_true, y_pred, zero_division=0)
            dev_f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                dev_roc_auc = roc_auc_score(y_true, y_scores)
            except Exception:
                dev_roc_auc = float("nan")
        metrics_list.append({
            "patch_size": int(gs),
            "patch_thr": float(pthr),
            "cluster_thr": float(cthr),
            "dev_acc": dev_acc,
            "dev_prec": dev_prec,
            "dev_rec": dev_rec,
            "dev_f1": dev_f1,
            "dev_roc_auc": dev_roc_auc,
        })

    # save grid metrics
    save_threshold_metrics_csv(metrics_list, METRICS_OUTPUT_CSV)

    valid = [m for m in metrics_list if not np.isnan(m["dev_f1"])]
    if not valid:
        logger.warning("Grid search produced no valid metrics")
        return None, metrics_list
    best = max(valid, key=lambda m: (m["dev_f1"], m["dev_prec"]))
    best_params = (int(best["patch_size"]), float(best["patch_thr"]), float(best["cluster_thr"]))
    logger.info("Best grid params: patch_size=%d patch_thr=%.3f cluster_thr=%.3f (dev_f1=%.4f dev_prec=%.4f)", best_params[0], best_params[1], best_params[2], best["dev_f1"], best["dev_prec"])
    return best_params, metrics_list


def evaluate_on_test_with_params(results_df: pd.DataFrame, params: tuple[int, float, float]) -> dict:
    """Run final evaluation on test set (row==2) using best patch-based hyperparameters.

    Updates results_df with new cluster_crack_ratio and pred_label for test rows and returns
    a dict of test metrics.
    """
    if params is None:
        logger.warning("No hyperparameters provided for test evaluation")
        return {}
    patch_size, patch_thr, cluster_thr = params
    test = results_df[(results_df["row"] == 2) & results_df["prob_map_path"].notna() & (results_df["prob_map_path"] != "")].copy()
    if test.empty:
        logger.warning("No valid test rows to evaluate")
        return {}

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []  # For ROC AUC calculation

    for idx, r in test.iterrows():
        prob_path = r["prob_map_path"]
        try:
            prob_map = np.load(prob_path)
            cluster_ratio = compute_cluster_patch_crack_ratio(prob_map, patch_size, patch_thr)
            pred = 1 if cluster_ratio >= cluster_thr else 0
            results_df.at[idx, "crack_ratio"] = cluster_ratio
            results_df.at[idx, "pred_label"] = pred
            y_true.append(int(r["label"]))
            y_pred.append(pred)
            y_scores.append(cluster_ratio)  # Use cluster_ratio as confidence score
        except Exception as e:
            logger.debug("Test eval: failed on %s: %s", prob_path, e)

    metrics: dict[str, float] = {}
    if y_true:
        metrics["test_acc"] = accuracy_score(y_true, y_pred)
        metrics["test_prec"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["test_rec"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["test_f1"] = f1_score(y_true, y_pred, zero_division=0)
        # Calculate ROC AUC using cluster_ratio as scores
        try:
            metrics["test_roc_auc"] = roc_auc_score(y_true, y_scores)
        except Exception as e:
            logger.warning("Failed to calculate ROC AUC: %s", e)
            metrics["test_roc_auc"] = float("nan")
        logger.info("Test metrics -> acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f",
                    metrics["test_acc"], metrics["test_prec"], metrics["test_rec"],
                    metrics["test_f1"], metrics.get("test_roc_auc", float("nan")))
    else:
        logger.warning("No valid predictions for test metrics")
    return metrics


def find_band_index_for_wavelength(hdr_path: str, target_wavelength: Optional[float] = None) -> int:
    """Find band index closest to target_wavelength using ENVI header metadata if available.

    If metadata doesn't contain wavelengths, approximate with a linear 400-1000nm grid.
    If target_wavelength is None, default to ~753nm.
    """
    if target_wavelength is None:
        target_wavelength = 753.0
    try:
        img = spy.envi.open(hdr_path)
        meta = getattr(img, "metadata", {}) or {}
        wls = meta.get("wavelength") or meta.get("wavelengths")
        if isinstance(wls, list) and wls:
            try:
                wls_f = [float(w) for w in wls]
                diffs = [abs(w - target_wavelength) for w in wls_f]
                idx = int(np.argmin(diffs))
                logger.info("Selected band %d for ~%.1f nm from metadata", idx, target_wavelength)
                return idx
            except Exception:
                logger.debug("Failed to parse wavelength list from metadata")
    except Exception:
        logger.debug("No usable ENVI metadata for %s", hdr_path)

    # fallback: approximate from number of bands
    try:
        cube = load_cube(hdr_path)
        bands = cube.shape[2]
        low, high = 400.0, 1000.0
        frac = max(0.0, min(1.0, (target_wavelength - low) / (high - low)))
        idx = int(round(frac * (bands - 1)))
        logger.info("Estimated band %d for ~%.1f nm using linear spacing", idx, target_wavelength)
        return idx
    except Exception:
        logger.exception("Failed to estimate band index; defaulting to 0")
        return 0


def _find_rgb_for_cluster_folder(folder: str) -> Optional[np.ndarray]:
    """Try to locate a standard RGB/Canon image under the cluster folder and load as RGB array."""
    if not folder or not os.path.isdir(folder):
        return None
    candidate = None
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                if any(k in f.lower() for k in ("canon", "rgb")):
                    candidate = os.path.join(root, f)
                    break
        if candidate:
            break
    if not candidate:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    candidate = os.path.join(root, f)
                    break
            if candidate:
                break
    if not candidate:
        return None
    bgr = cv2.imread(candidate)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ...existing visualize_results or add new CLI visualizer...

# Modify main to add batch mode handling
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI LDA Patch Classifier / Batch Late Detection Inference")
    parser.add_argument('--folder', type=str, help='Folder to load images from (UI mode)')
    parser.add_argument('--detect_only', action='store_true', help='Run detection only (UI mode)')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout in seconds to force exit (default: 10)')

    # Batch mode args
    parser.add_argument('--input_csv', type=str, default='late_detection_dataset.csv', help='CSV dataset for late detection (batch mode)')
    parser.add_argument('--model', type=str, default=list(AVAILABLE_MODELS.values())[0], help='Path to LDA model for batch inference')
    parser.add_argument('--output', type=str, default='late_detection_with_crack_ratio.csv', help='Output CSV for base per-cluster results')
    parser.add_argument('--visualize', action='store_true', help='Open interactive viewer over test results after grid search')
    parser.add_argument('--target_wavelength', type=float, default=None, help='Target wavelength (nm) for HSI band visualization (default ~753nm)')

    args = parser.parse_args()

    # ----- Pure CLI workflow: if input_csv exists, run batch grid-search pipeline and exit -----
    if args.input_csv and os.path.exists(args.input_csv):
        # Stage 1: compute prob_maps for all dev+test clusters
        base_df = prepare_and_run_inference(args.input_csv, args.model, args.output)
        # Stage 2: 3-parameter grid search on dev rows
        best_params, metrics_list = run_grid_search(base_df)
        # Stage 3: evaluate on test using best hyperparameters
        test_metrics = evaluate_on_test_with_params(base_df, best_params)

        # Save combined results and test metrics
        try:
            base_df.to_csv("late_detection_patch_based_results.csv", index=False)
            logger.info("Saved patch-based results to late_detection_patch_based_results.csv")
        except Exception:
            logger.exception("Failed to save patch-based results CSV")
        try:
            with open("late_detection_patch_based_test_metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["metric", "value"])
                w.writeheader()
                for k, v in (test_metrics or {}).items():
                    w.writerow({"metric": k, "value": v})
            logger.info("Saved test metrics to late_detection_patch_based_test_metrics.csv")
        except Exception:
            logger.exception("Failed to save test metrics CSV")

        # Optional visualization over test set
        if args.visualize:
            test_vis = base_df[(base_df["row"] == 2) & (base_df["status"] == "ok") & base_df["prob_map_path"].notna() & (base_df["prob_map_path"] != "")].reset_index(drop=True)
            if test_vis.empty:
                print("No valid test rows to visualize.")
            else:
                idx = 0
                while 0 <= idx < len(test_vis):
                    r = test_vis.loc[idx]
                    image_path = r["image_path"]
                    prob_path = r["prob_map_path"]
                    # locate HDR
                    hdr_glob = glob.glob(os.path.join(image_path, "HS", "results", "*.hdr")) if isinstance(image_path, str) else []
                    if not hdr_glob:
                        idx += 1
                        continue
                    hdr_path = hdr_glob[0]
                    try:
                        # --- Load HSI cube ---
                        cube = load_cube(hdr_path)
                        H, W, B = cube.shape

                        # --- 1) HSI single band near target wavelength (e.g. 753nm) ---
                        band_idx = find_band_index_for_wavelength(hdr_path, args.target_wavelength)
                        band_idx = max(0, min(B - 1, band_idx))
                        band_img = cv2.normalize(cube[:, :, band_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        # Rotate 90° to align with classification_by_patch orientation
                        band_img_rot = cv2.rotate(band_img, cv2.ROTATE_90_CLOCKWISE)
                        band_rgb = cv2.cvtColor(band_img_rot, cv2.COLOR_GRAY2RGB)
                        band_rgb = cv2.resize(band_rgb, (256, 256))

                        # --- 2) HSI detection mask at DEFAULT_PIX_THR ---
                        overlay_det = np.full_like(band_rgb, 80, dtype=np.uint8)
                        if isinstance(prob_path, str) and os.path.exists(prob_path):
                            prob_map = np.load(prob_path)
                            # rotate prob_map the same way as band image so they align
                            prob_rot = cv2.rotate(prob_map, cv2.ROTATE_90_CLOCKWISE)
                            mask = (prob_rot >= DEFAULT_PIX_THR).astype(np.uint8) * 255
                            mask = cv2.resize(mask, (256, 256))
                            overlay_det = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

                        # --- 3) HSI patch/grid visualization (using same prob_map) ---
                        overlay_grid = np.full_like(band_rgb, 120, dtype=np.uint8)
                        try:
                            if isinstance(prob_path, str) and os.path.exists(prob_path):
                                # work in original orientation for grid, then rotate to match display
                                prob_map_full = np.load(prob_path)
                                # Use best patch size and thresholds if available, otherwise defaults
                                patch_size, patch_thr, _ = best_params if best_params is not None else (DEFAULT_CELL_SIZE, 0.10, 0.05)
                                # Build synthetic grid stats from prob_map_full at DEFAULT_PIX_THR
                                grid_stats = analyze_grid(prob_map_full, int(patch_size), float(DEFAULT_PIX_THR))
                                # base band for grid rendering (same band index as above but unrotated)
                                band_base = cv2.normalize(cube[:, :, band_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                grid_img = overlay_on_band(band_base, grid_stats, alpha=0.45)
                                grid_img_rot = cv2.rotate(grid_img, cv2.ROTATE_90_CLOCKWISE)
                                overlay_grid = cv2.resize(grid_img_rot, (256, 256))
                        except Exception as e_vis_grid:
                            logger.debug("Grid visualization failed for %s: %s", r.get("grape_id"), e_vis_grid)

                        # --- 4) RGB image derived from HSI (3-band composite) ---
                        # Many pipelines build an RGB by picking 3 bands; here we approximate with 3 fixed indices.
                        try:
                            # choose three bands spread across spectrum
                            b_idx = max(0, min(B - 1, int(B * 0.1)))
                            g_idx = max(0, min(B - 1, int(B * 0.5)))
                            r_idx = max(0, min(B - 1, int(B * 0.9)))
                            hsi_r = cv2.normalize(cube[:, :, r_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            hsi_g = cv2.normalize(cube[:, :, g_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            hsi_b = cv2.normalize(cube[:, :, b_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            hsi_rgb = cv2.merge([hsi_r, hsi_g, hsi_b])  # R,G,B order
                            hsi_rgb_rot = cv2.rotate(hsi_rgb, cv2.ROTATE_90_CLOCKWISE)
                            hsi_rgb_disp = cv2.resize(hsi_rgb_rot, (256, 256))
                        except Exception:
                            # fallback to band_rgb if anything goes wrong
                            hsi_rgb_disp = band_rgb.copy()

                        # --- Stack the 4 panels horizontally ---
                        combined = cv2.hconcat([band_rgb, overlay_det, overlay_grid, hsi_rgb_disp])

                        # Add text overlay with key metadata
                        txt = f"id={r['grape_id']} label={int(r['label'])} pred={int(r.get('pred_label', 0))} crack_ratio={float(r.get('crack_ratio', float('nan'))):.4f}"
                        cv2.putText(combined, txt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                        win = "Late detection test viewer (HSI 753nm | HSI detection | HSI patches | HSI-RGB)"
                        cv2.imshow(win, combined)
                        key = cv2.waitKey(0) & 0xFF
                        if key in (ord('q'), 27):
                            cv2.destroyAllWindows()
                            break
                        elif key == ord('n'):
                            idx += 1
                        elif key == ord('p'):
                            idx = max(0, idx - 1)
                        else:
                            idx += 1
                    except Exception as e:
                        logger.exception("Failed to visualize row %s: %s", r.get("grape_id"), e)
                        idx += 1
                cv2.destroyAllWindows()
        sys.exit(0)

def save_threshold_metrics_csv(metrics_list: list[dict], path: str = METRICS_OUTPUT_CSV) -> None:
    """Save grid-search metrics (one row per hyperparameter combo) to CSV.

    Each element in metrics_list is a dict with keys like:
    patch_size, patch_thr, cluster_thr, dev_acc, dev_prec, dev_rec, dev_f1.
    """
    if not metrics_list:
        logger.info("No grid-search metrics to save.")
        return
    try:
        fieldnames = list(metrics_list[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)
        logger.info("Saved grid-search metrics to %s (rows=%d)", path, len(metrics_list))
    except Exception as e:
        logger.exception("Failed to save grid-search metrics to %s: %s", path, e)
