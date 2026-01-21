"""
Late Detection Core Logic - Batch Processing and Inference Pipeline

This module contains all the heavy lifting for late detection:
- Model loading and prediction
- Per-pixel probability computation
- Grid-based patch analysis
- Hyperparameter grid search
- Dataset batch processing
"""

import os
import sys
import csv
import logging
import time
import glob
import itertools
from typing import Optional, Tuple, List, Dict
import argparse

import numpy as np
import pandas as pd
import joblib
import cv2
import spectral as spy
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from joblib import Parallel, delayed

# ===== Hyperparameter search config =====
GRID_SIZE_CANDIDATES = [16, 32, 48, 64]
GRID_CRACK_RATIO_THR_CANDIDATES = [0.05, 0.10, 0.15, 0.20]  # per-patch crack_fraction threshold
CLUSTER_CRACK_RATIO_THR_CANDIDATES = [0.02, 0.05, 0.10, 0.15, 0.20]  # threshold on cluster-level crack_ratio
METRICS_OUTPUT_CSV = "late_detection_hparam_metrics.csv"

# ===== Dataset config =====
DEFAULT_CELL_SIZE = 64
DEFAULT_PIX_THR = 0.5

# ===== Grid visualization config =====
GRID_COLOR_BUCKETS = [(50, (0, 0, 180)), (40, (0, 30, 200)), (30, (0, 120, 255)), (20, (0, 200, 255))]

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HSI.Patch.Core")
if os.environ.get("HSI_DEBUG") == "1":
    logger.setLevel(logging.DEBUG)


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


class DecisionTreeModel:
    """Minimal shim: restore state and delegate predict_proba to inner estimator if present."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
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


# ===== Helper functions =====

def _predict_proba_any(model, X):
    """Try common attributes to find a classifier with predict_proba."""
    for attr in (None, "estimator_", "clf", "model", "lda", "tree", "classifier", "classifier_"):
        m = getattr(model, attr, model) if attr else model
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)
    raise AttributeError("No predict_proba on provided model")


def _classes_any(model):
    """Robustly fetch class labels from the model/pipeline."""
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    # If wrapped in a pipeline, try to find final estimator
    est = getattr(model, "named_steps", {}).get("model", None) or getattr(model, "model", None)
    if est is not None and hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    # safe fallback
    return np.asarray([0, 1])


def _get_model_expected_features(model) -> Optional[int]:
    """Try to infer number of input features the model expects."""
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
                if hasattr(coef, "shape") and len(coef.shape) >= 1:
                    return int(coef.shape[-1])
            except Exception:
                pass
    return None


def find_scaler(path_dir: str, exclude: Optional[List[str]] = None) -> Optional[str]:
    """Find a scaler file in the given directory."""
    exclude = set(exclude or [])
    cands = [f for f in os.listdir(path_dir) if f.lower().endswith((".joblib", ".pkl")) and f not in exclude]
    for name in cands:  # prefer names that look like a scaler
        if any(k in name.lower() for k in ("scaler", "standard", "std")):
            p = os.path.join(path_dir, name)
            try:
                if hasattr(joblib.load(p), "transform"):
                    return p
            except Exception:
                pass
    # fallback: any obj with transform
    for name in cands:
        p = os.path.join(path_dir, name)
        try:
            if hasattr(joblib.load(p), "transform"):
                return p
        except Exception:
            pass
    return None


def load_model_and_scaler(lda_path: str, scaler_path: Optional[str] = None) -> Tuple[object, Optional[object], int, np.ndarray, Optional[float]]:
    """Load model and scaler, return (model, scaler, pos_idx, classes, optimal_threshold)."""
    logger.info("Loading model: %s", lda_path)
    try:
        import types
        main_mod = sys.modules.get("__main__")
        if main_mod is None:
            main_mod = types.ModuleType("__main__")
            sys.modules["__main__"] = main_mod
        # Inject shims so joblib/pickle can resolve custom classes
        for cls in (LDAModel, DecisionTreeModel):
            setattr(main_mod, cls.__name__, cls)
    except Exception:
        logger.debug("Failed to inject shim classes into __main__ (non-fatal)")

    try:
        loaded = joblib.load(lda_path)
    except AttributeError as e:
        if "Can't get attribute" in str(e):
            model_name = os.path.basename(lda_path)
            logger.error(f"Failed to load {model_name}: Model not compatible.")
            raise ValueError(f"Model {model_name} is not compatible") from e
        raise

    # Handle dictionary structure (new F1-optimized models)
    optimal_threshold = None
    if isinstance(loaded, dict):
        lda = loaded.get('model')
        optimal_threshold = loaded.get('optimal_threshold')
        if lda is None:
            raise ValueError(f"Dictionary model missing 'model' key: {lda_path}")
        if optimal_threshold is not None:
            logger.info("Loaded model with optimal_threshold: %.4f", optimal_threshold)
    else:
        # Legacy raw model object
        lda = loaded

    if scaler_path is None:
        scaler_path = find_scaler(os.path.dirname(lda_path), [os.path.basename(lda_path)])
    scaler = joblib.load(scaler_path) if scaler_path else None
    classes = _classes_any(lda)

    # Try to load label encoder if it exists (for XGBoost models with numeric classes)
    label_encoder_path = os.path.join(os.path.dirname(lda_path), "label_encoder.joblib")
    if os.path.exists(label_encoder_path) and len(classes) > 0 and not isinstance(classes[0], str):
        try:
            label_encoder = joblib.load(label_encoder_path)
            if hasattr(label_encoder, 'classes_'):
                # Map numeric classes to string labels
                classes = np.array([label_encoder.inverse_transform([int(c)])[0] for c in classes])
                logger.info("Loaded label encoder and mapped numeric classes to: %s", classes.tolist())
        except Exception as e:
            logger.warning(f"Failed to load label encoder: {e}")

    pos_idx = 0
    if len(classes) > 0 and isinstance(classes[0], str):
        crack_candidates = ['CRACK', 'crack', 'Crack', 'DECAY', 'decay']
        for i, cls in enumerate(classes):
            if cls in crack_candidates:
                pos_idx = i
                break

    logger.info("Model loaded. classes=%s | pos_idx=%d | scaler=%s | optimal_threshold=%s",
                classes.tolist(), pos_idx, bool(scaler), optimal_threshold)
    expected = _get_model_expected_features(lda)
    logger.info("Model expected input features: %s", expected)
    return lda, scaler, pos_idx, classes, optimal_threshold


def load_cube(hdr_path: str) -> np.ndarray:
    """Load hyperspectral cube from ENVI header."""
    t0 = time.perf_counter()
    dat_path = hdr_path.replace(".hdr", ".dat")
    logger.info(f"Loading HSI cube: {hdr_path}")
    cube = np.array(spy.envi.open(hdr_path, dat_path).load())
    logger.info(f"Loaded cube shape: {cube.shape}, dtype: {cube.dtype} (%.2fs)",
                time.perf_counter() - t0)
    return cube


def per_pixel_probs(cube: np.ndarray, lda: object, scaler: Optional[object], pos_idx: int) -> np.ndarray:
    """Compute per-pixel crack probabilities."""
    t0 = time.perf_counter()
    H, W, C = cube.shape
    X_raw = np.nan_to_num(cube.reshape(-1, C), copy=False)

    # Prepare candidate feature preps and test which one the model accepts
    candidates = [("raw", X_raw[:min(200, X_raw.shape[0])])]
    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            X_scaled_sample = scaler.transform(X_raw[:min(200, X_raw.shape[0])])
            candidates.append(("scaled", X_scaled_sample))
        except Exception as e:
            logger.warning(f"Scaler transform failed on sample: {e}")

    chosen = None
    for name, sample_mat in candidates:
        try:
            _predict_proba_any(lda, sample_mat)
            chosen = name
            logger.info(f"Using feature prep: {chosen}")
            break
        except Exception as e:
            logger.debug(f"Candidate '{name}' rejected: {e}")

    if chosen is None:
        logger.warning("No compatible feature preparation found. Falling back to raw features.")
        chosen = "raw"

    # Apply the chosen preparation
    X = X_raw
    if chosen == "scaled":
        try:
            X = scaler.transform(X_raw)
        except Exception as e:
            logger.warning(f"Failed to apply scaler to full X: {e}. Falling back to raw.")
            X = X_raw

    # Predict probabilities
    try:
        probs = _predict_proba_any(lda, X)
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=-1)
        if pos_idx >= probs.shape[1]:
            logger.warning(f"pos_idx {pos_idx} out of bounds, using last index.")
            pos_idx = probs.shape[1] - 1
        probs = probs[:, pos_idx].reshape(H, W)
    except Exception as e:
        logger.error(f"Error in per_pixel_probs: {e}")
        probs = np.zeros((H, W), dtype=np.float32)

    logger.info("Per-pixel probs: time=%.2fs | min=%.4f max=%.4f mean=%.4f",
                time.perf_counter()-t0, float(probs.min()), float(probs.max()), float(probs.mean()))
    return probs


def iter_grid(H: int, W: int, cell: int):
    """Iterator for grid patches."""
    for r0 in range(0, H, cell):
        r1 = min(r0 + cell, H)
        for c0 in range(0, W, cell):
            c1 = min(c0 + cell, W)
            yield r0, c0, r1, c1


def analyze_grid(prob_map: np.ndarray, cell: int, pix_thr: float) -> List[Dict]:
    """Analyze grid patches and return statistics."""
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
    logger.info("Grid summary: patches=%d | mean=%.2f%% p50=%.2f%% p90=%.2f%%",
                len(out), mean_p, p50, p90)
    return out


def filter_blobs_by_shape(proba, thr, *, border_r=20, area_min=10, area_max=5000,
                          circularity_min=0.0, circularity_max=1.0,
                          use_border=True, use_area=True, use_shape=True):
    """Filter blobs by border proximity, area constraints, and shape geometry.

    Three independent filters:
    - Border filter: Remove blobs near image edges
    - Area filter: Remove blobs outside size range
    - Shape filter: Remove blobs outside circularity range

    Circularity = 4π * Area / Perimeter²
    - Circle = 1.0 (most compact)
    - Square ≈ 0.785
    - Long thin shapes < 0.5
    """
    H, W = proba.shape
    mask = ((proba >= thr).astype("uint8") * 255)
    out = np.zeros_like(mask)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in contours:
        # Area filter
        A = cv2.contourArea(c)
        if use_area and (A < area_min or A > area_max):
            continue

        # Border filter
        x, y, w, h = cv2.boundingRect(c)
        near_border = (x <= border_r or y <= border_r or
                      (x + w) >= (W - 1 - border_r) or
                      (y + h) >= (H - 1 - border_r))
        if use_border and near_border:
            continue

        # Shape filter (circularity/compactness)
        if use_shape and A > 0:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = (4 * np.pi * A) / (perimeter * perimeter)
                if circularity < circularity_min or circularity > circularity_max:
                    continue

        cv2.drawContours(out, [c], -1, 255, -1)
    return (out > 0)


def filter_blobs_advanced(proba, thr, *, morph_size: int = 0, border_r=20, area_min=10, area_max=5000,
                          circularity_min: Optional[float] = None, circularity_max: Optional[float] = None,
                          aspect_ratio_min: Optional[float] = None, aspect_ratio_max: Optional[float] = None,
                          solidity_min: Optional[float] = None, solidity_max: Optional[float] = None,
                          use_border=True, use_area=True, use_shape=True,
                          use_circularity: Optional[bool] = None, use_aspect_ratio: Optional[bool] = None, use_solidity: Optional[bool] = None):
    """Advanced blob filtering with optional morphological closing and geometric filters.

    Parameters:
        proba: 2D array of per-pixel probabilities
        thr: probability threshold for binarization
        morph_size: kernel size for morphological closing (0 = disabled)
        border_r: distance in pixels from image border to consider as "near border"
        area_min/area_max: area bounds to keep blobs
        circularity_min/max: keep blobs within circularity range (None = disabled)
        aspect_ratio_min/max: keep blobs within aspect ratio range (None = disabled)
        solidity_min/max: keep blobs within solidity range (None = disabled)
        use_border/use_area/use_shape: legacy flags to enable/disable corresponding checks
        use_circularity/use_aspect_ratio/use_solidity: individual toggles for each filter (None = use use_shape)

    Returns:
        Boolean mask (same shape as proba) where True indicates kept pixels.
    """
    H, W = proba.shape
    # Binarize
    mask = ((proba >= thr).astype("uint8") * 255)

    # Morphological closing (merge fragmented pixel clusters)
    if morph_size and int(morph_size) > 0:
        try:
            k = int(morph_size)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            logger.debug("Applied morphological closing with kernel=%d", k)
        except Exception:
            logger.exception("Morphological closing failed; continuing without it")

    out = np.zeros_like(mask)

    # Robustly handle OpenCV findContours return signature
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    logger.debug("Found %d contours before advanced filtering", len(contours))

    for c in contours:
        # Area
        A = cv2.contourArea(c)
        if use_area and (A < area_min or A > area_max):
            continue

        # Bounding box and border proximity
        x, y, w, h = cv2.boundingRect(c)
        near_border = (x <= border_r or y <= border_r or
                      (x + w) >= (W - 1 - border_r) or
                      (y + h) >= (H - 1 - border_r))
        if use_border and near_border:
            continue

        # Geometric filters
        use_circ = use_circularity if use_circularity is not None else use_shape
        use_ar = use_aspect_ratio if use_aspect_ratio is not None else use_shape
        use_sol = use_solidity if use_solidity is not None else use_shape

        # Compute circularity if needed
        circularity = None
        if use_circ and (circularity_min is not None or circularity_max is not None) and A > 0:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = (4 * np.pi * A) / (perimeter * perimeter)
            else:
                circularity = 0.0

        # Circularity filters
        if use_circ and circularity is not None:
            if circularity_min is not None and circularity < circularity_min:
                continue
            if circularity_max is not None and circularity > circularity_max:
                continue

        # Compute aspect ratio if needed
        aspect = None
        if use_ar and (aspect_ratio_min is not None or aspect_ratio_max is not None) and h > 0 and w > 0:
            aspect = max(float(w) / float(h), float(h) / float(w))

        # Aspect ratio filters
        if use_ar and aspect is not None:
            if aspect_ratio_min is not None and aspect < aspect_ratio_min:
                continue
            if aspect_ratio_max is not None and aspect > aspect_ratio_max:
                continue

        # Compute solidity if needed
        solidity = None
        if use_sol and (solidity_min is not None or solidity_max is not None):
            try:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = (A / hull_area) if hull_area > 0 else 0.0
            except Exception:
                solidity = 0.0

        # Solidity filters
        if use_sol and solidity is not None:
            if solidity_min is not None and solidity < solidity_min:
                continue
            if solidity_max is not None and solidity > solidity_max:
                continue

        # If passed all filters, draw into output mask
        cv2.drawContours(out, [c], -1, 255, -1)

    logger.debug("Contours kept after filtering: %d", int((out > 0).sum()))
    return (out > 0)


def compute_cluster_patch_crack_ratio(prob_map: np.ndarray, patch_size: int,
                                     patch_crack_ratio_thr: float,
                                     pix_thr: float = DEFAULT_PIX_THR) -> float:
    """Compute cluster-level crack ratio based on patch analysis.

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


def find_band_index_for_wavelength(hdr_path: str, target_wavelength: Optional[float] = None) -> int:
    """Find band index closest to target_wavelength.

    Uses ENVI header metadata if available, otherwise approximates with linear spacing.
    If target_wavelength is None, defaults to ~753nm.
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
        dat_path = hdr_path.replace(".hdr", ".dat")
        cube = np.array(spy.envi.open(hdr_path, dat_path).load())
        bands = cube.shape[2]
        low, high = 400.0, 1000.0
        frac = max(0.0, min(1.0, (target_wavelength - low) / (high - low)))
        idx = int(round(frac * (bands - 1)))
        logger.info("Estimated band %d for ~%.1f nm using linear spacing", idx, target_wavelength)
        return idx
    except Exception:
        logger.exception("Failed to estimate band index; defaulting to 0")
        return 0


# ===== Batch processing functions =====

def _run_single_cluster_prob_map(lda, scaler, pos_idx, row: pd.Series) -> dict:
    """Helper to compute prob_map once per cluster; returns dict row for results_df."""
    grape_id = row.get("grape_id")
    rownum = int(row.get("row", -1))
    image_path = row.get("image_path")
    label = row.get("label")
    status = "ok"
    prob_map_path = ""
    crack_ratio_pix = float("nan")

    try:
        if not isinstance(image_path, str) or not os.path.isdir(image_path):
            if os.path.isdir(os.path.dirname(image_path)):
                image_path = os.path.dirname(image_path)
            else:
                status = "missing_folder"
                logger.warning("Missing image folder for grape_id=%s path=%s", grape_id, image_path)
                return {
                    "grape_id": grape_id, "row": rownum, "image_path": image_path,
                    "label": label, "crack_ratio": crack_ratio_pix, "status": status,
                    "prob_map_path": prob_map_path
                }

        hdr_glob = glob.glob(os.path.join(image_path, "HS", "results", "*.hdr"))
        if not hdr_glob:
            status = "no_hdr"
            logger.warning("No .hdr found for %s (path=%s)", grape_id, image_path)
            return {
                "grape_id": grape_id, "row": rownum, "image_path": image_path,
                "label": label, "crack_ratio": crack_ratio_pix, "status": status,
                "prob_map_path": prob_map_path
            }

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

        # Simple pixel-based crack ratio (for backward compatibility)
        mask = (prob_map >= DEFAULT_PIX_THR)
        H, W = prob_map.shape
        crack_ratio_pix = float(mask.sum()) / float(max(1, H * W))
        logger.info("Cluster %s: saved prob_map=%s pixel_crack_ratio=%.4f",
                   grape_id, prob_map_path, crack_ratio_pix)
    except Exception as e:
        logger.exception("Error processing cluster %s: %s", grape_id, e)
        status = "error"

    return {
        "grape_id": grape_id, "row": rownum, "image_path": image_path,
        "label": label, "crack_ratio": crack_ratio_pix, "status": status,
        "prob_map_path": prob_map_path
    }


def prepare_and_run_inference(input_csv: str, lda_path: str,
                              output_csv: str = "late_detection_with_crack_ratio.csv") -> pd.DataFrame:
    """Batch mode stage 1: compute and cache per-pixel probability maps for all dev+test clusters."""
    logger.info("Preparing and running per-cluster probability inference on dataset: %s", input_csv)
    df = pd.read_csv(input_csv)
    df = df[df["row"].isin([1, 2])].copy()
    if df.empty:
        logger.warning("No dev/test rows in input CSV")
        return df

    lda, scaler, pos_idx, _, _ = load_model_and_scaler(lda_path, None)

    # Parallelize over clusters
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
    logger.info("Saved base inference results to %s (rows=%d)", output_csv, len(results_df))
    logger.info("Dev rows: %d | Test rows: %d",
               (results_df["row"] == 1).sum(), (results_df["row"] == 2).sum())
    return results_df


def run_grid_search(results_df: pd.DataFrame) -> tuple[Optional[tuple[int, float, float]], list[dict]]:
    """3-parameter grid search on dev set.

    Parameters searched:
    - patch_size in GRID_SIZE_CANDIDATES
    - patch_crack_ratio_thr in GRID_CRACK_RATIO_THR_CANDIDATES
    - cluster_crack_ratio_thr in CLUSTER_CRACK_RATIO_THR_CANDIDATES

    Returns (best_params, metrics_list) where best_params=(patch_size, patch_thr, cluster_thr).
    """
    dev = results_df[
        (results_df["row"] == 1) &
        (results_df["status"] == "ok") &
        results_df["prob_map_path"].notna() &
        (results_df["prob_map_path"] != "")
    ].copy()

    if dev.empty:
        logger.warning("No valid dev rows for grid search")
        return None, []

    metrics_list: list[dict] = []

    for gs, pthr, cthr in itertools.product(
        GRID_SIZE_CANDIDATES,
        GRID_CRACK_RATIO_THR_CANDIDATES,
        CLUSTER_CRACK_RATIO_THR_CANDIDATES
    ):
        y_true: list[int] = []
        y_pred: list[int] = []
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
            except Exception as e:
                logger.debug("Grid search: failed on %s: %s", prob_path, e)

        if not y_true:
            dev_acc = dev_prec = dev_rec = dev_f1 = float("nan")
        else:
            dev_acc = accuracy_score(y_true, y_pred)
            dev_prec = precision_score(y_true, y_pred, zero_division=0)
            dev_rec = recall_score(y_true, y_pred, zero_division=0)
            dev_f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics_list.append({
            "patch_size": int(gs),
            "patch_thr": float(pthr),
            "cluster_thr": float(cthr),
            "dev_acc": dev_acc,
            "dev_prec": dev_prec,
            "dev_rec": dev_rec,
            "dev_f1": dev_f1,
        })

    # Save grid metrics
    save_threshold_metrics_csv(metrics_list, METRICS_OUTPUT_CSV)

    valid = [m for m in metrics_list if not np.isnan(m["dev_f1"])]
    if not valid:
        logger.warning("Grid search produced no valid metrics")
        return None, metrics_list

    best = max(valid, key=lambda m: (m["dev_f1"], m["dev_prec"]))
    best_params = (int(best["patch_size"]), float(best["patch_thr"]), float(best["cluster_thr"]))
    logger.info("Best grid params: patch_size=%d patch_thr=%.3f cluster_thr=%.3f (dev_f1=%.4f dev_prec=%.4f)",
               best_params[0], best_params[1], best_params[2], best["dev_f1"], best["dev_prec"])
    return best_params, metrics_list


def evaluate_on_test_with_params(results_df: pd.DataFrame, params: tuple[int, float, float]) -> dict:
    """Run final evaluation on test set using best patch-based hyperparameters."""
    if params is None:
        logger.warning("No hyperparameters provided for test evaluation")
        return {}

    patch_size, patch_thr, cluster_thr = params
    test = results_df[
        (results_df["row"] == 2) &
        results_df["prob_map_path"].notna() &
        (results_df["prob_map_path"] != "")
    ].copy()

    if test.empty:
        logger.warning("No valid test rows to evaluate")
        return {}

    y_true: list[int] = []
    y_pred: list[int] = []

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
        except Exception as e:
            logger.debug("Test eval: failed on %s: %s", prob_path, e)

    metrics: dict[str, float] = {}
    if y_true:
        metrics["test_acc"] = accuracy_score(y_true, y_pred)
        metrics["test_prec"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["test_rec"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["test_f1"] = f1_score(y_true, y_pred, zero_division=0)
        logger.info("Test metrics -> acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
                   metrics["test_acc"], metrics["test_prec"], metrics["test_rec"], metrics["test_f1"])
    else:
        logger.warning("No valid predictions for test metrics")
    return metrics


def save_threshold_metrics_csv(metrics_list: list[dict], path: str = METRICS_OUTPUT_CSV) -> None:
    """Save grid-search metrics to CSV."""
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


# ===== CLI Entry Point =====
def main():
    """CLI entry point for batch processing."""
    parser = argparse.ArgumentParser(description="HSI Late Detection - Batch Processing Pipeline")
    parser.add_argument('--input_csv', type=str, required=True,
                       help='CSV dataset for late detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to LDA model for batch inference')
    parser.add_argument('--output', type=str, default='late_detection_with_crack_ratio.csv',
                       help='Output CSV for base per-cluster results')
    parser.add_argument('--target_wavelength', type=float, default=753.0,
                       help='Target wavelength (nm) for visualization (default: 753)')

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logger.error("Input CSV not found: %s", args.input_csv)
        sys.exit(1)

    if not os.path.exists(args.model):
        logger.error("Model file not found: %s", args.model)
        sys.exit(1)

    # Stage 1: Compute prob_maps for all dev+test clusters
    logger.info("=" * 60)
    logger.info("STAGE 1: Computing probability maps for all clusters")
    logger.info("=" * 60)
    base_df = prepare_and_run_inference(args.input_csv, args.model, args.output)

    # Stage 2: 3-parameter grid search on dev rows
    logger.info("=" * 60)
    logger.info("STAGE 2: Running grid search on dev set")
    logger.info("=" * 60)
    best_params, metrics_list = run_grid_search(base_df)

    # Stage 3: Evaluate on test using best hyperparameters
    logger.info("=" * 60)
    logger.info("STAGE 3: Evaluating on test set with best parameters")
    logger.info("=" * 60)
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

    logger.info("=" * 60)
    logger.info("Batch processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

