"""
Grid Search for HSI Crack Detection with Blob & Patch Filtering

This script performs hyperparameter optimization using:
1. Pixel Classification (LDA probability maps)
2. Blob Filtering (morphological cleaning of connected components)
3. Patch-Level Analysis (PLA) - grid-based local crack detection
4. Image-Level Aggregation (global threshold decision)

Author: Generated for Grape Project
Date: 2025-11-19
"""

import os
import sys
import logging
import itertools
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from joblib import Parallel, delayed
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GridSearch.BlobPatch")

# ============================================================================
# HYPERPARAMETER GRID CONFIGURATION
# ============================================================================

@dataclass
class GridSearchConfig:
    """Configuration for grid search hyperparameters."""

    # Pixel-level thresholds (high confidence to reduce noise)
    prob_thr_candidates: List[float] = None

    # Blob filtering (morphological cleaning)
    min_blob_size_candidates: List[int] = None

    # Advanced morphology filters
    circularity_min_candidates: List[Optional[float]] = None
    circularity_max_candidates: List[Optional[float]] = None
    aspect_ratio_min_candidates: List[Optional[float]] = None
    aspect_ratio_limit_candidates: List[Optional[float]] = None
    solidity_min_candidates: List[Optional[float]] = None
    solidity_max_candidates: List[Optional[float]] = None
    # Morphological closing kernel sizes to try (0 = disabled)
    morph_size_candidates: List[int] = None

    # Patch-level analysis
    patch_size_candidates: List[int] = None
    patch_pixel_ratio_candidates: List[float] = None

    # Global aggregation threshold
    global_threshold_candidates: List[float] = None

    def __post_init__(self):
        """Set default values if not provided."""
        if self.prob_thr_candidates is None:
            self.prob_thr_candidates = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

        if self.min_blob_size_candidates is None:
            self.min_blob_size_candidates = [0, 50, 100, 300, 500, 1000]

        if self.circularity_min_candidates is None:
            self.circularity_min_candidates = [ 0.0]

        if self.circularity_max_candidates is None:
            self.circularity_max_candidates = [1.0]

        if self.aspect_ratio_limit_candidates is None:
            self.aspect_ratio_limit_candidates = [ 5.0]

        if self.aspect_ratio_min_candidates is None:
            self.aspect_ratio_min_candidates = [None, 1.0, 2.0]

        if self.solidity_min_candidates is None:
            self.solidity_min_candidates = [None, 0.0, 0.5]

        if self.solidity_max_candidates is None:
            self.solidity_max_candidates = [None, 0.9, 0.95]

        if self.morph_size_candidates is None:
            self.morph_size_candidates = [0, 3]

        if self.patch_size_candidates is None:
            self.patch_size_candidates = [16, 32, 64, 128]

        if self.patch_pixel_ratio_candidates is None:
            self.patch_pixel_ratio_candidates = [0.01, 0.05, 0.10, 0.20]

        if self.global_threshold_candidates is None:
            self.global_threshold_candidates = [0.01, 0.05, 0.10]

    def total_combinations(self) -> int:
        """Calculate total number of hyperparameter combinations."""
        return (len(self.prob_thr_candidates) *
                len(self.min_blob_size_candidates) *
                len(self.circularity_min_candidates) *
                len(self.circularity_max_candidates) *
                len(self.aspect_ratio_min_candidates) *
                len(self.aspect_ratio_limit_candidates) *
                len(self.solidity_min_candidates) *
                len(self.solidity_max_candidates) *
                len(self.morph_size_candidates) *
                len(self.patch_size_candidates) *
                len(self.patch_pixel_ratio_candidates) *
                len(self.global_threshold_candidates))


# ============================================================================
# BLOB FILTERING FUNCTIONS
# ============================================================================

def calculate_blob_metrics(contour: np.ndarray) -> Dict[str, float]:
    """
    Calculate morphological metrics for a single blob.

    Args:
        contour: OpenCV contour

    Returns:
        Dictionary with metrics: area, circularity, aspect_ratio, solidity
    """
    metrics = {
        'area': 0.0,
        'circularity': 0.0,
        'aspect_ratio': 0.0,
        'solidity': 0.0
    }

    # Area
    area = cv2.contourArea(contour)
    metrics['area'] = area

    if area < 1:
        return metrics

    # Circularity: (4 * pi * Area) / (Perimeter^2)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        metrics['circularity'] = (4 * np.pi * area) / (perimeter * perimeter)

    # Aspect Ratio: width / height of bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    if h > 0:
        metrics['aspect_ratio'] = w / h

    # Solidity: ContourArea / ConvexHullArea
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        metrics['solidity'] = area / hull_area

    return metrics


def filter_blobs_advanced(binary_mask: np.ndarray,
                          min_blob_size: int,
                          circularity_min: Optional[float] = None,
                          circularity_max: Optional[float] = None,
                          aspect_ratio_min: Optional[float] = None,
                          aspect_ratio_limit: Optional[float] = None,
                          solidity_min: Optional[float] = None,
                          solidity_max: Optional[float] = None) -> np.ndarray:
    """
    Remove blobs based on size and advanced morphological criteria.

    Args:
        binary_mask: Binary mask (0 or 255)
        min_blob_size: Minimum blob size in pixels
        circularity_min: Minimum circularity (4*pi*Area/Perimeter^2), None = no filter
        circularity_max: Maximum circularity, None = no filter
        aspect_ratio_min: Minimum aspect ratio (width/height), None = no filter
        aspect_ratio_limit: Max ratio of width/height (or height/width), None = no filter
        solidity_min: Minimum solidity (Area/ConvexHullArea), None = no filter
        solidity_max: Maximum solidity (Area/ConvexHullArea), None = no filter

    Returns:
        Cleaned binary mask with filtered blobs
    """
    if min_blob_size <= 0 and circularity_min is None and circularity_max is None and \
       aspect_ratio_min is None and aspect_ratio_limit is None and solidity_min is None and solidity_max is None:
        return binary_mask.copy()

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    # Create output mask
    filtered_mask = np.zeros_like(binary_mask)

    # Process each blob (skip label 0 = background)
    for label_id in range(1, num_labels):
        # Basic area check
        blob_area = stats[label_id, cv2.CC_STAT_AREA]
        if blob_area < min_blob_size:
            continue

        # Get blob mask and find contour for shape analysis
        blob_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        contour = contours[0]

        # Calculate morphological metrics
        metrics = calculate_blob_metrics(contour)

        # Apply circularity filter
        if circularity_min is not None and metrics['circularity'] < circularity_min:
            continue
        if circularity_max is not None and metrics['circularity'] > circularity_max:
            continue

        # Apply aspect ratio filter (check both orientations)
        if aspect_ratio_limit is not None:
            aspect = metrics['aspect_ratio']
            # Reject if either width/height OR height/width exceeds limit
            if aspect > aspect_ratio_limit and (1.0 / aspect) > aspect_ratio_limit:
                continue

        # Apply solidity filter
        if solidity_min is not None and metrics['solidity'] < solidity_min:
            continue
        if solidity_max is not None and metrics['solidity'] > solidity_max:
            continue

        # Keep this blob
        filtered_mask[labels == label_id] = 255

    return filtered_mask


# ============================================================================
# PATCH-LEVEL ANALYSIS (PLA) FUNCTIONS
# ============================================================================

def compute_patch_labels(binary_mask: np.ndarray,
                         patch_size: int,
                         patch_pixel_ratio: float) -> Tuple[np.ndarray, int, int]:
    """
    Divide mask into patches and label each patch as cracked or not.

    Args:
        binary_mask: Binary mask (0 or 255)
        patch_size: Size of each square patch
        patch_pixel_ratio: Threshold for % of positive pixels in patch

    Returns:
        Tuple of (patch_labels, num_cracked_patches, total_patches)
        - patch_labels: 2D array of patch labels (1=cracked, 0=healthy)
        - num_cracked_patches: Count of cracked patches
        - total_patches: Total number of patches
    """
    H, W = binary_mask.shape

    # Calculate grid dimensions
    num_rows = (H + patch_size - 1) // patch_size
    num_cols = (W + patch_size - 1) // patch_size

    patch_labels = np.zeros((num_rows, num_cols), dtype=np.uint8)
    num_cracked_patches = 0

    for i in range(num_rows):
        for j in range(num_cols):
            # Extract patch
            r_start = i * patch_size
            r_end = min(r_start + patch_size, H)
            c_start = j * patch_size
            c_end = min(c_start + patch_size, W)

            patch = binary_mask[r_start:r_end, c_start:c_end]

            # Calculate ratio of positive pixels
            patch_pixels = patch.size
            if patch_pixels == 0:
                continue

            positive_pixels = np.sum(patch > 0)
            ratio = positive_pixels / patch_pixels

            # Label patch
            if ratio > patch_pixel_ratio:
                patch_labels[i, j] = 1
                num_cracked_patches += 1

    total_patches = num_rows * num_cols
    return patch_labels, num_cracked_patches, total_patches


def classify_image_by_patch_ratio(binary_mask: np.ndarray,
                                   patch_size: int,
                                   patch_pixel_ratio: float,
                                   global_threshold: float) -> Tuple[int, float]:
    """
    Classify entire image based on patch-level analysis.

    Args:
        binary_mask: Binary mask after blob filtering
        patch_size: Size of patches for grid
        patch_pixel_ratio: Threshold for patch labeling
        global_threshold: Minimum ratio of cracked patches for image to be cracked

    Returns:
        Tuple of (prediction, global_ratio)
        - prediction: 1 if cracked, 0 if healthy
        - global_ratio: Ratio of cracked patches to total patches
    """
    patch_labels, num_cracked, total = compute_patch_labels(
        binary_mask, patch_size, patch_pixel_ratio
    )

    if total == 0:
        return 0, 0.0

    global_ratio = num_cracked / total
    prediction = 1 if global_ratio > global_threshold else 0

    return prediction, global_ratio


# ============================================================================
# SINGLE IMAGE PROCESSING
# ============================================================================

def process_single_image(prob_map: np.ndarray,
                        prob_thr: float,
                        min_blob_size: int,
                        circularity_min: Optional[float],
                        circularity_max: Optional[float],
                        aspect_ratio_min: Optional[float],
                        aspect_ratio_limit: Optional[float],
                        solidity_min: Optional[float],
                        solidity_max: Optional[float],
                        patch_size: int,
                        patch_pixel_ratio: float,
                        global_threshold: float,
                        morph_size: int = 0) -> Tuple[int, float, np.ndarray]:
    """
    Process a single image through the full pipeline.

    Args:
        prob_map: Probability map from LDA model
        prob_thr: Probability threshold for binarization
        min_blob_size: Minimum blob size for filtering
        circularity_min: Minimum circularity filter
        circularity_max: Maximum circularity filter
        aspect_ratio_min: Minimum aspect ratio filter
        aspect_ratio_limit: Maximum aspect ratio filter
        solidity_min: Minimum solidity filter
        solidity_max: Maximum solidity filter
        patch_size: Patch size for PLA
        patch_pixel_ratio: Patch pixel ratio threshold
        global_threshold: Global threshold for image classification

    Returns:
        Tuple of (prediction, global_ratio, filtered_mask)
    """
    # Step 1: Thresholding
    binary_mask = ((prob_map >= prob_thr) * 255).astype(np.uint8)

    # Step 1.5: Morphological closing (optional) - apply before blob analysis
    if morph_size and int(morph_size) > 0:
        try:
            k = int(morph_size)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        except Exception:
            logger.exception("Morphological closing in process_single_image failed; continuing without it")

    # Step 2: Advanced Blob Filtering (filters by size and geometry)
    filtered_mask = filter_blobs_advanced(
        binary_mask,
        min_blob_size,
        circularity_min=circularity_min,
        circularity_max=circularity_max,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_limit=aspect_ratio_limit,
        solidity_min=solidity_min,
        solidity_max=solidity_max
    )

    # Step 3 & 4: Patch-Level Analysis and Image Classification
    prediction, global_ratio = classify_image_by_patch_ratio(
        filtered_mask, patch_size, patch_pixel_ratio, global_threshold
    )

    return prediction, global_ratio, filtered_mask


# ============================================================================
# GRID SEARCH IMPLEMENTATION
# ============================================================================

def evaluate_single_combination(prob_maps_dict: Dict[str, np.ndarray],
                                labels_dict: Dict[str, int],
                                prob_thr: float,
                                min_blob_size: int,
                                circularity_min: Optional[float],
                                circularity_max: Optional[float],
                                aspect_ratio_min: Optional[float],
                                aspect_ratio_limit: Optional[float],
                                solidity_min: Optional[float],
                                solidity_max: Optional[float],
                                patch_size: int,
                                patch_pixel_ratio: float,
                                global_threshold: float,
                                morph_size: int = 0) -> Dict:
    """
    Evaluate a single hyperparameter combination on all dev samples.

    Args:
        prob_maps_dict: Dictionary mapping sample_id -> probability map
        labels_dict: Dictionary mapping sample_id -> ground truth label
        prob_thr, min_blob_size, circularity_min, circularity_max,
        aspect_ratio_limit, solidity_max, patch_size, patch_pixel_ratio,
        global_threshold: Hyperparameters to evaluate

    Returns:
        Dictionary with hyperparameters and metrics
    """
    y_true = []
    y_pred = []
    y_scores = []

    for sample_id in prob_maps_dict.keys():
        prob_map = prob_maps_dict[sample_id]
        true_label = labels_dict[sample_id]

        # Process image
        pred, global_ratio, _ = process_single_image(
            prob_map, prob_thr, min_blob_size, circularity_min, circularity_max,
            aspect_ratio_min, aspect_ratio_limit, solidity_min, solidity_max,
            patch_size, patch_pixel_ratio, global_threshold,
            morph_size=morph_size
        )

        y_true.append(true_label)
        y_pred.append(pred)
        y_scores.append(global_ratio)

    # Calculate metrics
    if len(y_true) == 0:
        return {
            'prob_thr': prob_thr,
            'min_blob_size': min_blob_size,
            'circularity_min': circularity_min,
            'circularity_max': circularity_max,
            'aspect_ratio_min': aspect_ratio_min,
            'aspect_ratio_limit': aspect_ratio_limit,
            'solidity_min': solidity_min,
            'solidity_max': solidity_max,
            'patch_size': patch_size,
            'patch_pixel_ratio': patch_pixel_ratio,
            'global_threshold': global_threshold,
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1_score': float('nan'),
            'roc_auc': float('nan'),
            'n_samples': 0
        }

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = float('nan')

    return {
        'prob_thr': prob_thr,
        'min_blob_size': min_blob_size,
        'circularity_min': circularity_min,
        'circularity_max': circularity_max,
        'aspect_ratio_min': aspect_ratio_min,
        'aspect_ratio_limit': aspect_ratio_limit,
        'solidity_min': solidity_min,
        'solidity_max': solidity_max,
        'morph_size': morph_size,
        'patch_size': patch_size,
        'patch_pixel_ratio': patch_pixel_ratio,
        'global_threshold': global_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'n_samples': len(y_true)
    }


def run_grid_search(prob_maps_dict: Dict[str, np.ndarray],
                   labels_dict: Dict[str, int],
                   config: GridSearchConfig,
                   n_jobs: int = -1,
                   progress_callback: Optional[callable] = None) -> pd.DataFrame:
    """
    Run full grid search over all hyperparameter combinations.

    Args:
        prob_maps_dict: Dictionary mapping sample_id -> probability map
        labels_dict: Dictionary mapping sample_id -> ground truth label
        config: Grid search configuration
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        progress_callback: Optional callback function(current, total, message) for progress updates

    Returns:
        DataFrame with all results sorted by F1 score
    """
    logger.info("="*80)
    logger.info("STARTING GRID SEARCH ON VALIDATION SET")
    logger.info("="*80)
    logger.info(f"📊 Evaluation dataset: VALIDATION SET (row=1)")
    logger.info(f"🎯 Purpose: Hyperparameter tuning (NOT model training)")
    logger.info(f"ℹ️  Note: Model was already trained. Grid search tunes post-processing parameters.")
    logger.info(f"Total combinations to test: {config.total_combinations():,}")
    logger.info(f"Validation set samples: {len(prob_maps_dict)}")
    logger.info(f"Label distribution: {dict(pd.Series(list(labels_dict.values())).value_counts())}")
    logger.info("")
    logger.info("⚠️  Metrics below are on VALIDATION set. TEST set (row=2) evaluated after.")

    # Log parameter space
    logger.info("")
    logger.info("Parameter Space:")
    logger.info(f"  • prob_thr: {len(config.prob_thr_candidates)} values")
    logger.info(f"  • min_blob_size: {len(config.min_blob_size_candidates)} values")
    logger.info(f"  • circularity_min: {len(config.circularity_min_candidates)} values")
    logger.info(f"  • circularity_max: {len(config.circularity_max_candidates)} values")
    logger.info(f"  • aspect_ratio_limit: {len(config.aspect_ratio_limit_candidates)} values")
    logger.info(f"  • solidity_max: {len(config.solidity_max_candidates)} values")
    logger.info(f"  • morph_size: {len(config.morph_size_candidates)} values")
    logger.info(f"  • patch_size: {len(config.patch_size_candidates)} values")
    logger.info(f"  • patch_pixel_ratio: {len(config.patch_pixel_ratio_candidates)} values")
    logger.info(f"  • global_threshold: {len(config.global_threshold_candidates)} values")

    # Generate all combinations
    logger.info("")
    logger.info("Generating parameter combinations...")
    combinations = list(itertools.product(
        config.prob_thr_candidates,
        config.min_blob_size_candidates,
        config.circularity_min_candidates,
        config.circularity_max_candidates,
        config.aspect_ratio_min_candidates,
        config.aspect_ratio_limit_candidates,
        config.solidity_min_candidates,
        config.solidity_max_candidates,
        config.morph_size_candidates,
        config.patch_size_candidates,
        config.patch_pixel_ratio_candidates,
        config.global_threshold_candidates
    ))

    logger.info(f"✓ Generated {len(combinations):,} combinations")

    # Progress tracking
    start_time = time.time()

    # Use parallel processing if n_jobs != 1
    if n_jobs != 1:
        import multiprocessing
        actual_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        logger.info("")
        logger.info(f"⚡ Running in PARALLEL mode with {actual_jobs} CPU cores")
        logger.info(f"   Backend: joblib.Parallel")
        if progress_callback:
            logger.info("   Progress: Callback enabled (updates every ~1000 tasks)")
        else:
            logger.info("   Verbose level: 10 (progress updates enabled)")
        logger.info("")

        # Process in batches if callback is provided for better progress updates
        if progress_callback:
            batch_size = 100  # Process 100 combinations at a time for more frequent updates
            results = []
            total_processed = 0

            logger.info(f"Processing in batches of {batch_size} for progress tracking...")
            logger.info(f"Updates every ~{batch_size} combinations")

            for batch_start in range(0, len(combinations), batch_size):
                batch_end = min(batch_start + batch_size, len(combinations))
                batch_combinations = combinations[batch_start:batch_end]

                # Process this batch
                batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(evaluate_single_combination)(
                        prob_maps_dict, labels_dict,
                        prob_thr, min_blob, circ_min, circ_max, aspect_min, aspect_lim, solid_min, solid_max,
                        patch_sz, patch_ratio, global_thr, morph_sz
                    )
                    for prob_thr, min_blob, circ_min, circ_max, aspect_min, aspect_lim, solid_min, solid_max, morph_sz,
                        patch_sz, patch_ratio, global_thr in batch_combinations
                )

                results.extend(batch_results)
                total_processed += len(batch_results)

                # Update progress with percentage
                progress_pct = 100.0 * total_processed / len(combinations)
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                eta_sec = (len(combinations) - total_processed) / speed if speed > 0 else 0
                eta_min = eta_sec / 60

                message = (f"Grid Search: {total_processed:,}/{len(combinations):,} ({progress_pct:.1f}%) | "
                          f"Speed: {speed:.1f} comb/s | ETA: {eta_min:.1f}m ({eta_sec:.0f}s)")

                logger.info(message)
                progress_callback(total_processed, len(combinations), message)
        else:
            # Standard parallel execution without callback
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(evaluate_single_combination)(
                    prob_maps_dict, labels_dict,
                    prob_thr, min_blob, circ_min, circ_max, aspect_min, aspect_lim, solid_min, solid_max,
                    patch_sz, patch_ratio, global_thr, morph_sz
                )
                for prob_thr, min_blob, circ_min, circ_max, aspect_min, aspect_lim, solid_min, solid_max, morph_sz,
                    patch_sz, patch_ratio, global_thr in combinations
            )
    else:
        # Sequential processing with progress
        logger.info("")
        logger.info("Running in SEQUENTIAL mode (n_jobs=1)")
        logger.info("Progress updates every 100 combinations...")
        logger.info("")
        results = []
        last_logged_pct = -1  # Track last logged percentage
        for idx, (prob_thr, min_blob, circ_min, circ_max, aspect_min, aspect_lim, solid_min, solid_max,
                  morph_sz, patch_sz, patch_ratio, global_thr) in enumerate(combinations):
            if idx % 100 == 0:
                if idx > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(combinations) - idx)
                    completion_pct = 100 * idx / len(combinations)
                    logger.info(f"Progress: {idx:,}/{len(combinations):,} ({completion_pct:.1f}%) | "
                               f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                               f"Speed: {idx/elapsed:.1f} comb/s")

                    # Log every 5% milestone
                    current_pct_milestone = int(completion_pct / 5) * 5
                    if current_pct_milestone > last_logged_pct and current_pct_milestone % 5 == 0:
                        logger.info(f"📊 {current_pct_milestone}% complete")
                        last_logged_pct = current_pct_milestone

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(idx, len(combinations),
                                        f"Progress: {completion_pct:.1f}%")
                else:
                    logger.info("Starting evaluation...")

            result = evaluate_single_combination(
                prob_maps_dict, labels_dict,
                prob_thr, min_blob, circ_min, circ_max, aspect_min, aspect_lim, solid_min, solid_max,
                patch_sz, patch_ratio, global_thr, morph_size=morph_sz
            )
            results.append(result)

        # Final progress update
        elapsed = time.time() - start_time
        logger.info(f"Progress: {len(combinations):,}/{len(combinations):,} (100.0%) | "
                   f"Total time: {elapsed:.1f}s")

    # Create DataFrame
    logger.info("")
    logger.info("Creating results DataFrame...")
    df_results = pd.DataFrame(results)
    logger.info(f"✓ Created DataFrame with {len(df_results):,} rows")

    # Sort by F1 score (descending)
    logger.info("Sorting results by F1 score...")
    df_results = df_results.sort_values('f1_score', ascending=False, na_position='last')

    elapsed = time.time() - start_time

    # Log summary statistics
    logger.info("")
    logger.info("="*80)
    logger.info("GRID SEARCH COMPLETED")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"📊 Combinations tested: {len(df_results):,}")
    logger.info(f"⚡ Speed: {len(df_results)/elapsed:.2f} combinations/second")

    # Statistics on results
    valid_results = df_results['f1_score'].notna().sum()
    logger.info(f"✓ Valid results: {valid_results:,}/{len(df_results):,} ({100*valid_results/len(df_results):.1f}%)")

    if valid_results > 0:
        logger.info("")
        logger.info("📈 F1 Score Distribution:")
        logger.info(f"   • Best:    {df_results['f1_score'].max():.4f}")
        logger.info(f"   • Median:  {df_results['f1_score'].median():.4f}")
        logger.info(f"   • Mean:    {df_results['f1_score'].mean():.4f}")
        logger.info(f"   • Worst:   {df_results['f1_score'].min():.4f}")
        logger.info(f"   • Std:     {df_results['f1_score'].std():.4f}")

        # Log best configuration
        best = df_results.iloc[0]
        logger.info("")
        logger.info("🏆 BEST CONFIGURATION (on VALIDATION set):")
        logger.info(f"   • prob_thr:          {best['prob_thr']:.4f}")
        logger.info(f"   • min_blob_size:     {best['min_blob_size']}")
        logger.info(f"   • circularity_min:   {best.get('circularity_min', 'N/A')}")
        logger.info(f"   • circularity_max:   {best.get('circularity_max', 'N/A')}")
        logger.info(f"   • aspect_ratio_lim:  {best.get('aspect_ratio_limit', 'N/A')}")
        logger.info(f"   • solidity_max:      {best.get('solidity_max', 'N/A')}")
        logger.info(f"   • morph_size:       {best['morph_size']}")
        logger.info(f"   • patch_size:        {best['patch_size']}")
        logger.info(f"   • patch_pixel_ratio: {best['patch_pixel_ratio']:.4f}")
        logger.info(f"   • global_threshold:  {best['global_threshold']:.4f}")
        logger.info("")
        logger.info("   📊 VALIDATION SET Metrics (for hyperparameter selection):")
        logger.info(f"   • F1 Score:  {best['f1_score']:.4f}")
        logger.info(f"   • Accuracy:  {best['accuracy']:.4f}")
        logger.info(f"   • Precision: {best['precision']:.4f}")
        logger.info(f"   • Recall:    {best['recall']:.4f}")
        logger.info(f"   • ROC AUC:   {best['roc_auc']:.4f}")

        # Show top 5
        logger.info("")
        logger.info("📋 Top 5 Configurations (VALIDATION SET metrics):")
        for i in range(min(5, len(df_results))):
            row = df_results.iloc[i]
            logger.info(f"   #{i+1}: F1={row['f1_score']:.4f}, "
                       f"Acc={row['accuracy']:.4f}, "
                       f"Prec={row['precision']:.4f}, "
                       f"Rec={row['recall']:.4f}")

    logger.info("")
    logger.info("="*80)
    logger.info(f"Grid search completed in {elapsed:.1f}s")

    return df_results


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def load_probability_maps_from_results(results_df: pd.DataFrame,
                                      row_filter: int = 1) -> Tuple[Dict, Dict]:
    """
    Load probability maps and labels from results DataFrame.

    Args:
        results_df: DataFrame with columns: grape_id, row, label, prob_map_path, status
        row_filter: Filter by row (1=dev, 2=test)

    Returns:
        Tuple of (prob_maps_dict, labels_dict)
    """
    # Filter to dev set and valid rows
    # Build filter conditions dynamically based on available columns
    filter_conditions = (results_df['row'] == row_filter)

    # Only filter by status if the column exists
    if 'status' in results_df.columns:
        filter_conditions = filter_conditions & (results_df['status'] == 'ok')

    filter_conditions = filter_conditions & results_df['prob_map_path'].notna() & (results_df['prob_map_path'] != '')

    dev_df = results_df[filter_conditions].copy()

    logger.info(f"Loading probability maps for {len(dev_df)} samples (row={row_filter})...")

    prob_maps_dict = {}
    labels_dict = {}

    for idx, row in dev_df.iterrows():
        sample_id = str(row['grape_id'])
        prob_path = row['prob_map_path']
        label = int(row['label'])

        try:
            if os.path.exists(prob_path):
                prob_map = np.load(prob_path)
                prob_maps_dict[sample_id] = prob_map
                labels_dict[sample_id] = label
            else:
                logger.warning(f"Probability map not found: {prob_path}")
        except Exception as e:
            logger.warning(f"Failed to load {prob_path}: {e}")

    logger.info(f"Successfully loaded {len(prob_maps_dict)} probability maps")
    return prob_maps_dict, labels_dict


def run_grid_search_pipeline(results_csv_path: str,
                             output_csv_path: str = "grid_search_results.csv",
                             config: Optional[GridSearchConfig] = None) -> pd.DataFrame:
    """
    Run complete grid search pipeline from results CSV.

    Args:
        results_csv_path: Path to CSV with probability map paths
        output_csv_path: Path to save grid search results
        config: Grid search configuration (uses defaults if None)

    Returns:
        DataFrame with grid search results
    """
    logger.info("="*80)
    logger.info("GRID SEARCH WITH BLOB & PATCH FILTERING")
    logger.info("="*80)

    # Load results
    logger.info(f"Loading results from: {results_csv_path}")
    results_df = pd.read_csv(results_csv_path)

    # Load probability maps and labels for dev set
    prob_maps_dict, labels_dict = load_probability_maps_from_results(results_df, row_filter=1)

    if len(prob_maps_dict) == 0:
        logger.error("No probability maps loaded! Cannot run grid search.")
        return pd.DataFrame()

    # Create config if not provided
    if config is None:
        config = GridSearchConfig()

    logger.info("Grid Search Configuration:")
    logger.info(f"  prob_thr: {config.prob_thr_candidates}")
    logger.info(f"  min_blob_size: {config.min_blob_size_candidates}")
    logger.info(f"  patch_size: {config.patch_size_candidates}")
    logger.info(f"  patch_pixel_ratio: {config.patch_pixel_ratio_candidates}")
    logger.info(f"  global_threshold: {config.global_threshold_candidates}")
    logger.info(f"  Total combinations: {config.total_combinations()}")

    # Run grid search
    df_results = run_grid_search(prob_maps_dict, labels_dict, config)

    # Save results
    df_results.to_csv(output_csv_path, index=False)
    logger.info(f"Saved grid search results to: {output_csv_path}")

    # Print best result
    if len(df_results) > 0:
        best = df_results.iloc[0]
        logger.info("="*80)
        logger.info("BEST HYPERPARAMETERS (by F1 Score):")
        logger.info("="*80)
        logger.info(f"  prob_thr:          {best['prob_thr']:.3f}")
        logger.info(f"  min_blob_size:     {best['min_blob_size']:.0f}")
        logger.info(f"  patch_size:        {best['patch_size']:.0f}")
        logger.info(f"  patch_pixel_ratio: {best['patch_pixel_ratio']:.3f}")
        logger.info(f"  global_threshold:  {best['global_threshold']:.3f}")
        logger.info("")
        logger.info("METRICS:")
        logger.info(f"  Accuracy:  {best['accuracy']:.4f}")
        logger.info(f"  Precision: {best['precision']:.4f}")
        logger.info(f"  Recall:    {best['recall']:.4f}")
        logger.info(f"  F1 Score:  {best['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {best['roc_auc']:.4f}")
        logger.info("="*80)

    return df_results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid Search for HSI Crack Detection with Blob & Patch Filtering"
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to CSV with probability map paths (output from prepare_and_run_inference)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='grid_search_results.csv',
        help='Path to save grid search results (default: grid_search_results.csv)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1, use -1 for all CPUs)'
    )

    args = parser.parse_args()

    # Run grid search
    results = run_grid_search_pipeline(
        results_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        config=GridSearchConfig()
    )

    print(f"\n✓ Grid search completed!")
    print(f"✓ Results saved to: {args.output_csv}")
    print(f"✓ Total combinations tested: {len(results)}")
