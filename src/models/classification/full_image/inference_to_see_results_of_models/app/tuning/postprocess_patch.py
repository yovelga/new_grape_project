"""
Postprocess and patch-based classifier for full-image classification.

Implements the complete pipeline:
1. Pixel-level thresholding
2. Morphological operations
3. Blob filtering
4. Patch-based decision logic
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatchClassifierParams:
    """
    Parameters for postprocess + patch-based classifier.
    
    These are the hyperparameters that will be tuned by Optuna.
    
    Attributes:
        pixel_threshold: Probability threshold for binarization [0, 1]
        min_blob_area: Minimum blob area in pixels (blobs smaller removed)
        max_blob_area: Maximum blob area in pixels (blobs larger removed). None = no limit.
        morph_size: Morphological closing kernel size (0=disabled, must be odd)
        patch_size: Square patch size in pixels (e.g., 4, 8, 16, 32, 64, 128)
        patch_crack_pct_threshold: If any patch has >= this % crack pixels, classify as CRACK
        global_crack_pct_threshold: Global crack % threshold - if total image crack % >= this, classify as CRACK
    """
    pixel_threshold: float = 0.90
    min_blob_area: int = 100
    max_blob_area: Optional[int] = None  # None = no limit
    morph_size: int = 5
    patch_size: int = 32
    patch_crack_pct_threshold: float = 10.0  # percentage (0-100)
    global_crack_pct_threshold: float = 1.0  # percentage (0-100)
    
    def validate(self) -> None:
        """Validate parameters."""
        if not 0.0 <= self.pixel_threshold <= 1.0:
            raise ValueError(f"pixel_threshold must be in [0, 1], got {self.pixel_threshold}")
        
        if self.min_blob_area < 0:
            raise ValueError(f"min_blob_area must be >= 0, got {self.min_blob_area}")
        
        if self.max_blob_area is not None:
            if self.max_blob_area < 0:
                raise ValueError(f"max_blob_area must be >= 0 (or None), got {self.max_blob_area}")
            if self.max_blob_area <= self.min_blob_area:
                raise ValueError(f"max_blob_area ({self.max_blob_area}) must be > min_blob_area ({self.min_blob_area})")
        
        if self.morph_size < 0:
            raise ValueError(f"morph_size must be >= 0, got {self.morph_size}")
        
        if self.morph_size > 0 and self.morph_size % 2 == 0:
            raise ValueError(f"morph_size must be odd or 0, got {self.morph_size}")
        
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        
        if not 0.0 <= self.patch_crack_pct_threshold <= 100.0:
            raise ValueError(f"patch_crack_pct_threshold must be in [0, 100], got {self.patch_crack_pct_threshold}")
        
        if not 0.0 <= self.global_crack_pct_threshold <= 100.0:
            raise ValueError(f"global_crack_pct_threshold must be in [0, 100], got {self.global_crack_pct_threshold}")
        
        if self.patch_size < 4:
            raise ValueError(f"patch_size must be >= 4 (smaller values cause extreme slowdown), got {self.patch_size}")


@dataclass
class ClassificationResult:
    """
    Result of patch-based classification for a single image.
    
    Attributes:
        predicted_label: Final image-level prediction (0=HEALTHY, 1=CRACK)
        max_patch_crack_pct: Maximum crack percentage found in any patch
        num_flagged_patches: Number of patches exceeding threshold
        total_patches: Total number of patches analyzed
        global_crack_pct: Global crack percentage of entire image
        num_blobs: Number of blobs after filtering
        max_blob_area: Area of largest blob (pixels)
        total_crack_pixels: Total crack pixels in final mask
        final_mask: Final binary mask after all postprocessing (H, W)
    """
    predicted_label: int
    max_patch_crack_pct: float
    num_flagged_patches: int
    total_patches: int
    global_crack_pct: float
    num_blobs: int
    max_blob_area: int
    total_crack_pixels: int
    final_mask: np.ndarray


class PostprocessPatchClassifier:
    """
    Postprocessing + patch-based classifier.
    
    Applies threshold, morphology, blob filtering, and patch analysis
    to produce image-level classification from probability maps.
    """
    
    def __init__(self, params: PatchClassifierParams):
        """
        Initialize classifier with parameters.
        
        Args:
            params: Classification parameters
        """
        self.params = params
        self.params.validate()
    
    def _apply_threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply probability threshold to get binary mask.
        
        Args:
            prob_map: Probability map (H, W) or (H, W, C) - uses class 1 if multi-class
            
        Returns:
            Binary mask (H, W) as uint8
        """
        # Handle multi-class probability maps (take class 1 = CRACK)
        if prob_map.ndim == 3:
            if prob_map.shape[2] >= 2:
                prob_map = prob_map[:, :, 1]
            else:
                prob_map = prob_map[:, :, 0]
        
        mask = (prob_map >= self.params.pixel_threshold).astype(np.uint8)
        return mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing operation.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Morphologically processed mask
        """
        if self.params.morph_size == 0:
            return mask
        
        import cv2
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.params.morph_size, self.params.morph_size)
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closed
    
    def _filter_blobs(self, mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Filter blobs by area.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Tuple of (filtered_mask, num_blobs, max_blob_area)
        """
        if self.params.min_blob_area == 0 and self.params.max_blob_area is None:
            # No filtering, just count blobs
            import cv2
            num_labels, labels = cv2.connectedComponents(mask)
            num_blobs = num_labels - 1  # Exclude background
            
            if num_blobs > 0:
                # Find max blob area
                max_blob_area = 0
                for label_id in range(1, num_labels):
                    area = np.sum(labels == label_id)
                    max_blob_area = max(max_blob_area, area)
            else:
                max_blob_area = 0
            
            return mask, num_blobs, max_blob_area
        
        # Filter by area (min and/or max)
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create filtered mask
        filtered_mask = np.zeros_like(mask)
        num_blobs = 0
        max_blob_area = 0
        
        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            # Check min area constraint
            if area < self.params.min_blob_area:
                continue
            
            # Check max area constraint (if set)
            if self.params.max_blob_area is not None and area > self.params.max_blob_area:
                continue
            
            filtered_mask[labels == label_id] = 1
            num_blobs += 1
            max_blob_area = max(max_blob_area, area)
        
        return filtered_mask, num_blobs, max_blob_area
    
    def _analyze_patches(self, mask: np.ndarray) -> Tuple[float, int, int]:
        """
        Analyze patches and compute crack percentages.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Tuple of (max_patch_crack_pct, num_flagged_patches, total_patches)
        """
        h, w = mask.shape
        patch_size = self.params.patch_size
        
        # Calculate number of patches (handle partial patches at edges)
        num_patches_h = (h + patch_size - 1) // patch_size
        num_patches_w = (w + patch_size - 1) // patch_size
        total_patches = num_patches_h * num_patches_w
        
        # Early exit if no crack pixels at all
        if not np.any(mask):
            return 0.0, 0, total_patches
        
        max_patch_crack_pct = 0.0
        num_flagged_patches = 0
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Define patch bounds
                y_start = i * patch_size
                y_end = min((i + 1) * patch_size, h)
                x_start = j * patch_size
                x_end = min((j + 1) * patch_size, w)
                
                # Extract patch
                patch = mask[y_start:y_end, x_start:x_end]
                
                # Early skip for empty patches (optimization)
                if not np.any(patch):
                    continue
                
                # Compute crack percentage
                crack_pct = 100.0 * np.mean(patch)
                
                if crack_pct > max_patch_crack_pct:
                    max_patch_crack_pct = crack_pct
                
                if crack_pct >= self.params.patch_crack_pct_threshold:
                    num_flagged_patches += 1
        
        return max_patch_crack_pct, num_flagged_patches, total_patches
    
    def classify(self, prob_map: np.ndarray) -> ClassificationResult:
        """
        Classify full image using postprocess + patch pipeline.
        
        Args:
            prob_map: Probability map (H, W) or (H, W, C)
            
        Returns:
            ClassificationResult with decision and diagnostics
        """
        # Step 1: Threshold
        mask = self._apply_threshold(prob_map)
        
        # Step 2: Morphology
        mask = self._apply_morphology(mask)
        
        # Step 3: Blob filtering
        mask, num_blobs, max_blob_area = self._filter_blobs(mask)
        
        # Step 4: Patch analysis
        max_patch_crack_pct, num_flagged_patches, total_patches = self._analyze_patches(mask)
        
        # Step 5: Calculate global crack percentage
        total_crack_pixels = int(np.sum(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        global_crack_pct = 100.0 * total_crack_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Step 6: Image-level decision (patch-based OR global percentage)
        predicted_label = 1 if (num_flagged_patches > 0 or global_crack_pct >= self.params.global_crack_pct_threshold) else 0
        
        return ClassificationResult(
            predicted_label=predicted_label,
            max_patch_crack_pct=max_patch_crack_pct,
            num_flagged_patches=num_flagged_patches,
            total_patches=total_patches,
            global_crack_pct=global_crack_pct,
            num_blobs=num_blobs,
            max_blob_area=max_blob_area,
            total_crack_pixels=total_crack_pixels,
            final_mask=mask
        )
    
    def update_params(self, params: PatchClassifierParams) -> None:
        """
        Update classifier parameters.
        
        Args:
            params: New parameters
        """
        params.validate()
        self.params = params
