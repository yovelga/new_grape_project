"""
Inference engine for hyperspectral cube processing.

Provides per-pixel and patch-based inference for full image classification.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HyperspectralInferenceEngine:
    """
    Inference engine for hyperspectral cubes.

    Handles per-pixel classification and probability map generation.
    """

    def __init__(self,
                 model_wrapper: Any,
                 scaler: Optional[Any] = None,
                 preprocess_fn: Optional[callable] = None):
        """
        Initialize inference engine.

        Args:
            model_wrapper: Model wrapper (SklearnModelWrapper or similar)
            scaler: Optional scaler for feature normalization
            preprocess_fn: Optional preprocessing function (e.g., SNV)
        """
        self.model = model_wrapper
        self.scaler = scaler
        self.preprocess_fn = preprocess_fn

    def predict_pixel_probabilities(self,
                                    cube: np.ndarray,
                                    batch_size: int = 10000) -> np.ndarray:
        """
        Predict class probabilities for each pixel in hyperspectral cube.

        Args:
            cube: Hyperspectral cube (H, W, C)
            batch_size: Number of pixels to process at once

        Returns:
            Probability map (H, W, num_classes)
        """
        h, w, c = cube.shape

        # Reshape to (num_pixels, num_bands)
        pixels = cube.reshape(-1, c)

        # Apply preprocessing if specified
        if self.preprocess_fn is not None:
            pixels = self.preprocess_fn(pixels)

        # Apply scaling if available
        if self.scaler is not None:
            pixels = self.scaler.transform(pixels)

        # Predict in batches
        num_pixels = pixels.shape[0]
        num_batches = (num_pixels + batch_size - 1) // batch_size

        all_probs = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_pixels)
            batch = pixels[start_idx:end_idx]

            probs = self.model.predict_proba(batch)
            all_probs.append(probs)

        # Concatenate all batches
        prob_map_flat = np.vstack(all_probs)

        # Reshape to (H, W, num_classes)
        num_classes = prob_map_flat.shape[1]
        prob_map = prob_map_flat.reshape(h, w, num_classes)

        return prob_map

    def predict_pixel_labels(self,
                            cube: np.ndarray,
                            batch_size: int = 10000) -> np.ndarray:
        """
        Predict class labels for each pixel.

        Args:
            cube: Hyperspectral cube (H, W, C)
            batch_size: Batch size for processing

        Returns:
            Label map (H, W)
        """
        prob_map = self.predict_pixel_probabilities(cube, batch_size)
        label_map = np.argmax(prob_map, axis=2)
        return label_map

    def get_binary_mask(self,
                       cube: np.ndarray,
                       positive_class: int = 1,
                       threshold: float = 0.5,
                       batch_size: int = 10000) -> np.ndarray:
        """
        Get binary mask for specific class.

        Args:
            cube: Hyperspectral cube (H, W, C)
            positive_class: Class index to extract
            threshold: Probability threshold
            batch_size: Batch size for processing

        Returns:
            Binary mask (H, W) with dtype bool
        """
        prob_map = self.predict_pixel_probabilities(cube, batch_size)

        if prob_map.shape[2] <= positive_class:
            raise ValueError(f"Class {positive_class} not in probability map")

        class_probs = prob_map[:, :, positive_class]
        binary_mask = class_probs >= threshold

        return binary_mask


class GridAnalyzer:
    """
    Grid-based patch analysis for hyperspectral inference.

    Divides image into grid and analyzes each cell for crack detection.
    """

    def __init__(self,
                 cell_size: int = 64,
                 crack_ratio_threshold: float = 0.1):
        """
        Initialize grid analyzer.

        Args:
            cell_size: Size of each grid cell (pixels)
            crack_ratio_threshold: Minimum fraction of crack pixels to flag cell
        """
        self.cell_size = cell_size
        self.crack_ratio_threshold = crack_ratio_threshold

    def analyze_grid(self,
                    binary_mask: np.ndarray,
                    return_stats: bool = True) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Analyze grid cells in binary mask.

        Args:
            binary_mask: Binary detection mask (H, W)
            return_stats: Whether to return detailed statistics

        Returns:
            Tuple of:
                - Grid visualization (H, W, 3) colored by crack percentage
                - List of cell statistics (if return_stats=True)
        """
        h, w = binary_mask.shape
        cell_size = self.cell_size

        # Calculate grid dimensions
        n_rows = (h + cell_size - 1) // cell_size
        n_cols = (w + cell_size - 1) // cell_size

        # Create visualization
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        stats = []

        # Color buckets: (threshold, color_bgr)
        color_buckets = [
            (50, (0, 0, 180)),    # Dark red
            (40, (0, 30, 200)),   # Red
            (30, (0, 120, 255)),  # Orange
            (20, (0, 200, 255)),  # Yellow
        ]

        for i in range(n_rows):
            for j in range(n_cols):
                # Get cell bounds
                r0 = i * cell_size
                c0 = j * cell_size
                r1 = min((i + 1) * cell_size, h)
                c1 = min((j + 1) * cell_size, w)

                # Extract cell
                cell = binary_mask[r0:r1, c0:c1]
                cell_area = cell.size
                crack_pixels = np.sum(cell)
                crack_percent = (crack_pixels / cell_area) * 100 if cell_area > 0 else 0

                # Determine color based on percentage
                color = None
                for threshold, col in color_buckets:
                    if crack_percent >= threshold:
                        color = col
                        break

                # Fill visualization
                if color is not None:
                    viz[r0:r1, c0:c1] = color

                # Store statistics
                if return_stats:
                    stats.append({
                        'row': i,
                        'col': j,
                        'row0': r0,
                        'col0': c0,
                        'row1': r1,
                        'col1': c1,
                        'area': cell_area,
                        'crack_pixels': int(crack_pixels),
                        'percent_cracked': crack_percent,
                        'flagged': crack_percent >= (self.crack_ratio_threshold * 100)
                    })

        return viz, stats if return_stats else None

    def get_flagged_cells(self, binary_mask: np.ndarray) -> List[Dict]:
        """
        Get only cells that exceed crack ratio threshold.

        Args:
            binary_mask: Binary detection mask (H, W)

        Returns:
            List of flagged cell statistics
        """
        _, stats = self.analyze_grid(binary_mask, return_stats=True)
        flagged = [s for s in stats if s['flagged']]
        return flagged
