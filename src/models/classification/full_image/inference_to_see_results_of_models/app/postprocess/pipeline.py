"""
Postprocessing pipeline for full-image inference results.

Provides clean, deterministic post-processing of probability maps:
- Thresholding
- Morphological operations (closing)
- Blob filtering (area, shape, border exclusion)
- Statistics computation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .morphology import morphological_close
from .blob_filters import (
    extract_all_blob_features,
    filter_blobs,
)


@dataclass
class PostprocessConfig:
    """
    Configuration for post-processing pipeline.

    Attributes:
        prob_threshold: Probability threshold for binarization [0, 1]
        morph_close_size: Kernel size for morphological closing.
            Must be odd positive integer. 0 disables morphological closing.
        min_blob_area: Minimum blob area in pixels. Blobs smaller than this are removed.
        max_blob_area: Maximum blob area in pixels. Blobs larger than this are removed.
            None disables this filter. Must be > min_blob_area if set.
        exclude_border: Whether to exclude blobs touching the image border.
        border_margin_px: Extra margin from border (pixels). Only used if exclude_border=True.
        circularity_min: Minimum circularity [0, 1]. None disables filter.
            1.0 = perfect circle, lower values are more elongated.
        solidity_min: Minimum solidity [0, 1]. None disables filter.
            1.0 = convex shape, lower values have concavities.
        aspect_ratio_range: (min, max) aspect ratio (width/height). None disables filter.

    Example:
        >>> config = PostprocessConfig(
        ...     prob_threshold=0.5,
        ...     morph_close_size=5,
        ...     min_blob_area=100,
        ...     max_blob_area=5000,
        ...     exclude_border=True
        ... )
    """
    prob_threshold: float = 0.5
    morph_close_size: int = 0  # 0 disables
    min_blob_area: int = 0
    max_blob_area: Optional[int] = None  # None = no limit
    exclude_border: bool = False
    border_margin_px: int = 0
    circularity_min: Optional[float] = None
    solidity_min: Optional[float] = None
    aspect_ratio_range: Optional[Tuple[float, float]] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.prob_threshold <= 1.0:
            raise ValueError(
                f"prob_threshold must be in [0, 1], got {self.prob_threshold}"
            )

        if self.morph_close_size < 0:
            raise ValueError(
                f"morph_close_size must be >= 0, got {self.morph_close_size}"
            )

        if self.morph_close_size > 0 and self.morph_close_size % 2 == 0:
            raise ValueError(
                f"morph_close_size must be odd (or 0 to disable), got {self.morph_close_size}"
            )

        if self.min_blob_area < 0:
            raise ValueError(
                f"min_blob_area must be >= 0, got {self.min_blob_area}"
            )

        if self.max_blob_area is not None:
            if self.max_blob_area < 0:
                raise ValueError(
                    f"max_blob_area must be >= 0 (or None), got {self.max_blob_area}"
                )
            if self.max_blob_area <= self.min_blob_area:
                raise ValueError(
                    f"max_blob_area ({self.max_blob_area}) must be > min_blob_area ({self.min_blob_area})"
                )

        if self.border_margin_px < 0:
            raise ValueError(
                f"border_margin_px must be >= 0, got {self.border_margin_px}"
            )

        if self.circularity_min is not None:
            if not 0.0 <= self.circularity_min <= 1.0:
                raise ValueError(
                    f"circularity_min must be in [0, 1], got {self.circularity_min}"
                )

        if self.solidity_min is not None:
            if not 0.0 <= self.solidity_min <= 1.0:
                raise ValueError(
                    f"solidity_min must be in [0, 1], got {self.solidity_min}"
                )

        if self.aspect_ratio_range is not None:
            ar_min, ar_max = self.aspect_ratio_range
            if ar_min < 0 or ar_max < ar_min:
                raise ValueError(
                    f"aspect_ratio_range must be (min, max) with 0 <= min <= max, "
                    f"got {self.aspect_ratio_range}"
                )


class PostprocessPipeline:
    """
    Pipeline for post-processing probability maps into binary masks.

    Applies the following steps in order:
    1. Threshold probability map to binary mask
    2. Apply morphological closing (if enabled)
    3. Extract connected components
    4. Filter blobs by area, shape, and position
    5. Compute statistics

    The pipeline is deterministic - same inputs always produce same outputs.

    Example:
        >>> config = PostprocessConfig(prob_threshold=0.5, min_blob_area=100)
        >>> pipeline = PostprocessPipeline(config)
        >>> mask, stats = pipeline.run(prob_map)
    """

    def __init__(self, config: PostprocessConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: PostprocessConfig with pipeline parameters
        """
        config.validate()
        self.config = config

    def run(self, prob_map: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run the post-processing pipeline.

        Args:
            prob_map: Probability map (H, W) with values in [0, 1]

        Returns:
            Tuple of:
                - mask: Binary mask (H, W) with dtype bool
                - stats: Dictionary with pipeline statistics:
                    - num_blobs_before: Blobs before filtering
                    - num_blobs_after: Blobs after filtering
                    - total_positive_pixels: Sum of True pixels in final mask
                    - crack_ratio: positive_pixels / total_pixels
                    - accepted_blobs: List of BlobFeatures for kept blobs
                    - rejected_blobs: List of BlobFeatures for removed blobs

        Raises:
            ValueError: If prob_map is not 2D
        """
        if prob_map.ndim != 2:
            raise ValueError(f"prob_map must be 2D, got {prob_map.ndim}D")

        H, W = prob_map.shape
        total_pixels = H * W

        # Step 1: Threshold
        mask = prob_map >= self.config.prob_threshold

        # Step 2: Morphological closing (if enabled)
        if self.config.morph_close_size > 0:
            mask = morphological_close(mask, self.config.morph_close_size)

        # Step 3 & 4: Extract blobs and filter
        labeled, features = extract_all_blob_features(mask)
        num_blobs_before = len(features)

        # Parse aspect ratio
        ar_min, ar_max = None, None
        if self.config.aspect_ratio_range is not None:
            ar_min, ar_max = self.config.aspect_ratio_range

        filtered_mask, accepted, rejected = filter_blobs(
            labeled=labeled,
            features=features,
            min_area=self.config.min_blob_area,
            max_area=self.config.max_blob_area,
            circularity_min=self.config.circularity_min,
            solidity_min=self.config.solidity_min,
            aspect_ratio_min=ar_min,
            aspect_ratio_max=ar_max,
            exclude_border=self.config.exclude_border,
            border_margin_px=self.config.border_margin_px
        )

        num_blobs_after = len(accepted)

        # Step 5: Compute statistics
        total_positive_pixels = int(np.sum(filtered_mask))
        crack_ratio = total_positive_pixels / total_pixels if total_pixels > 0 else 0.0

        stats = {
            "num_blobs_before": num_blobs_before,
            "num_blobs_after": num_blobs_after,
            "total_positive_pixels": total_positive_pixels,
            "crack_ratio": crack_ratio,
            "accepted_blobs": accepted,
            "rejected_blobs": rejected,
        }

        return filtered_mask, stats

    def run_debug(self, prob_map: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Run the post-processing pipeline with debug outputs.

        This method captures intermediate masks and labeled components for visualization
        and debugging. It produces the same final result as run() but returns additional
        debug information.

        Args:
            prob_map: Probability map (H, W) with values in [0, 1]

        Returns:
            Tuple of:
                - final_mask: Binary mask (H, W) with dtype bool (same as run())
                - stats: Dictionary with pipeline statistics (same as run())
                - debug: Dictionary with intermediate debug outputs:
                    - mask_threshold: Binary mask after thresholding (bool)
                    - mask_after_morph: Binary mask after morphological ops (bool)
                    - labeled_components: Labeled component map (int32), or None if no blobs
                    - accepted_blobs: List of BlobFeatures for kept blobs
                    - rejected_blobs: List of BlobFeatures for removed blobs

        Raises:
            ValueError: If prob_map is not 2D

        Example:
            >>> config = PostprocessConfig(prob_threshold=0.5, morph_close_size=5)
            >>> pipeline = PostprocessPipeline(config)
            >>> mask, stats, debug = pipeline.run_debug(prob_map)
            >>> # Visualize intermediate steps
            >>> viewer.set_image(debug['mask_threshold'])
            >>> viewer.set_overlay(debug['mask_after_morph'])
        """
        if prob_map.ndim != 2:
            raise ValueError(f"prob_map must be 2D, got {prob_map.ndim}D")

        H, W = prob_map.shape
        total_pixels = H * W

        # Step 1: Threshold
        mask_threshold = prob_map >= self.config.prob_threshold

        # Step 2: Morphological closing (if enabled)
        if self.config.morph_close_size > 0:
            mask_after_morph = morphological_close(mask_threshold, self.config.morph_close_size)
        else:
            mask_after_morph = mask_threshold.copy()

        # Step 3 & 4: Extract blobs and filter
        labeled, features = extract_all_blob_features(mask_after_morph)
        num_blobs_before = len(features)

        # Parse aspect ratio
        ar_min, ar_max = None, None
        if self.config.aspect_ratio_range is not None:
            ar_min, ar_max = self.config.aspect_ratio_range

        filtered_mask, accepted, rejected = filter_blobs(
            labeled=labeled,
            features=features,
            min_area=self.config.min_blob_area,
            max_area=self.config.max_blob_area,
            circularity_min=self.config.circularity_min,
            solidity_min=self.config.solidity_min,
            aspect_ratio_min=ar_min,
            aspect_ratio_max=ar_max,
            exclude_border=self.config.exclude_border,
            border_margin_px=self.config.border_margin_px
        )

        num_blobs_after = len(accepted)

        # Step 5: Compute statistics
        total_positive_pixels = int(np.sum(filtered_mask))
        crack_ratio = total_positive_pixels / total_pixels if total_pixels > 0 else 0.0

        stats = {
            "num_blobs_before": num_blobs_before,
            "num_blobs_after": num_blobs_after,
            "total_positive_pixels": total_positive_pixels,
            "crack_ratio": crack_ratio,
            "accepted_blobs": accepted,
            "rejected_blobs": rejected,
        }

        # Build debug dictionary
        debug = {
            "mask_threshold": mask_threshold,
            "mask_after_morph": mask_after_morph,
            "labeled_components": labeled if num_blobs_before > 0 else None,
            "accepted_blobs": accepted,
            "rejected_blobs": rejected,
        }

        return filtered_mask, stats, debug

    def __repr__(self) -> str:
        return f"PostprocessPipeline(config={self.config})"


# ============================================================================
# Legacy Classes (Backward Compatibility)
# ============================================================================

class LegacyPostprocessPipeline:
    """
    DEPRECATED: Legacy pipeline for multi-class probability maps.

    Use PostprocessPipeline with PostprocessConfig for new code.
    """

    def __init__(self,
                 apply_smoothing: bool = False,
                 apply_morphology: bool = False,
                 min_confidence: float = 0.5):
        """
        Initialize postprocessing pipeline.

        Args:
            apply_smoothing: Whether to apply smoothing
            apply_morphology: Whether to apply morphological operations
            min_confidence: Minimum confidence threshold
        """
        self.apply_smoothing = apply_smoothing
        self.apply_morphology = apply_morphology
        self.min_confidence = min_confidence

    def __call__(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing pipeline.

        Args:
            prob_map: Probability map (H, W, C)

        Returns:
            Processed probability map
        """
        result = prob_map.copy()

        if self.apply_smoothing:
            result = self._smooth(result)

        if self.apply_morphology:
            result = self._morphological_ops(result)

        # Apply confidence threshold
        result = self._threshold(result, self.min_confidence)

        return result

    def _smooth(self, prob_map: np.ndarray,
                kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian smoothing."""
        try:
            from scipy.ndimage import gaussian_filter

            smoothed = np.zeros_like(prob_map)
            for i in range(prob_map.shape[2]):
                smoothed[:, :, i] = gaussian_filter(prob_map[:, :, i],
                                                     sigma=kernel_size/3)

            # Re-normalize to sum to 1
            total = smoothed.sum(axis=2, keepdims=True)
            total = np.where(total == 0, 1, total)
            smoothed = smoothed / total

            return smoothed
        except ImportError:
            return prob_map

    def _morphological_ops(self, prob_map: np.ndarray) -> np.ndarray:
        """Apply morphological operations to class predictions."""
        try:
            from scipy.ndimage import binary_opening, binary_closing

            # Get class predictions
            class_map = np.argmax(prob_map, axis=2)

            # Apply morphological operations to each class
            processed = np.zeros_like(class_map)
            for class_id in range(prob_map.shape[2]):
                mask = (class_map == class_id)
                mask = binary_opening(mask, iterations=1)
                mask = binary_closing(mask, iterations=1)
                processed[mask] = class_id

            # Convert back to probability map
            result = np.zeros_like(prob_map)
            for i in range(prob_map.shape[0]):
                for j in range(prob_map.shape[1]):
                    result[i, j, processed[i, j]] = 1.0

            return result
        except ImportError:
            return prob_map

    def _threshold(self, prob_map: np.ndarray,
                   min_conf: float) -> np.ndarray:
        """Apply confidence threshold."""
        max_probs = np.max(prob_map, axis=2)
        mask = max_probs >= min_conf

        result = prob_map.copy()
        result[~mask] = 0

        return result


def visualize_predictions(prob_map: np.ndarray,
                          class_names: list,
                          colormap: Optional[dict] = None) -> np.ndarray:
    """
    Create visualization of predictions.

    Args:
        prob_map: Probability map (H, W, C)
        class_names: Names of classes
        colormap: Optional color map for classes

    Returns:
        RGB visualization
    """
    class_map = np.argmax(prob_map, axis=2)

    if colormap is None:
        # Default colormap
        colormap = {
            0: [255, 0, 0],      # Red
            1: [0, 255, 0],      # Green
            2: [0, 0, 255],      # Blue
            3: [255, 255, 0],    # Yellow
        }

    h, w = class_map.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(class_names)):
        if class_id in colormap:
            mask = class_map == class_id
            viz[mask] = colormap[class_id]

    return viz


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on postprocessing pipeline."""
    print("Running pipeline sanity checks...")

    # Test 1: Config validation
    config = PostprocessConfig(prob_threshold=0.5, morph_close_size=5)
    config.validate()
    print("  ✓ Config validation works")

    # Test 2: Invalid config
    try:
        bad_config = PostprocessConfig(prob_threshold=1.5)
        bad_config.validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "prob_threshold" in str(e)
        print("  ✓ Invalid prob_threshold caught")

    try:
        bad_config = PostprocessConfig(morph_close_size=4)
        bad_config.validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "odd" in str(e).lower()
        print("  ✓ Even morph_close_size caught")

    # Test 3: Basic pipeline run
    prob_map = np.zeros((50, 50), dtype=np.float32)
    prob_map[10:20, 10:20] = 0.8  # Large blob
    prob_map[30:33, 30:33] = 0.7  # Small blob (9 pixels)

    config = PostprocessConfig(
        prob_threshold=0.5,
        min_blob_area=10  # Filter out small blob
    )
    pipeline = PostprocessPipeline(config)

    mask, stats = pipeline.run(prob_map)

    assert mask.dtype == bool
    assert mask.shape == prob_map.shape
    assert stats["num_blobs_before"] == 2
    assert stats["num_blobs_after"] == 1
    assert mask[15, 15] == True  # Large blob kept
    assert mask[31, 31] == False  # Small blob removed
    print("  ✓ Basic pipeline run works")

    # Test 4: Morphological closing
    prob_map_hole = np.zeros((30, 30), dtype=np.float32)
    prob_map_hole[10:20, 10:20] = 0.9
    prob_map_hole[14:16, 14:16] = 0.0  # Small hole

    config_morph = PostprocessConfig(
        prob_threshold=0.5,
        morph_close_size=5
    )
    pipeline_morph = PostprocessPipeline(config_morph)

    mask_morph, _ = pipeline_morph.run(prob_map_hole)

    # Hole should be filled by closing
    assert mask_morph[14, 14] == True, "Morphological closing should fill hole"
    print("  ✓ Morphological closing fills holes")

    # Test 5: Statistics computation
    prob_map_stats = np.zeros((100, 100), dtype=np.float32)
    prob_map_stats[20:40, 20:40] = 0.8  # 400 pixels

    config_stats = PostprocessConfig(prob_threshold=0.5)
    pipeline_stats = PostprocessPipeline(config_stats)

    _, stats = pipeline_stats.run(prob_map_stats)

    assert stats["total_positive_pixels"] == 400
    assert stats["crack_ratio"] == 400 / 10000  # 0.04
    print("  ✓ Statistics computed correctly")

    # Test 6: Border exclusion
    prob_map_border = np.zeros((30, 30), dtype=np.float32)
    prob_map_border[0:5, 10:15] = 0.8  # Touches top border
    prob_map_border[15:20, 15:20] = 0.8  # Interior

    config_border = PostprocessConfig(
        prob_threshold=0.5,
        exclude_border=True
    )
    pipeline_border = PostprocessPipeline(config_border)

    mask_border, stats_border = pipeline_border.run(prob_map_border)

    assert stats_border["num_blobs_after"] == 1
    assert mask_border[17, 17] == True  # Interior kept
    assert mask_border[2, 12] == False  # Border removed
    print("  ✓ Border exclusion works")

    # Test 7: Determinism
    prob_map_det = np.random.rand(50, 50).astype(np.float32)
    config_det = PostprocessConfig(prob_threshold=0.5, min_blob_area=5)
    pipeline_det = PostprocessPipeline(config_det)

    mask1, stats1 = pipeline_det.run(prob_map_det)
    mask2, stats2 = pipeline_det.run(prob_map_det)

    assert np.array_equal(mask1, mask2), "Pipeline should be deterministic"
    assert stats1["num_blobs_after"] == stats2["num_blobs_after"]
    print("  ✓ Pipeline is deterministic")

    # Test 8: run_debug returns same final result as run
    prob_map_debug = np.random.rand(60, 60).astype(np.float32)
    config_debug = PostprocessConfig(
        prob_threshold=0.5,
        morph_close_size=5,
        min_blob_area=10
    )
    pipeline_debug = PostprocessPipeline(config_debug)

    mask_regular, stats_regular = pipeline_debug.run(prob_map_debug)
    mask_debug, stats_debug, debug = pipeline_debug.run_debug(prob_map_debug)

    assert np.array_equal(mask_regular, mask_debug), "run_debug should produce same final mask as run"
    assert stats_regular["num_blobs_after"] == stats_debug["num_blobs_after"]
    assert stats_regular["total_positive_pixels"] == stats_debug["total_positive_pixels"]
    print("  ✓ run_debug produces same result as run (backward compatible)")

    # Test 9: Debug dictionary contains expected keys
    assert "mask_threshold" in debug, "debug should contain mask_threshold"
    assert "mask_after_morph" in debug, "debug should contain mask_after_morph"
    assert "labeled_components" in debug, "debug should contain labeled_components"
    assert "accepted_blobs" in debug, "debug should contain accepted_blobs"
    assert "rejected_blobs" in debug, "debug should contain rejected_blobs"
    print("  ✓ Debug dictionary contains all required keys")

    # Test 10: Debug intermediate masks have correct types and shapes
    assert debug["mask_threshold"].dtype == bool, "mask_threshold should be bool"
    assert debug["mask_after_morph"].dtype == bool, "mask_after_morph should be bool"
    assert debug["mask_threshold"].shape == prob_map_debug.shape
    assert debug["mask_after_morph"].shape == prob_map_debug.shape
    print("  ✓ Debug masks have correct types and shapes")

    # Test 11: Labeled components are int32 or None
    if debug["labeled_components"] is not None:
        assert debug["labeled_components"].dtype == np.int32, "labeled_components should be int32"
        assert debug["labeled_components"].shape == prob_map_debug.shape
    print("  ✓ Labeled components have correct type")

    # Test 12: Verify intermediate steps are captured correctly
    prob_map_steps = np.zeros((40, 40), dtype=np.float32)
    prob_map_steps[10:20, 10:20] = 0.8  # Blob with hole
    prob_map_steps[14:16, 14:16] = 0.3  # Hole below threshold

    config_steps = PostprocessConfig(
        prob_threshold=0.5,
        morph_close_size=5  # Should fill the hole
    )
    pipeline_steps = PostprocessPipeline(config_steps)

    _, _, debug_steps = pipeline_steps.run_debug(prob_map_steps)

    # After threshold, hole should still be False
    assert debug_steps["mask_threshold"][14, 14] == False, "Hole should be False after threshold"
    # After morph, hole should be filled
    assert debug_steps["mask_after_morph"][14, 14] == True, "Hole should be filled after morph"
    print("  ✓ Intermediate steps captured correctly (threshold vs morph)")

    # Test 13: Debug mode is deterministic
    mask_d1, stats_d1, debug_d1 = pipeline_debug.run_debug(prob_map_debug)
    mask_d2, stats_d2, debug_d2 = pipeline_debug.run_debug(prob_map_debug)

    assert np.array_equal(mask_d1, mask_d2), "run_debug should be deterministic (mask)"
    assert np.array_equal(debug_d1["mask_threshold"], debug_d2["mask_threshold"])
    assert np.array_equal(debug_d1["mask_after_morph"], debug_d2["mask_after_morph"])
    print("  ✓ run_debug is deterministic")

    # Test 14: Debug with no blobs (labeled_components should be None)
    prob_map_empty = np.zeros((30, 30), dtype=np.float32)
    config_empty = PostprocessConfig(prob_threshold=0.5)
    pipeline_empty = PostprocessPipeline(config_empty)

    _, _, debug_empty = pipeline_empty.run_debug(prob_map_empty)

    assert debug_empty["labeled_components"] is None, "labeled_components should be None when no blobs"
    assert len(debug_empty["accepted_blobs"]) == 0
    assert len(debug_empty["rejected_blobs"]) == 0
    print("  ✓ labeled_components is None when no blobs detected")

    # Test 15: Debug blob lists match stats
    prob_map_blobs = np.zeros((50, 50), dtype=np.float32)
    prob_map_blobs[5:15, 5:15] = 0.9  # Large blob (100 px)
    prob_map_blobs[20:22, 20:22] = 0.8  # Small blob (4 px)

    config_blobs = PostprocessConfig(
        prob_threshold=0.5,
        min_blob_area=10  # Filter small blob
    )
    pipeline_blobs = PostprocessPipeline(config_blobs)

    _, stats_blobs, debug_blobs = pipeline_blobs.run_debug(prob_map_blobs)

    assert len(debug_blobs["accepted_blobs"]) == stats_blobs["num_blobs_after"]
    assert len(debug_blobs["rejected_blobs"]) == stats_blobs["num_blobs_before"] - stats_blobs["num_blobs_after"]
    assert debug_blobs["accepted_blobs"] == stats_blobs["accepted_blobs"]
    assert debug_blobs["rejected_blobs"] == stats_blobs["rejected_blobs"]
    print("  ✓ Debug blob lists match stats")

    print("\n✅ All pipeline sanity checks passed (including run_debug)!")
    return True


if __name__ == "__main__":
    _run_sanity_checks()


