"""
Blob filtering utilities for connected component analysis.

Extracts connected components from binary masks and computes blob features
for filtering based on area, shape, and position criteria.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# Try to import image processing libraries
_CV2_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore

try:
    from scipy import ndimage
    from scipy.ndimage import label as scipy_label
    _SCIPY_AVAILABLE = True
except ImportError:
    ndimage = None  # type: ignore
    scipy_label = None  # type: ignore


@dataclass
class BlobFeatures:
    """
    Features computed for a single connected component (blob).

    Attributes:
        label: Unique identifier for this blob in the labeled image
        area: Number of pixels in the blob
        perimeter: Perimeter length in pixels (approximate)
        circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
        centroid: (row, col) center of mass
        bbox: (row_min, col_min, row_max, col_max) bounding box
        solidity: area / convex_hull_area (1.0 = convex shape)
        aspect_ratio: width / height of bounding box
        touches_border: Whether blob touches image border
    """
    label: int
    area: int
    perimeter: float
    circularity: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max)
    solidity: float
    aspect_ratio: float
    touches_border: bool


def _check_backend_available() -> str:
    """Check which backend is available for connected components."""
    if _CV2_AVAILABLE:
        return "cv2"
    elif _SCIPY_AVAILABLE:
        return "scipy"
    else:
        raise ImportError(
            "Blob filtering requires either OpenCV (cv2) or scipy.\n"
            "Install one with:\n"
            "  pip install opencv-python\n"
            "  or\n"
            "  pip install scipy"
        )


def extract_connected_components(
    mask: np.ndarray,
    connectivity: int = 8
) -> Tuple[np.ndarray, int]:
    """
    Extract connected components from a binary mask.

    Args:
        mask: Binary mask (H, W) with dtype bool or uint8
        connectivity: 4 or 8 for pixel connectivity

    Returns:
        Tuple of:
            - labeled: Array (H, W) where each pixel has its component label (0=background)
            - num_labels: Number of components found (excluding background)
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D")

    backend = _check_backend_available()

    # Convert to appropriate type
    mask_binary = mask.astype(bool) if mask.dtype != bool else mask

    if backend == "cv2":
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        cv_connectivity = 8 if connectivity == 8 else 4
        num_labels, labeled = cv2.connectedComponents(mask_uint8, connectivity=cv_connectivity)
        # num_labels includes background (label 0), so subtract 1
        return labeled, num_labels - 1

    else:  # scipy
        if connectivity == 8:
            struct = np.ones((3, 3), dtype=bool)
        else:  # 4-connectivity
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        labeled, num_labels = scipy_label(mask_binary, structure=struct)
        return labeled, num_labels


def compute_blob_features(
    mask: np.ndarray,
    labeled: np.ndarray,
    label_id: int
) -> BlobFeatures:
    """
    Compute features for a single blob.

    Args:
        mask: Original binary mask (H, W)
        labeled: Labeled image from extract_connected_components
        label_id: Label of the blob to analyze

    Returns:
        BlobFeatures dataclass with computed features
    """
    H, W = mask.shape

    # Get blob mask
    blob_mask = (labeled == label_id)

    # Area
    area = int(np.sum(blob_mask))

    if area == 0:
        # Empty blob
        return BlobFeatures(
            label=label_id,
            area=0,
            perimeter=0.0,
            circularity=0.0,
            centroid=(0.0, 0.0),
            bbox=(0, 0, 0, 0),
            solidity=0.0,
            aspect_ratio=1.0,
            touches_border=False
        )

    # Centroid
    rows, cols = np.where(blob_mask)
    centroid = (float(np.mean(rows)), float(np.mean(cols)))

    # Bounding box
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    bbox = (row_min, col_min, row_max, col_max)

    # Aspect ratio
    bbox_height = row_max - row_min + 1
    bbox_width = col_max - col_min + 1
    aspect_ratio = bbox_width / max(bbox_height, 1)

    # Touches border
    touches_border = (
        row_min == 0 or row_max == H - 1 or
        col_min == 0 or col_max == W - 1
    )

    # Perimeter and circularity
    perimeter = _compute_perimeter(blob_mask)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularity = min(circularity, 1.0)  # Clamp to [0, 1]
    else:
        circularity = 0.0

    # Solidity (area / convex hull area)
    solidity = _compute_solidity(blob_mask, area)

    return BlobFeatures(
        label=label_id,
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        centroid=centroid,
        bbox=bbox,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        touches_border=touches_border
    )


def _compute_perimeter(blob_mask: np.ndarray) -> float:
    """Compute perimeter of a blob using edge detection."""
    if _CV2_AVAILABLE:
        mask_uint8 = blob_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            return float(cv2.arcLength(contours[0], closed=True))
        return 0.0
    else:
        # Approximate perimeter by counting boundary pixels
        # A pixel is on boundary if it's True and has at least one False neighbor
        padded = np.pad(blob_mask, 1, mode='constant', constant_values=False)

        # Check 4-neighbors
        boundary = blob_mask & (
            ~padded[:-2, 1:-1] |  # top
            ~padded[2:, 1:-1] |   # bottom
            ~padded[1:-1, :-2] |  # left
            ~padded[1:-1, 2:]     # right
        )

        # Approximate perimeter (boundary pixels)
        return float(np.sum(boundary))


def _compute_solidity(blob_mask: np.ndarray, area: int) -> float:
    """Compute solidity = area / convex_hull_area."""
    if area == 0:
        return 0.0

    if _CV2_AVAILABLE:
        mask_uint8 = blob_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                return min(area / hull_area, 1.0)
        return 1.0
    else:
        # Approximate convex hull using bounding box as rough estimate
        # This is less accurate but doesn't require scipy.spatial
        rows, cols = np.where(blob_mask)
        bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
        # Solidity approximation: use area/bbox_area as lower bound estimate
        # Real solidity would be higher, so we scale up
        approx_solidity = min((area / bbox_area) * 1.27, 1.0)  # π/4 ≈ 0.785, 1/0.785 ≈ 1.27
        return approx_solidity


def extract_all_blob_features(
    mask: np.ndarray,
    connectivity: int = 8
) -> Tuple[np.ndarray, List[BlobFeatures]]:
    """
    Extract all connected components and compute features for each.

    Args:
        mask: Binary mask (H, W)
        connectivity: 4 or 8 for pixel connectivity

    Returns:
        Tuple of:
            - labeled: Labeled image (H, W)
            - features: List of BlobFeatures for each component
    """
    labeled, num_labels = extract_connected_components(mask, connectivity)

    features = []
    for label_id in range(1, num_labels + 1):  # Skip background (0)
        feat = compute_blob_features(mask, labeled, label_id)
        features.append(feat)

    return labeled, features


def filter_blobs(
    labeled: np.ndarray,
    features: List[BlobFeatures],
    min_area: int = 0,
    max_area: Optional[int] = None,
    circularity_min: Optional[float] = None,
    circularity_max: Optional[float] = None,
    solidity_min: Optional[float] = None,
    aspect_ratio_min: Optional[float] = None,
    aspect_ratio_max: Optional[float] = None,
    exclude_border: bool = False,
    border_margin_px: int = 0
) -> Tuple[np.ndarray, List[BlobFeatures], List[BlobFeatures]]:
    """
    Filter blobs based on feature criteria.

    Args:
        labeled: Labeled image (H, W)
        features: List of BlobFeatures for each blob
        min_area: Minimum blob area (pixels)
        max_area: Maximum blob area (None = no limit)
        circularity_min: Minimum circularity [0, 1]
        circularity_max: Maximum circularity [0, 1]
        solidity_min: Minimum solidity [0, 1]
        aspect_ratio_min: Minimum aspect ratio (width/height)
        aspect_ratio_max: Maximum aspect ratio
        exclude_border: Whether to exclude blobs touching border
        border_margin_px: Extra margin from border to exclude

    Returns:
        Tuple of:
            - filtered_mask: Boolean mask with only accepted blobs
            - accepted: List of BlobFeatures for accepted blobs
            - rejected: List of BlobFeatures for rejected blobs
    """
    H, W = labeled.shape
    accepted = []
    rejected = []

    for feat in features:
        # Check each criterion
        keep = True

        # Area
        if feat.area < min_area:
            keep = False
        if max_area is not None and feat.area > max_area:
            keep = False

        # Circularity
        if circularity_min is not None and feat.circularity < circularity_min:
            keep = False
        if circularity_max is not None and feat.circularity > circularity_max:
            keep = False

        # Solidity
        if solidity_min is not None and feat.solidity < solidity_min:
            keep = False

        # Aspect ratio
        if aspect_ratio_min is not None and feat.aspect_ratio < aspect_ratio_min:
            keep = False
        if aspect_ratio_max is not None and feat.aspect_ratio > aspect_ratio_max:
            keep = False

        # Border exclusion
        if exclude_border:
            if feat.touches_border:
                keep = False
            elif border_margin_px > 0:
                row_min, col_min, row_max, col_max = feat.bbox
                if (row_min < border_margin_px or
                    col_min < border_margin_px or
                    row_max >= H - border_margin_px or
                    col_max >= W - border_margin_px):
                    keep = False

        if keep:
            accepted.append(feat)
        else:
            rejected.append(feat)

    # Create filtered mask
    accepted_labels = {feat.label for feat in accepted}
    filtered_mask = np.isin(labeled, list(accepted_labels))

    return filtered_mask, accepted, rejected


def create_filtered_mask(
    mask: np.ndarray,
    min_area: int = 0,
    max_area: Optional[int] = None,
    circularity_min: Optional[float] = None,
    solidity_min: Optional[float] = None,
    aspect_ratio_range: Optional[Tuple[float, float]] = None,
    exclude_border: bool = False,
    border_margin_px: int = 0,
    connectivity: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    One-step blob filtering: extract, compute features, and filter.

    Args:
        mask: Binary mask (H, W)
        min_area: Minimum blob area
        max_area: Maximum blob area (None = no limit)
        circularity_min: Minimum circularity
        solidity_min: Minimum solidity
        aspect_ratio_range: (min, max) aspect ratio tuple
        exclude_border: Whether to exclude border-touching blobs
        border_margin_px: Extra border margin
        connectivity: 4 or 8 connectivity

    Returns:
        Tuple of:
            - filtered_mask: Boolean mask with only accepted blobs
            - stats: Dictionary with filtering statistics
    """
    # Extract and compute features
    labeled, features = extract_all_blob_features(mask, connectivity)

    num_blobs_before = len(features)

    # Parse aspect ratio range
    ar_min, ar_max = None, None
    if aspect_ratio_range is not None:
        ar_min, ar_max = aspect_ratio_range

    # Filter
    filtered_mask, accepted, rejected = filter_blobs(
        labeled=labeled,
        features=features,
        min_area=min_area,
        max_area=max_area,
        circularity_min=circularity_min,
        solidity_min=solidity_min,
        aspect_ratio_min=ar_min,
        aspect_ratio_max=ar_max,
        exclude_border=exclude_border,
        border_margin_px=border_margin_px
    )

    num_blobs_after = len(accepted)

    stats = {
        "num_blobs_before": num_blobs_before,
        "num_blobs_after": num_blobs_after,
        "num_blobs_rejected": len(rejected),
        "accepted_features": accepted,
        "rejected_features": rejected,
    }

    return filtered_mask, stats


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on blob filtering functions."""
    print("Running blob_filters sanity checks...")

    backend = _check_backend_available()
    print(f"  Using backend: {backend}")

    # Test 1: Extract connected components
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True   # Blob 1: 9 pixels
    mask[10:15, 10:15] = True  # Blob 2: 25 pixels

    labeled, num_labels = extract_connected_components(mask)

    assert num_labels == 2, f"Expected 2 components, got {num_labels}"
    assert labeled[3, 3] != 0, "Blob 1 should be labeled"
    assert labeled[12, 12] != 0, "Blob 2 should be labeled"
    assert labeled[0, 0] == 0, "Background should be 0"
    print("  ✓ Connected component extraction works")

    # Test 2: Compute blob features
    labeled, features = extract_all_blob_features(mask)

    assert len(features) == 2, f"Expected 2 blobs, got {len(features)}"

    areas = sorted([f.area for f in features])
    assert areas == [9, 25], f"Expected areas [9, 25], got {areas}"
    print("  ✓ Blob feature computation works")

    # Test 3: Filter by area
    filtered_mask, stats = create_filtered_mask(mask, min_area=10)

    assert stats["num_blobs_before"] == 2
    assert stats["num_blobs_after"] == 1
    assert np.sum(filtered_mask) == 25  # Only large blob remains
    print("  ✓ Area filtering works")

    # Test 4: Border exclusion
    mask_border = np.zeros((20, 20), dtype=bool)
    mask_border[0:3, 5:8] = True  # Touches top border
    mask_border[10:13, 10:13] = True  # Interior blob

    filtered_mask, stats = create_filtered_mask(mask_border, exclude_border=True)

    assert stats["num_blobs_after"] == 1, "Only interior blob should remain"
    assert filtered_mask[11, 11] == True, "Interior blob should be kept"
    assert filtered_mask[1, 6] == False, "Border blob should be removed"
    print("  ✓ Border exclusion works")

    # Test 5: Border margin
    mask_margin = np.zeros((30, 30), dtype=bool)
    mask_margin[3:6, 3:6] = True  # Near border (within 5px)
    mask_margin[15:18, 15:18] = True  # Far from border

    filtered_mask, stats = create_filtered_mask(
        mask_margin,
        exclude_border=True,
        border_margin_px=5
    )

    assert stats["num_blobs_after"] == 1
    assert filtered_mask[16, 16] == True
    print("  ✓ Border margin works")

    # Test 6: Circularity filter (square has circularity ~0.785)
    mask_shapes = np.zeros((50, 50), dtype=bool)
    # Create a square blob
    mask_shapes[10:20, 10:20] = True

    labeled, features = extract_all_blob_features(mask_shapes)
    square_circ = features[0].circularity

    assert 0.5 < square_circ < 1.0, f"Square circularity should be ~0.785, got {square_circ}"
    print(f"  ✓ Circularity computed: {square_circ:.3f}")

    print("\n✅ All blob_filters sanity checks passed!")
    return True


if __name__ == "__main__":
    _run_sanity_checks()
