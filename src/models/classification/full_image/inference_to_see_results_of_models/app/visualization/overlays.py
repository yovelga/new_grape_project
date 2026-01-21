"""
Visualization utilities for inference results.

Provides functions for creating overlays, heatmaps, and grid visualizations.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


# Color buckets for grid visualization: (threshold_percent, color_BGR)
GRID_COLOR_BUCKETS = [
    (50, (0, 0, 180)),    # Dark red for >50%
    (40, (0, 30, 200)),   # Red for >40%
    (30, (0, 120, 255)),  # Orange for >30%
    (20, (0, 200, 255)),  # Yellow for >20%
]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 range [0, 255].

    Args:
        image: Input image (any dtype)

    Returns:
        Normalized uint8 image
    """
    if image.dtype == np.uint8:
        return image

    img_min = image.min()
    img_max = image.max()

    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return normalized


def create_heatmap(prob_map: np.ndarray,
                   class_index: int = 1,
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Create heatmap visualization from probability map.

    Args:
        prob_map: Probability map (H, W, C)
        class_index: Which class to visualize
        colormap: OpenCV colormap constant

    Returns:
        RGB heatmap (H, W, 3)
    """
    if prob_map.ndim == 2:
        probs = prob_map
    else:
        probs = prob_map[:, :, class_index]

    # Normalize to 0-255
    probs_uint8 = (probs * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(probs_uint8, colormap)

    return heatmap


def create_binary_overlay(base_image: np.ndarray,
                         binary_mask: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 255),
                         alpha: float = 0.35) -> np.ndarray:
    """
    Create overlay of binary mask on base image.

    Args:
        base_image: Base image (H, W) or (H, W, 3)
        binary_mask: Binary mask (H, W)
        color: Overlay color in BGR
        alpha: Transparency (0-1)

    Returns:
        RGB overlay (H, W, 3)
    """
    # Convert base to RGB if grayscale
    if base_image.ndim == 2:
        base_rgb = cv2.cvtColor(normalize_to_uint8(base_image), cv2.COLOR_GRAY2BGR)
    else:
        base_rgb = base_image.copy()

    # Ensure base is uint8
    if base_rgb.dtype != np.uint8:
        base_rgb = normalize_to_uint8(base_rgb)

    # Create colored overlay
    overlay = base_rgb.copy()
    color_array = np.array(color, dtype=np.uint8)
    overlay[binary_mask] = (alpha * color_array + (1 - alpha) * base_rgb[binary_mask]).astype(np.uint8)

    return overlay


def create_grid_overlay(base_image: np.ndarray,
                       grid_stats: List[Dict],
                       alpha: float = 0.35,
                       draw_borders: bool = True) -> np.ndarray:
    """
    Create grid visualization overlay on base image.

    Args:
        base_image: Base image (H, W) or (H, W, 3)
        grid_stats: List of grid cell statistics from GridAnalyzer
        alpha: Transparency for colored cells
        draw_borders: Whether to draw cell borders

    Returns:
        RGB overlay with colored grid cells
    """
    # Convert to RGB
    if base_image.ndim == 2:
        rgb = cv2.cvtColor(normalize_to_uint8(base_image), cv2.COLOR_GRAY2RGB)
    else:
        rgb = base_image.copy()

    if rgb.dtype != np.uint8:
        rgb = normalize_to_uint8(rgb)

    overlay = rgb.copy()

    # Draw each cell
    for cell in grid_stats:
        percent = cell['percent_cracked']

        # Determine color based on percentage
        color = None
        for threshold, col in GRID_COLOR_BUCKETS:
            if percent >= threshold:
                color = col
                break

        if color is not None:
            r0, c0 = cell['row0'], cell['col0']
            r1, c1 = cell['row1'], cell['col1']

            # Fill cell with color
            cv2.rectangle(overlay, (c0, r0), (c1 - 1, r1 - 1), color, -1)

            # Draw border if requested
            if draw_borders:
                cv2.rectangle(overlay, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 0), 1)

    # Blend overlay with original
    result = cv2.addWeighted(overlay, max(alpha, 0.45), rgb, 1 - max(alpha, 0.45), 0)

    return result


def create_rgb_composite(cube: np.ndarray,
                        r_band: int = None,
                        g_band: int = None,
                        b_band: int = None) -> np.ndarray:
    """
    Create RGB composite from hyperspectral cube.

    Args:
        cube: Hyperspectral cube (H, W, C)
        r_band: Red band index (if None, uses band at 90% of spectrum)
        g_band: Green band index (if None, uses band at 50% of spectrum)
        b_band: Blue band index (if None, uses band at 10% of spectrum)

    Returns:
        RGB image (H, W, 3)
    """
    h, w, num_bands = cube.shape

    # Auto-select bands if not specified
    if r_band is None:
        r_band = int(num_bands * 0.9)
    if g_band is None:
        g_band = int(num_bands * 0.5)
    if b_band is None:
        b_band = int(num_bands * 0.1)

    # Clamp to valid range
    r_band = max(0, min(num_bands - 1, r_band))
    g_band = max(0, min(num_bands - 1, g_band))
    b_band = max(0, min(num_bands - 1, b_band))

    # Extract and normalize bands
    r_channel = normalize_to_uint8(cube[:, :, r_band])
    g_channel = normalize_to_uint8(cube[:, :, g_band])
    b_channel = normalize_to_uint8(cube[:, :, b_band])

    # Merge to RGB
    rgb = cv2.merge([b_channel, g_channel, r_channel])  # OpenCV uses BGR

    return rgb


def add_text_overlay(image: np.ndarray,
                     text: str,
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 1.0,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to image.

    Args:
        image: Input image
        text: Text to add
        position: (x, y) position for text
        font_scale: Font size scale
        color: Text color in BGR
        thickness: Text thickness

    Returns:
        Image with text overlay
    """
    result = image.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return result


def create_side_by_side(images: List[np.ndarray],
                       titles: Optional[List[str]] = None,
                       padding: int = 10) -> np.ndarray:
    """
    Create side-by-side visualization of multiple images.

    Args:
        images: List of images (all same height)
        titles: Optional titles for each image
        padding: Padding between images

    Returns:
        Combined image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Ensure all images are RGB and same height
    rgb_images = []
    max_height = max(img.shape[0] for img in images)

    for img in images:
        # Convert to RGB if needed
        if img.ndim == 2:
            img = cv2.cvtColor(normalize_to_uint8(img), cv2.COLOR_GRAY2RGB)
        elif img.shape[2] != 3:
            img = cv2.cvtColor(normalize_to_uint8(img[:, :, 0]), cv2.COLOR_GRAY2RGB)

        # Ensure uint8
        if img.dtype != np.uint8:
            img = normalize_to_uint8(img)

        # Resize to max height if needed
        if img.shape[0] != max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height), interpolation=cv2.INTER_LINEAR)

        rgb_images.append(img)

    # Add titles if provided
    if titles:
        titled_images = []
        for img, title in zip(rgb_images, titles):
            img_with_title = add_text_overlay(img, title, position=(10, 30))
            titled_images.append(img_with_title)
        rgb_images = titled_images

    # Create horizontal concatenation with padding
    result_parts = []
    for i, img in enumerate(rgb_images):
        result_parts.append(img)
        if i < len(rgb_images) - 1:
            # Add white padding
            pad = np.ones((max_height, padding, 3), dtype=np.uint8) * 255
            result_parts.append(pad)

    result = np.hstack(result_parts)

    return result
