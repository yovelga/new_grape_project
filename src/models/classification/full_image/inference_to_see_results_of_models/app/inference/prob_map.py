"""
Probability map generation for full images.

Creates pixel-wise or patch-wise probability maps.
Supports both patch-based CNN inference and pixel-level classification with chunked inference.
"""

import numpy as np
import time
from typing import Tuple, Optional, Callable

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# Local imports (within app scope)
from ..utils.logging import logger
from ..config.types import PreprocessConfig
from ..preprocess.spectral import (
    snv_normalize,
    l2_normalize,
    select_wavelength_range,
    select_bands,
)


def _preprocess_pixels(
    pixels: np.ndarray,
    preprocess_cfg: PreprocessConfig
) -> np.ndarray:
    """
    Apply preprocessing pipeline to flattened pixels.

    Uses functions from app.preprocess.spectral for consistency.

    Args:
        pixels: Input pixels (N, C) as float32
        preprocess_cfg: PreprocessConfig from app.config.types

    Returns:
        Preprocessed pixels (N, C_out) as float32
    """
    # Band selection (explicit indices take precedence over wavelength filtering)
    if preprocess_cfg.needs_band_selection:
        pixels = select_bands(pixels, preprocess_cfg.band_indices)
    elif preprocess_cfg.needs_wavelength_filter and preprocess_cfg.wavelengths is not None:
        pixels, _ = select_wavelength_range(
            pixels,
            preprocess_cfg.wavelengths,
            preprocess_cfg.wl_min,
            preprocess_cfg.wl_max
        )

    # SNV normalization
    if preprocess_cfg.use_snv:
        pixels = snv_normalize(pixels)

    # L2 normalization (per-pixel)
    if preprocess_cfg.use_l2_norm:
        pixels = l2_normalize(pixels)

    return pixels.astype(np.float32)


def build_prob_map(
    cube: np.ndarray,
    model,  # ModelAdapter from adapters_new.py
    preprocess_cfg: PreprocessConfig,
    target_class_index: Optional[int] = None,
    chunk_size: int = 200_000
) -> np.ndarray:
    """
    Build probability map from hyperspectral cube using pixel-level classification.

    Converts an HSI cube into a probability map for a chosen class index,
    with chunked inference for memory efficiency.

    Args:
        cube: Hyperspectral cube with shape (H, W, C) where C is spectral channels
        model: ModelAdapter instance with predict_proba method
        preprocess_cfg: Configuration for spectral preprocessing
        target_class_index: Index of target class for output probabilities.
            - For binary classification: None defaults to positive class (index 1)
            - For multiclass: must be specified
        chunk_size: Number of pixels to process per batch (default 200,000)

    Returns:
        Probability map with shape (H, W) as float32

    Raises:
        ValueError: If cube shape is invalid or target_class_index is out of range
        TypeError: If inputs are wrong type
    """
    # ===== Input Validation =====
    if not isinstance(cube, np.ndarray):
        raise TypeError(f"cube must be numpy.ndarray, got {type(cube).__name__}")

    if cube.ndim != 3:
        raise ValueError(
            f"cube must have 3 dimensions (H, W, C), got {cube.ndim}D with shape {cube.shape}"
        )

    H, W, C = cube.shape
    total_pixels = H * W

    logger.info(f"Building probability map for cube shape ({H}, {W}, {C})")
    logger.info(f"Total pixels: {total_pixels:,}, chunk size: {chunk_size:,}")

    # Validate preprocess config
    preprocess_cfg.validate()

    start_time = time.time()

    # ===== Flatten cube to pixels =====
    flatten_start = time.time()
    pixels = cube.reshape(total_pixels, C).astype(np.float32)
    logger.debug(f"Flatten to ({total_pixels}, {C}) took {time.time() - flatten_start:.3f}s")

    # ===== Preprocess pixels =====
    preprocess_start = time.time()
    pixels_preprocessed = _preprocess_pixels(pixels, preprocess_cfg)
    C_preprocessed = pixels_preprocessed.shape[1]
    logger.info(
        f"Preprocessing took {time.time() - preprocess_start:.3f}s, "
        f"output channels: {C_preprocessed}"
    )

    # ===== Determine target class index =====
    n_classes = model.n_classes
    is_binary = n_classes == 2

    if target_class_index is None:
        if is_binary:
            target_class_index = 1  # Default to positive class for binary
            logger.info(f"Binary model: using positive class (index {target_class_index})")
        else:
            raise ValueError(
                f"target_class_index must be specified for multiclass model "
                f"with {n_classes} classes"
            )

    if target_class_index < 0 or target_class_index >= n_classes:
        raise ValueError(
            f"target_class_index {target_class_index} out of range [0, {n_classes})"
        )

    logger.info(f"Model: {n_classes} classes, target class index: {target_class_index}")

    # ===== Chunked inference =====
    prob_flat = np.zeros(total_pixels, dtype=np.float32)
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size

    inference_start = time.time()
    logger.info(f"Running chunked inference ({num_chunks} chunks)...")

    for chunk_idx in range(num_chunks):
        chunk_start_time = time.time()

        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_pixels)

        chunk_pixels = pixels_preprocessed[start_idx:end_idx]

        # Get probabilities from model
        try:
            proba = model.predict_proba(chunk_pixels)
        except Exception as e:
            raise RuntimeError(
                f"Model inference failed at chunk {chunk_idx + 1}/{num_chunks}: {e}"
            ) from e

        # Validate output shape
        expected_shape = (end_idx - start_idx, n_classes)
        if proba.shape != expected_shape:
            raise ValueError(
                f"Model returned wrong shape: expected {expected_shape}, got {proba.shape}"
            )

        # Extract target class probability
        prob_flat[start_idx:end_idx] = proba[:, target_class_index].astype(np.float32)

        chunk_time = time.time() - chunk_start_time
        if (chunk_idx + 1) % max(1, num_chunks // 10) == 0 or chunk_idx == num_chunks - 1:
            progress = (chunk_idx + 1) / num_chunks * 100
            logger.info(
                f"Chunk {chunk_idx + 1}/{num_chunks} ({progress:.1f}%) "
                f"[{end_idx - start_idx} pixels, {chunk_time:.3f}s]"
            )

    inference_time = time.time() - inference_start
    logger.info(f"Inference took {inference_time:.3f}s ({total_pixels / inference_time:.0f} px/s)")

    # ===== Reshape to 2D map =====
    prob_map = prob_flat.reshape(H, W)

    total_time = time.time() - start_time
    logger.info(
        f"Probability map built in {total_time:.3f}s total, "
        f"output shape: {prob_map.shape}, dtype: {prob_map.dtype}"
    )

    return prob_map


class ProbabilityMapGenerator:
    """Generate probability maps for images."""

    def __init__(self,
                 model_adapter,
                 patch_size: Tuple[int, int] = (224, 224),
                 stride: Optional[Tuple[int, int]] = None,
                 device: str = "cpu"):
        """
        Initialize probability map generator.

        Args:
            model_adapter: Model adapter for inference
            patch_size: Size of patches to extract
            stride: Stride for patch extraction (defaults to patch_size)
            device: Device to run on
        """
        self.model_adapter = model_adapter
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.device = device

    def generate(self,
                 image: np.ndarray,
                 preprocess_fn: Optional[Callable] = None) -> np.ndarray:
        """
        Generate probability map for image.

        Args:
            image: Input image (H, W, C)
            preprocess_fn: Optional preprocessing function

        Returns:
            Probability map (H, W, num_classes)
        """
        h, w = image.shape[:2]
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride

        # Calculate output dimensions
        out_h = ((h - patch_h) // stride_h) + 1
        out_w = ((w - patch_w) // stride_w) + 1

        # Extract patches
        patches = self._extract_patches(image)

        if preprocess_fn:
            patches = [preprocess_fn(p) for p in patches]

        # Convert to tensors
        patches_tensor = torch.stack([
            torch.from_numpy(p).permute(2, 0, 1).float() for p in patches
        ])

        # Run inference
        probs = self.model_adapter.predict_proba(patches_tensor)
        probs = probs.cpu().numpy()

        # Reshape to map
        num_classes = probs.shape[1]
        prob_map = np.zeros((out_h, out_w, num_classes))

        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                prob_map[i, j] = probs[idx]
                idx += 1

        return prob_map

    def _extract_patches(self, image: np.ndarray) -> list:
        """
        Extract patches from image.

        Args:
            image: Input image

        Returns:
            List of patches
        """
        h, w = image.shape[:2]
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride

        patches = []

        for i in range(0, h - patch_h + 1, stride_h):
            for j in range(0, w - patch_w + 1, stride_w):
                patch = image[i:i+patch_h, j:j+patch_w]
                patches.append(patch)

        return patches

    def upsample_map(self,
                     prob_map: np.ndarray,
                     target_size: Tuple[int, int]) -> np.ndarray:
        """
        Upsample probability map to target size.

        Args:
            prob_map: Probability map (H, W, C)
            target_size: Target (height, width)

        Returns:
            Upsampled probability map
        """
        from scipy.ndimage import zoom

        h_target, w_target = target_size
        h_curr, w_curr = prob_map.shape[:2]

        zoom_factors = (h_target / h_curr, w_target / w_curr, 1)

        return zoom(prob_map, zoom_factors, order=1)
