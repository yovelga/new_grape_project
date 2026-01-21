"""
CNN Blob Verifier - Filter False Positives from Blob Detection

This module provides a CNN-based verifier to filter blobs detected by the LDA model.
Uses EfficientNet-B0 trained on grayscale patches to classify blobs as Crack vs Noise.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)


class BlobVerifier:
    """
    CNN-based blob verifier for filtering false positives.

    Uses EfficientNet-B0 modified for grayscale input and 2-class output.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the CNN verifier.

        Args:
            model_path: Path to the trained model weights (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing BlobVerifier on device: {self.device}")

        # Load model
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

        # Define transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        logger.info("BlobVerifier initialized successfully")

    def _build_model(self) -> nn.Module:
        """
        Build EfficientNet-B0 model modified for grayscale input and 2-class output.

        Returns:
            Modified EfficientNet model
        """
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(pretrained=False)

        # Modify first conv layer to accept 1 channel (grayscale)
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=1,  # Grayscale input
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Modify classifier to output 2 classes (Noise vs Crack)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 2)

        return model.to(self.device)

    def _load_weights(self):
        """Load trained weights from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Handle 3-channel to 1-channel conversion for first conv layer
            first_conv_key = 'features.0.0.weight'
            if first_conv_key in state_dict:
                conv_weight = state_dict[first_conv_key]
                # If model has 3 channels but we need 1, average across RGB channels
                if conv_weight.shape[1] == 3 and self.model.features[0][0].in_channels == 1:
                    logger.info("Converting 3-channel weights to 1-channel (averaging RGB)")
                    state_dict[first_conv_key] = conv_weight.mean(dim=1, keepdim=True)

            self.model.load_state_dict(state_dict)

            logger.info(f"Loaded model weights from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise

    def predict_blobs(
        self,
        rgb_image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        padding: int = 20,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict whether each blob is a true crack or false positive with detailed metadata.

        Args:
            rgb_image: Full RGB image as numpy array (H, W, 3)
            bboxes: List of bounding boxes [(x, y, w, h), ...]
            padding: Pixels to add around each bbox for context (enlarged mode)
            batch_size: Number of patches to process at once

        Returns:
            List of dictionaries with detailed metadata:
            [
                {'bbox': (x, y, w, h), 'prob_grape': 0.95, 'is_grape': True},
                {'bbox': (x, y, w, h), 'prob_grape': 0.12, 'is_grape': False},
                ...
            ]
        """
        if len(bboxes) == 0:
            return []

        logger.info(f"Verifying {len(bboxes)} blobs with CNN...")

        # Convert RGB to grayscale for processing
        if len(rgb_image.shape) == 3:
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = rgb_image

        h, w = gray_image.shape

        # Extract and transform patches
        patches = []
        for (x, y, bbox_w, bbox_h) in bboxes:
            # Add padding for context (enlarged mode)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bbox_w + padding)
            y2 = min(h, y + bbox_h + padding)

            # Crop patch
            patch = gray_image[y1:y2, x1:x2]

            # Convert to PIL Image and apply transforms
            patch_pil = Image.fromarray(patch)
            patch_tensor = self.transform(patch_pil)
            patches.append(patch_tensor)

        # Process in batches
        results = []
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size]
                batch_tensor = torch.stack(batch).to(self.device)

                # Forward pass
                outputs = self.model(batch_tensor)

                # Get probabilities using softmax
                probabilities = torch.softmax(outputs, dim=1)

                # Get predicted class (0=Noise, 1=Crack/Grape)
                predicted_probs, predicted_classes = torch.max(probabilities, 1)

                # Extract probabilities for grape class (class 1)
                grape_probs = probabilities[:, 1].cpu().numpy()
                predicted = predicted_classes.cpu().numpy()

                # Convert to list of dicts
                for idx, (prob_grape, pred_class) in enumerate(zip(grape_probs, predicted)):
                    batch_idx = i + idx
                    bbox = bboxes[batch_idx]

                    result = {
                        'bbox': bbox,
                        'prob_grape': float(prob_grape),
                        'is_grape': bool(pred_class == 1)
                    }
                    results.append(result)

                    # Log each blob
                    x, y, w_box, h_box = bbox
                    classification = "GRAPE" if result['is_grape'] else "NOISE"
                    logger.info(f"  Blob at [{x:4d},{y:4d},{w_box:3d},{h_box:3d}] -> "
                              f"Probability: {prob_grape*100:5.1f}%, Classified as: {classification}")

        grape_count = sum(1 for r in results if r['is_grape'])
        noise_count = len(results) - grape_count
        logger.info(f"CNN Results: {grape_count} grapes (kept), {noise_count} noise (filtered out)")

        return results

    def predict_single_patch(self, patch: np.ndarray) -> Tuple[bool, float]:
        """
        Predict a single patch.

        Args:
            patch: Grayscale patch as numpy array

        Returns:
            Tuple of (is_crack, confidence)
        """
        # Convert to PIL and apply transforms
        patch_pil = Image.fromarray(patch)
        patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(patch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            is_crack = (predicted.item() == 1)
            conf_value = confidence.item()

        return is_crack, conf_value


def extract_bboxes_from_mask(binary_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from binary mask using contour detection.

    Args:
        binary_mask: Binary mask (H, W) with True/1 for detected pixels

    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to uint8 if needed
    if binary_mask.dtype == bool:
        mask_uint8 = binary_mask.astype(np.uint8) * 255
    else:
        mask_uint8 = binary_mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))

    return bboxes


def filter_mask_by_bboxes(
    binary_mask: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    keep_flags: List[bool]
) -> np.ndarray:
    """
    Filter binary mask to keep only blobs that passed CNN verification.

    Args:
        binary_mask: Original binary mask
        bboxes: List of bounding boxes
        keep_flags: Boolean flags indicating which blobs to keep

    Returns:
        Filtered binary mask
    """
    if len(bboxes) != len(keep_flags):
        raise ValueError("Number of bboxes must match number of keep_flags")

    # Create output mask
    filtered_mask = np.zeros_like(binary_mask, dtype=bool)

    # Convert to uint8 for contour operations
    if binary_mask.dtype == bool:
        mask_uint8 = binary_mask.astype(np.uint8) * 255
    else:
        mask_uint8 = binary_mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only verified contours
    for idx, (contour, keep) in enumerate(zip(contours, keep_flags)):
        if keep:
            # Draw the contour on the filtered mask
            cv2.drawContours(filtered_mask.astype(np.uint8), [contour], -1, 1, -1)

    return filtered_mask.astype(bool)

