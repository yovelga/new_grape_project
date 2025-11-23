import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
import os


class PointSegmenter:
    """Per‑point segmentation wrapper around SAM2ImagePredictor."""

    def __init__(self, sam2_model):
        self.model = sam2_model  # instance of SAM2ImagePredictor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _points_to_np(points: List[Tuple[int, int]]) -> np.ndarray:
        return np.asarray(points, dtype=np.int32)

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def segment_object(
        self,
        image_path: str,
        point_coords: List[Tuple[int, int]],
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return RGB image and **boolean** mask of the segmented object."""

        # 1. load + set image
        image = self.load_image(image_path)
        self.model.set_image(image)

        # 2. prepare prompts
        pts = self._points_to_np(point_coords)
        labels = np.ones(len(pts), dtype=int)

        # 3. predict
        masks, _ious, _ = self.model.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=False,
        )
        raw_mask = masks[0]  # (H,W) float or 0/1
        mask = raw_mask.astype(bool) if raw_mask.dtype != bool else raw_mask

        # 4. optional viz
        if save_path:
            self.save_visualization(image, mask, save_path)

        return image, mask

    def segment_object_from_array(
        self,
        image_rgb: np.ndarray,
        point_coords: List[Tuple[int, int]],
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment object from numpy array instead of file path.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
            point_coords: List of (x, y) tuples
            save_path: Optional path to save visualization

        Returns:
            Tuple of (image_rgb, boolean_mask)
        """
        # 1. set image directly
        self.model.set_image(image_rgb)

        # 2. prepare prompts
        pts = self._points_to_np(point_coords)
        labels = np.ones(len(pts), dtype=int)

        # 3. predict
        masks, _ious, _ = self.model.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=False,
        )
        raw_mask = masks[0]  # (H,W) float or 0/1
        mask = raw_mask.astype(bool) if raw_mask.dtype != bool else raw_mask

        # 4. optional viz
        if save_path:
            self.save_visualization(image_rgb, mask, save_path)

        return image_rgb, mask

    # ------------------------------------------------------------------
    def save_visualization(self, image: np.ndarray, mask: np.ndarray, save_path: str):
        overlay = image.copy()
        overlay[mask] = overlay[mask] * 0.7 + np.array([255, 0, 0]) * 0.3
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------


def create_point_segmenter(sam2_model):
    """Factory wrapper so external code remains unchanged."""
    return PointSegmenter(sam2_model)
