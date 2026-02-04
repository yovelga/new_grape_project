"""
Model loading utilities.
"""

import torch
from pathlib import Path
from typing import Any, Dict


class ModelLoader:
    """Load trained models for inference."""

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Dictionary containing model state and metadata
        """
        checkpoint_path = Path(path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        return checkpoint

    @staticmethod
    def load_model(model: torch.nn.Module,
                   checkpoint_path: str,
                   device: str = "cpu") -> torch.nn.Module:
        """
        Load model weights from checkpoint.

        Args:
            model: Model instance
            checkpoint_path: Path to checkpoint
            device: Device to load on

        Returns:
            Model with loaded weights
        """
        checkpoint = ModelLoader.load_checkpoint(checkpoint_path, device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        return model

    @staticmethod
    def get_checkpoint_info(path: str) -> Dict[str, Any]:
        """
        Get metadata from checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = ModelLoader.load_checkpoint(path, device="cpu")

        info = {
            'keys': list(checkpoint.keys()),
            'has_model_state': 'model_state_dict' in checkpoint or 'state_dict' in checkpoint,
        }

        # Extract additional metadata if available
        for key in ['epoch', 'accuracy', 'loss', 'class_names', 'config']:
            if key in checkpoint:
                info[key] = checkpoint[key]

        return info
