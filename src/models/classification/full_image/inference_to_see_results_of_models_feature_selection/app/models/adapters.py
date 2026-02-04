"""
Model adapters for different architectures.

Provides unified interface for various model types.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ModelAdapter:
    """Base adapter for inference models."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize adapter.

        Args:
            model: PyTorch model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference.

        Args:
            x: Input tensor

        Returns:
            Model predictions
        """
        x = x.to(self.device)
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            x: Input tensor

        Returns:
            Class probabilities
        """
        logits = self.predict(x)
        return torch.softmax(logits, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from model.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        raise NotImplementedError("Feature extraction not implemented for base adapter")


class CNNAdapter(ModelAdapter):
    """Adapter for CNN-based models."""

    def __init__(self, model: nn.Module, device: str = "cpu",
                 feature_layer: Optional[str] = None):
        """
        Initialize CNN adapter.

        Args:
            model: CNN model
            device: Device to run on
            feature_layer: Name of layer to extract features from
        """
        super().__init__(model, device)
        self.feature_layer = feature_layer
        self._features = None

        if feature_layer:
            self._register_hook()

    def _register_hook(self):
        """Register forward hook for feature extraction."""
        def hook(module, input, output):
            self._features = output

        # Find and register hook on specified layer
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook)
                break

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from specified layer."""
        self._features = None
        _ = self.predict(x)

        if self._features is None:
            raise ValueError(f"Feature layer '{self.feature_layer}' not found")

        return self._features


class EnsembleAdapter(ModelAdapter):
    """Adapter for ensemble of models."""

    def __init__(self, models: list, device: str = "cpu",
                 aggregation: str = "mean"):
        """
        Initialize ensemble adapter.

        Args:
            models: List of models
            device: Device to run on
            aggregation: Aggregation method ('mean', 'max', 'voting')
        """
        self.models = [m.to(device).eval() for m in models]
        self.device = device
        self.aggregation = aggregation

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on ensemble."""
        x = x.to(self.device)

        predictions = [model(x) for model in self.models]

        if self.aggregation == "mean":
            return torch.mean(torch.stack(predictions), dim=0)
        elif self.aggregation == "max":
            return torch.max(torch.stack(predictions), dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
