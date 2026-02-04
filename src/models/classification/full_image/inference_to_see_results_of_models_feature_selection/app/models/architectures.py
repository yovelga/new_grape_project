"""
CNN architectures for binary classification.

Provides EfficientNet and other CNN models for inference.
"""

import torch
import torch.nn as nn
from typing import Optional


class EfficientNetBinary(nn.Module):
    """
    EfficientNet-B0 based binary classifier.

    Uses pretrained EfficientNet-B0 as feature extractor with custom classifier head.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2):
        """
        Initialize EfficientNet binary classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for classifier
        """
        super().__init__()

        # Import here to avoid dependency if not used
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        except ImportError:
            raise ImportError("torchvision required for EfficientNet. Install: pip install torchvision")

        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification layer.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Features (B, feature_dim)
        """
        # Forward through all layers except classifier
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification.

    Lightweight alternative to EfficientNet.
    """

    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        """
        Initialize simple CNN.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
        """
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def create_efficientnet_binary(num_classes: int = 2,
                               pretrained: bool = True,
                               checkpoint_path: Optional[str] = None,
                               device: str = "cpu") -> EfficientNetBinary:
    """
    Create EfficientNet binary classifier.

    Args:
        num_classes: Number of classes
        pretrained: Use ImageNet pretrained weights
        checkpoint_path: Optional path to trained weights
        device: Device to load model on

    Returns:
        EfficientNet model
    """
    model = EfficientNetBinary(num_classes=num_classes, pretrained=pretrained)

    if checkpoint_path:
        from .loader import ModelLoader
        model = ModelLoader.load_model(model, checkpoint_path, device)
    else:
        model = model.to(device)
        model.eval()

    return model
