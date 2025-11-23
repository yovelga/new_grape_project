import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

def get_model(num_classes=2):
    print("Loading pre-trained EfficientNet model...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    print("Model loaded and modified for {} classes.".format(num_classes))
    return model




def get_model_gray(num_classes=2):
    print("Loading pre-trained EfficientNet model...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # שינוי ה־conv הראשון לערוץ בודד:
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(  # <--- במקום torch.Conv2d
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    print("Model loaded and modified for {} classes.".format(num_classes))
    return model