import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def get_model(num_classes=2):
    print("Loading pre-trained EfficientNet model...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    print("Model loaded and modified for {} classes.".format(num_classes))
    return model
