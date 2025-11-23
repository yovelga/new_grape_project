import matplotlib.pyplot as plt
import torch
from dataset_multi import GrapeDataset
from data_transforms import get_test_transforms
from config import TRAIN_DIR

idx = 0

ds_original = GrapeDataset(
    TRAIN_DIR, input_mode="original", transform=get_test_transforms()
)
ds_enlarged = GrapeDataset(
    TRAIN_DIR, input_mode="enlarged", transform=get_test_transforms()
)
ds_segmentation = GrapeDataset(
    TRAIN_DIR, input_mode="segmentation", transform=get_test_transforms()
)

img_original, label = ds_original[idx]
img_enlarged, _ = ds_enlarged[idx]
img_segmentation, _ = ds_segmentation[idx]


def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img = img_tensor * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1)


import matplotlib.pyplot as plt

# Original Crop
plt.figure(figsize=(4, 4))
plt.imshow(unnormalize(img_original))
plt.title("Original Crop")
plt.axis("off")
plt.show()

# Enlarged Crop
plt.figure(figsize=(4, 4))
plt.imshow(unnormalize(img_enlarged))
plt.title("Enlarged Crop")
plt.axis("off")
plt.show()

# Segmentation Overlay
plt.figure(figsize=(4, 4))
plt.imshow(unnormalize(img_segmentation))
plt.title("Segmentation Overlay")
plt.axis("off")
plt.show()
