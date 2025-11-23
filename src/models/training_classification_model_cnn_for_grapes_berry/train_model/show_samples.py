import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import GrapeDataset
from config import TRAIN_DIR
from data_transforms import get_train_transforms


def unnormalize(img_tensor, mean, std):
    """
    מבטל את הנירמול כדי להציג את התמונה בצבעים תקינים.
    """
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def show_samples_per_class(num_samples_per_class=10):
    # הגדרת טרנספורמציות לאימון (ללא שינוי לעומת האימון)
    transform = get_train_transforms()

    # טעינת דאטה סט האימון
    dataset = GrapeDataset(TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # רשימות לאחסון תמונות לכל קלאס
    grape_images = []
    not_grape_images = []

    # איסוף תמונות עד להשגת מספר נדרש לכל קלאס
    for images, labels in dataloader:
        for i in range(len(labels)):
            label = labels[i].item()
            if label == 1 and len(grape_images) < num_samples_per_class:
                grape_images.append(images[i])
            elif label == 0 and len(not_grape_images) < num_samples_per_class:
                not_grape_images.append(images[i])
            if (
                len(grape_images) >= num_samples_per_class
                and len(not_grape_images) >= num_samples_per_class
            ):
                break
        if (
            len(grape_images) >= num_samples_per_class
            and len(not_grape_images) >= num_samples_per_class
        ):
            break

    # בדיקה אם אספנו מספיק תמונות
    if (
        len(grape_images) < num_samples_per_class
        or len(not_grape_images) < num_samples_per_class
    ):
        print("לא נמצאו מספיק תמונות לכל קלאס.")
        return

    # הגדרת ערכי נירמול כפי שהוגדרו בטרנספורמציות
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # יצירת גריד של 2 שורות ו-10 עמודות
    fig, axes = plt.subplots(
        2, num_samples_per_class, figsize=(num_samples_per_class * 3, 6)
    )

    # הצגת תמונות עבור קלאס Grape (1)
    for i in range(num_samples_per_class):
        img = unnormalize(grape_images[i], mean, std)
        axes[0, i].imshow(img)
        axes[0, i].set_title("Grape")
        axes[0, i].axis("off")

    # הצגת תמונות עבור קלאס Not Grape (0)
    for i in range(num_samples_per_class):
        img = unnormalize(not_grape_images[i], mean, std)
        axes[1, i].imshow(img)
        axes[1, i].set_title("Not Grape")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_samples_per_class(num_samples_per_class=20)
