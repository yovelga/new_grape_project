"""
Script to visualize the augmentation effects
Shows original and augmented versions side by side
"""
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from data_transforms import get_train_transforms
from config import TRAIN_DIR

def visualize_augmentations(image_path, num_augmentations=6):
    """
    Visualize the effect of data augmentation

    Args:
        image_path: Path to an image file
        num_augmentations: Number of augmented versions to show
    """
    transform = get_train_transforms()

    # Load original image
    img = Image.open(image_path).convert('RGB')

    # Create subplot
    fig, axes = plt.subplots(2, (num_augmentations + 2) // 2, figsize=(20, 8))
    fig.suptitle('Augmentation Examples (Including Darker Variations)', fontsize=16)
    axes = axes.flatten()

    # Show original
    axes[0].imshow(img)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Show augmented versions
    for i in range(1, num_augmentations):
        augmented = transform(img)

        # Convert tensor to numpy for display
        if augmented.shape[0] == 1:  # Grayscale
            aug_img = augmented.squeeze().numpy()
            axes[i].imshow(aug_img, cmap='gray')
        else:  # RGB
            aug_img = augmented.permute(1, 2, 0).numpy()
            axes[i].imshow(aug_img)

        axes[i].set_title(f'Augmented {i}', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved augmentation examples to: augmentation_examples.png")
    plt.show()


def find_sample_image():
    """Find a sample image from the training directory"""
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                return os.path.join(class_dir, images[0])
    return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUGMENTATION VISUALIZATION")
    print("="*60)

    # Try to find a sample image
    sample_image = find_sample_image()

    if sample_image:
        print(f"Using sample image: {sample_image}")
        print(f"\nGenerating augmentation examples...")
        print(f"Note: You'll see variations including:")
        print(f"  - Dark images (brightness 0.3-0.7)")
        print(f"  - Normal images (brightness 0.8-1.2)")
        print(f"  - Bright images (brightness 1.2-1.5)")
        print(f"  - Different contrasts, saturations, and hues")
        print(f"  - Some with blur or sharpness adjustments")
        visualize_augmentations(sample_image, num_augmentations=6)
    else:
        print(f"⚠ Could not find sample images in: {TRAIN_DIR}")
        print(f"Please provide a path to an image:")
        image_path = input("Image path: ").strip('"')
        if os.path.exists(image_path):
            visualize_augmentations(image_path, num_augmentations=6)
        else:
            print("❌ Image not found!")

    print("="*60 + "\n")

