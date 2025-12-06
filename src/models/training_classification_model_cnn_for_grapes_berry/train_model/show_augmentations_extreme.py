"""
Script to visualize the augmentation effects
Shows original and augmented versions side by side with extensive variations
"""
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile as tiff
import numpy as np
from data_transforms import get_train_transforms
from config import TRAIN_DIR

def visualize_augmentations(image_path, num_augmentations=12):
    """
    Visualize the effect of data augmentation

    Args:
        image_path: Path to an image file (supports .tif, .jpg, .png)
        num_augmentations: Number of augmented versions to show
    """
    transform = get_train_transforms()

    # Load image based on extension
    if image_path.lower().endswith('.tif'):
        # Load TIF file
        img_array = tiff.imread(image_path)
        # Convert to PIL Image
        if img_array.dtype == np.uint16:
            img_array = (img_array / 256).astype(np.uint8)
        img = Image.fromarray(img_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')

    # Create subplot - 3 rows to show more variations
    rows = 3
    cols = (num_augmentations + 1 + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle('EXTREME Augmentation Examples\n(Color, Rotation, Flip, Twist, Blur, Quality, Occlusion)',
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Show original
    axes[0].imshow(img)
    axes[0].set_title('ORIGINAL', fontsize=12, fontweight='bold', color='red')
    axes[0].axis('off')
    axes[0].set_facecolor('#f0f0f0')

    # Show augmented versions
    for i in range(1, num_augmentations + 1):
        if i < len(axes):
            augmented = transform(img)

            # Convert tensor to numpy for display
            if augmented.shape[0] == 1:  # Grayscale
                # Denormalize for display
                aug_img = augmented.squeeze().numpy()
                aug_img = (aug_img * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
                aug_img = np.clip(aug_img, 0, 1)
                axes[i].imshow(aug_img, cmap='gray', vmin=0, vmax=1)
            else:  # RGB
                aug_img = augmented.permute(1, 2, 0).numpy()
                aug_img = (aug_img * 0.5 + 0.5)  # Denormalize
                aug_img = np.clip(aug_img, 0, 1)
                axes[i].imshow(aug_img)

            axes[i].set_title(f'Aug #{i}', fontsize=10)
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_augmentations + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved augmentation examples to: augmentation_examples.png")
    plt.show()


def find_sample_image():
    """Find a sample TIF image from the training directory"""
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith('.tif')]
            if images:
                return os.path.join(class_dir, images[0])
    return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXTREME AUGMENTATION VISUALIZATION")
    print("="*70)

    # Try to find a sample image
    sample_image = find_sample_image()

    if sample_image:
        print(f"Using sample image: {sample_image}")
        print(f"\nGenerating augmentation examples with:")
        print(f"  ✓ Color: Dark (0.3x) to Bright (1.5x)")
        print(f"  ✓ Contrast: Low (0.5x) to High (1.8x)")
        print(f"  ✓ Saturation: 0.4x to 1.6x")
        print(f"  ✓ Hue shifts: ±20%")
        print(f"  ✓ Rotation: 0-360°")
        print(f"  ✓ Flipping: Horizontal & Vertical")
        print(f"  ✓ Perspective: Warping & Twisting")
        print(f"  ✓ Affine: Translation, Scale, Shear")
        print(f"  ✓ Blur: Gaussian blur variations")
        print(f"  ✓ Sharpness: Random adjustments")
        print(f"  ✓ Quality: Posterize, Solarize")
        print(f"  ✓ Occlusion: Random erasing")
        print(f"  ✓ Cropping: 60-100% with various ratios")
        print(f"\n  → Model will be EXTREMELY ROBUST! 🚀")
        print("="*70)
        visualize_augmentations(sample_image, num_augmentations=11)
    else:
        print(f"⚠ Could not find sample images in: {TRAIN_DIR}")
        print(f"Please provide a path to an image:")
        image_path = input("Image path: ").strip('"')
        if os.path.exists(image_path):
            visualize_augmentations(image_path, num_augmentations=11)
        else:
            print("❌ Image not found!")

    print("="*70 + "\n")

