"""
Visualize segmentation-only training with random grayscale variations
Shows how the model will see different grape colors as different grayscale intensities
"""
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile as tiff
import numpy as np
from data_transforms import get_train_transforms, get_test_transforms
from config import TRAIN_DIR

def visualize_segmentation_grayscale(image_path, num_augmentations=12):
    """
    Visualize segmentation with grayscale variations

    Args:
        image_path: Path to a TIF mask file
        num_augmentations: Number of augmented versions to show
    """
    transform = get_train_transforms()
    test_transform = get_test_transforms()

    # Load TIF mask
    with tiff.TiffFile(image_path) as tif:
        mask = tif.pages[0].asarray()
        tags = tif.pages[0].tags
        description = tags.get("ImageDescription")
        import json
        metadata = json.loads(description.value)

    # Load original image
    image_name = metadata.get("image_name")
    from config import IMAGES_DIR

    image_path_full = None
    for ext in [".png", ".jpg", ".jpeg"]:
        potential_path = os.path.join(IMAGES_DIR, image_name + ext)
        if os.path.exists(potential_path):
            image_path_full = potential_path
            break

    if image_path_full is None:
        print(f"Could not find image: {image_name}")
        return

    # Load and create segmentation
    image = Image.open(image_path_full).convert("RGB")
    original_bbox = metadata.get("original_bbox")
    bbox = list(map(int, original_bbox))

    # Crop mask and image
    mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    mask_crop = (mask_crop > 0).astype(np.uint8)
    cropped_image = image.crop(original_bbox)
    cropped_np = np.array(cropped_image)

    # Create segmentation overlay (only masked pixels, rest black)
    mask_crop_3 = np.stack([mask_crop, mask_crop, mask_crop], axis=-1)
    segmentation_overlay_np = cropped_np * mask_crop_3
    segmentation_overlay = Image.fromarray(segmentation_overlay_np)

    # Create subplot - 3 rows for more variations
    rows = 3
    cols = (num_augmentations + 2 + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle('SEGMENTATION MODE with Random Grayscale Variations\n' +
                 'Training on MASKED PIXELS ONLY (no bbox) in GRAYSCALE\n' +
                 'Robust to ALL grape colors (red, green, black)',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    # Show original segmentation in color
    axes[0].imshow(segmentation_overlay)
    axes[0].set_title('ORIGINAL\n(Color Segmentation)', fontsize=11, fontweight='bold', color='red')
    axes[0].axis('off')
    axes[0].set_facecolor('#f0f0f0')

    # Show test transform (no augmentation, just grayscale)
    test_result = test_transform(segmentation_overlay)
    test_img = test_result.squeeze().numpy()
    test_img = (test_img * 0.5 + 0.5)  # Denormalize
    test_img = np.clip(test_img, 0, 1)
    axes[1].imshow(test_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('TEST MODE\n(Grayscale, No Aug)', fontsize=11, fontweight='bold', color='green')
    axes[1].axis('off')
    axes[1].set_facecolor('#f0f0f0')

    # Show augmented versions with random gray variations
    for i in range(2, num_augmentations + 2):
        if i < len(axes):
            augmented = transform(segmentation_overlay)

            # Convert tensor to numpy for display
            aug_img = augmented.squeeze().numpy()
            # Denormalize from [-1, 1] to [0, 1]
            aug_img = (aug_img * 0.5 + 0.5)
            aug_img = np.clip(aug_img, 0, 1)

            axes[i].imshow(aug_img, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'Aug #{i-1}\n(Random Gray)', fontsize=10)
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_augmentations + 2, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_grayscale_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved segmentation grayscale examples to: segmentation_grayscale_examples.png")
    plt.show()


def find_sample_tif():
    """Find a sample TIF file from the training directory"""
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.isdir(class_dir):
            tif_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.tif')]
            if tif_files:
                return os.path.join(class_dir, tif_files[0])
    return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SEGMENTATION + GRAYSCALE VISUALIZATION")
    print("="*70)

    # Try to find a sample TIF
    sample_tif = find_sample_tif()

    if sample_tif:
        print(f"Using sample TIF: {sample_tif}")
        print(f"\nThis shows how the model will train:")
        print(f"  ✓ SEGMENTATION MODE: Only masked grape pixels (no bbox)")
        print(f"  ✓ GRAYSCALE: Removes color info (works for all grape colors)")
        print(f"  ✓ RANDOM GRAY VARIATIONS: Different intensities/contrasts")
        print(f"  ✓ Result: Robust to RED, GREEN, and BLACK grapes!")
        print(f"\nGenerating examples...")
        print("="*70)
        visualize_segmentation_grayscale(sample_tif, num_augmentations=10)
    else:
        print(f"⚠ Could not find sample TIF files in: {TRAIN_DIR}")
        print(f"Please provide a path to a TIF mask file:")
        tif_path = input("TIF path: ").strip('"')
        if os.path.exists(tif_path):
            visualize_segmentation_grayscale(tif_path, num_augmentations=10)
        else:
            print("❌ TIF file not found!")

    print("="*70 + "\n")

