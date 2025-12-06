"""
Visualize Context-Aware Square Crops

This script compares different cropping modes side-by-side to show:
1. Original tight crop
2. Enlarged padded crop
3. Segmentation crop (masked)
4. Context-aware square crop (NEW)
5. Context-aware square + segmentation (NEW)

Usage:
    python visualize_context_crops.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from dataset_multi import GrapeDataset
from config import TRAIN_DIR

def visualize_single_sample(sample_idx=0, class_name="Grape"):
    """
    Visualize a single sample across all cropping modes.

    Args:
        sample_idx: Index of sample to visualize
        class_name: "Grape" or "Not Grape"
    """
    class_dir = os.path.join(TRAIN_DIR, class_name)

    # All available modes
    modes = [
        "original",
        "enlarged",
        "segmentation",
        "context_square",
        "context_square_segmentation"
    ]

    mode_descriptions = {
        "original": "Tight Crop\n(Original BBox)",
        "enlarged": "Enlarged Crop\n(Padded BBox)",
        "segmentation": "Segmentation\n(Masked Pixels)",
        "context_square": "Context Square\n(40% Padding)",
        "context_square_segmentation": "Context Square + Mask\n(Square + Segmentation)"
    }

    # Create figure
    fig, axes = plt.subplots(1, len(modes), figsize=(20, 4))
    fig.suptitle(f"Comparison of Cropping Modes - {class_name} Sample #{sample_idx}",
                 fontsize=16, fontweight='bold')

    # Load and display each mode
    for i, mode in enumerate(modes):
        try:
            dataset = GrapeDataset(
                root_dir=class_dir,
                input_mode=mode,
                transform=None  # No transform to see raw crops
            )

            if sample_idx >= len(dataset):
                print(f"Warning: Sample {sample_idx} not found in dataset (size: {len(dataset)})")
                sample_idx = 0

            img, label = dataset[sample_idx]

            # Convert PIL Image to numpy array
            img_np = np.array(img)

            # Display
            axes[i].imshow(img_np)
            axes[i].set_title(f"{mode_descriptions[mode]}\nSize: {img.size}", fontsize=10)
            axes[i].axis('off')

            # Add border to highlight square crops
            if "context_square" in mode:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)

        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading\n{mode}:\n{str(e)}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"crop_comparison_{class_name.replace(' ', '_')}_sample_{sample_idx}.png", dpi=150)
    print(f"Saved: crop_comparison_{class_name.replace(' ', '_')}_sample_{sample_idx}.png")
    plt.show()


def visualize_multiple_samples(num_samples=6, mode="context_square"):
    """
    Visualize multiple samples from the same mode.

    Args:
        num_samples: Number of samples to display
        mode: Input mode to visualize
    """
    class_dir = os.path.join(TRAIN_DIR, "Grape")

    dataset = GrapeDataset(
        root_dir=class_dir,
        input_mode=mode,
        transform=None
    )

    # Calculate grid size
    cols = 3
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f"Multiple Samples - Mode: {mode}", fontsize=16, fontweight='bold')

    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        if i < len(dataset):
            img, label = dataset[i]
            img_np = np.array(img)

            axes[i].imshow(img_np)
            axes[i].set_title(f"Sample {i}\nSize: {img.size}", fontsize=10)
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"multiple_samples_{mode}.png", dpi=150)
    print(f"Saved: multiple_samples_{mode}.png")
    plt.show()


def visualize_aspect_ratios():
    """
    Compare aspect ratios across different modes.
    Shows that context_square modes produce perfect squares.
    """
    modes = ["original", "enlarged", "segmentation", "context_square", "context_square_segmentation"]
    class_dir = os.path.join(TRAIN_DIR, "Grape")

    aspect_ratios = {mode: [] for mode in modes}
    sizes = {mode: [] for mode in modes}

    # Collect aspect ratios from first 20 samples
    for mode in modes:
        try:
            dataset = GrapeDataset(root_dir=class_dir, input_mode=mode, transform=None)
            for i in range(min(20, len(dataset))):
                img, _ = dataset[i]
                w, h = img.size
                aspect_ratios[mode].append(w / h)
                sizes[mode].append((w, h))
        except Exception as e:
            print(f"Error loading mode {mode}: {e}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Aspect ratio distribution
    ax1.boxplot([aspect_ratios[mode] for mode in modes], labels=modes)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Perfect Square (1:1)')
    ax1.set_ylabel('Aspect Ratio (Width/Height)', fontsize=12)
    ax1.set_title('Aspect Ratio Distribution by Mode', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Average crop sizes
    avg_widths = [np.mean([s[0] for s in sizes[mode]]) if sizes[mode] else 0 for mode in modes]
    avg_heights = [np.mean([s[1] for s in sizes[mode]]) if sizes[mode] else 0 for mode in modes]

    x = np.arange(len(modes))
    width = 0.35

    ax2.bar(x - width/2, avg_widths, width, label='Avg Width', alpha=0.8)
    ax2.bar(x + width/2, avg_heights, width, label='Avg Height', alpha=0.8)
    ax2.set_ylabel('Pixels', fontsize=12)
    ax2.set_title('Average Crop Dimensions by Mode', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("aspect_ratio_analysis.png", dpi=150)
    print("Saved: aspect_ratio_analysis.png")
    plt.show()


def visualize_context_expansion():
    """
    Show how context expansion works by overlaying tight bbox and expanded bbox.
    """
    from PIL import ImageDraw

    class_dir = os.path.join(TRAIN_DIR, "Grape")

    # Get original and context_square crops
    dataset_original = GrapeDataset(root_dir=class_dir, input_mode="original", transform=None)
    dataset_context = GrapeDataset(root_dir=class_dir, input_mode="context_square", transform=None)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Context Expansion Visualization (Tight vs Expanded)", fontsize=16, fontweight='bold')

    for i in range(4):
        # Original tight crop
        img_tight, _ = dataset_original[i]
        axes[0, i].imshow(img_tight)
        axes[0, i].set_title(f"Sample {i}\nTight Crop: {img_tight.size}", fontsize=10)
        axes[0, i].axis('off')

        # Context-aware square crop
        img_context, _ = dataset_context[i]
        axes[1, i].imshow(img_context)
        axes[1, i].set_title(f"Sample {i}\nContext Square: {img_context.size}", fontsize=10)
        axes[1, i].axis('off')

        # Add green border to context crops
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig("context_expansion_comparison.png", dpi=150)
    print("Saved: context_expansion_comparison.png")
    plt.show()


def main():
    """
    Main function to run all visualizations.
    """
    print("=" * 60)
    print("Context-Aware Square Crop Visualization")
    print("=" * 60)

    print("\n1. Visualizing single sample across all modes...")
    visualize_single_sample(sample_idx=0, class_name="Grape")

    print("\n2. Visualizing multiple samples (context_square mode)...")
    visualize_multiple_samples(num_samples=6, mode="context_square")

    print("\n3. Analyzing aspect ratios...")
    visualize_aspect_ratios()

    print("\n4. Visualizing context expansion...")
    visualize_context_expansion()

    print("\n" + "=" * 60)
    print("✅ All visualizations complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - crop_comparison_Grape_sample_0.png")
    print("  - multiple_samples_context_square.png")
    print("  - aspect_ratio_analysis.png")
    print("  - context_expansion_comparison.png")


if __name__ == "__main__":
    main()

