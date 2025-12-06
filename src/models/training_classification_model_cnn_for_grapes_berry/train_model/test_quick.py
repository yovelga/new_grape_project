"""
Quick test to verify context_square_segmentation mode is working
"""
import torch
from dataset_multi import GrapeDataset
from data_transforms import get_train_transforms
from config import TRAIN_DIR, BATCH_SIZE
from torch.utils.data import DataLoader

print("\n" + "="*60)
print("TESTING CONTEXT-AWARE SQUARE CROP MODE")
print("="*60)

try:
    print("\n1. Loading dataset with context_square_segmentation mode...")
    train_dataset = GrapeDataset(
        root_dir=TRAIN_DIR,
        input_mode="context_square_segmentation",
        transform=None,  # No transform to see raw size
        balance_mode="oversample"
    )
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(train_dataset)}")

    print("\n2. Testing sample loading...")
    img, label = train_dataset[0]
    print(f"✅ Sample loaded successfully!")
    print(f"   Image size: {img.size}")
    print(f"   Is square: {img.size[0] == img.size[1]}")
    print(f"   Label: {'Grape' if label == 1 else 'Not Grape'}")

    print("\n3. Testing with transforms...")
    train_dataset_with_transforms = GrapeDataset(
        root_dir=TRAIN_DIR,
        input_mode="context_square_segmentation",
        transform=get_train_transforms(),
        balance_mode="oversample"
    )
    img_transformed, label = train_dataset_with_transforms[0]
    print(f"✅ Transform applied successfully!")
    print(f"   Transformed shape: {img_transformed.shape}")
    print(f"   Expected: torch.Size([1, 224, 224]) for grayscale")

    print("\n4. Testing DataLoader...")
    train_loader = DataLoader(
        train_dataset_with_transforms,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # 0 for testing
    )
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"✅ DataLoader working!")
    print(f"   Batch shape: {batch_imgs.shape}")
    print(f"   Labels shape: {batch_labels.shape}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - READY FOR TRAINING!")
    print("="*60)
    print("\nYou can now run: python main.py")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

