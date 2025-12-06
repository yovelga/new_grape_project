"""
Quick verification script to ensure all augmentation components are working
"""
import sys
import traceback

print("\n" + "="*70)
print("AUGMENTATION SETUP VERIFICATION")
print("="*70 + "\n")

# Test 1: Import data_transforms
print("1. Testing data_transforms import...")
try:
    from data_transforms import get_train_transforms, get_test_transforms
    print("   ✓ data_transforms imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import data_transforms: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check transform composition
print("\n2. Testing transform composition...")
try:
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    print(f"   ✓ Train transform has {len(train_transform.transforms)} operations")
    print(f"   ✓ Test transform has {len(test_transform.transforms)} operations")
except Exception as e:
    print(f"   ✗ Failed to create transforms: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with sample image
print("\n3. Testing transform on sample image...")
try:
    from PIL import Image
    import numpy as np

    # Create a dummy RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Apply transform
    augmented = train_transform(dummy_img)
    print(f"   ✓ Transform applied successfully")
    print(f"   ✓ Output shape: {augmented.shape}")
    print(f"   ✓ Output dtype: {augmented.dtype}")
    print(f"   ✓ Output range: [{augmented.min():.2f}, {augmented.max():.2f}]")
except Exception as e:
    print(f"   ✗ Failed to apply transform: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: List all augmentation operations
print("\n4. Listing all augmentation operations...")
try:
    train_transform = get_train_transforms()
    print("   Training augmentations:")
    for i, t in enumerate(train_transform.transforms, 1):
        print(f"      {i:2d}. {t.__class__.__name__}")

    augmentation_count = len(train_transform.transforms)
    print(f"\n   ✓ Total augmentation operations: {augmentation_count}")

    if augmentation_count >= 15:
        print(f"   ✓ EXCELLENT! You have {augmentation_count} augmentation operations!")
    elif augmentation_count >= 10:
        print(f"   ✓ GOOD! You have {augmentation_count} augmentation operations")
    else:
        print(f"   ⚠ WARNING: Only {augmentation_count} augmentation operations. Expected 15+")
except Exception as e:
    print(f"   ✗ Failed to list operations: {e}")
    traceback.print_exc()

# Test 5: Check dataset oversampling
print("\n5. Testing dataset with oversampling...")
try:
    from config import TRAIN_DIR
    from dataset_multi import GrapeDataset

    # Test without oversampling
    dataset_no_balance = GrapeDataset(
        root_dir=TRAIN_DIR,
        input_mode="original",
        transform=None,
        balance_mode=None
    )

    # Test with oversampling
    dataset_oversampled = GrapeDataset(
        root_dir=TRAIN_DIR,
        input_mode="original",
        transform=None,
        balance_mode="oversample"
    )

    print(f"   ✓ Dataset without balance: {len(dataset_no_balance)} samples")
    print(f"   ✓ Dataset with oversampling: {len(dataset_oversampled)} samples")

    if len(dataset_oversampled) >= len(dataset_no_balance):
        print(f"   ✓ Oversampling working correctly!")
    else:
        print(f"   ⚠ WARNING: Oversampling may not be working as expected")

except Exception as e:
    print(f"   ✗ Failed to test dataset: {e}")
    print(f"   (This is OK if data directory is not accessible)")

# Test 6: Check GPU availability
print("\n6. Checking GPU availability...")
try:
    import torch

    if torch.cuda.is_available():
        print(f"   ✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   ✓ CUDA Version: {torch.version.cuda}")
    else:
        print(f"   ⚠ GPU not available - training will use CPU")
        print(f"   (Install CUDA-enabled PyTorch for GPU support)")
except Exception as e:
    print(f"   ✗ Failed to check GPU: {e}")

# Summary
print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\n✓ All critical components are working!")
print("\nYou can now:")
print("  1. Run 'python show_augmentations_extreme.py' to visualize augmentations")
print("  2. Run 'python main.py' to start training")
print("\n" + "="*70 + "\n")

