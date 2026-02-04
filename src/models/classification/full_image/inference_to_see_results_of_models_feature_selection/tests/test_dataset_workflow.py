"""
Complete dataset workflow test.

Tests the full dataset handling pipeline from InferenceSession.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from binary_class_inference_ui import InferenceSession
from app.utils.logging import logger
import pandas as pd


def create_test_csv_files():
    """Create test CSV files for demonstration."""
    import tempfile

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create train/val CSV
    trainval_data = {
        'grape_id': [f'train{i:03d}' for i in range(30)],
        'image_path': [f'./images/train_{i:03d}.hdr' for i in range(30)],
        'label': ['healthy'] * 15 + ['cracked'] * 15
    }
    trainval_df = pd.DataFrame(trainval_data)
    trainval_csv = temp_dir / 'trainval.csv'
    trainval_df.to_csv(trainval_csv, index=False)

    # Create test CSV
    test_data = {
        'grape_id': [f'test{i:03d}' for i in range(10)],
        'image_path': [f'./images/test_{i:03d}.hdr' for i in range(10)],
        'label': ['healthy'] * 5 + ['cracked'] * 5
    }
    test_df = pd.DataFrame(test_data)
    test_csv = temp_dir / 'test.csv'
    test_df.to_csv(test_csv, index=False)

    return str(trainval_csv), str(test_csv)


def test_dataset_loading():
    """Test dataset loading through InferenceSession."""
    logger.info("=" * 60)
    logger.info("Test: Dataset Loading Workflow")
    logger.info("=" * 60)

    # Create test CSV files
    logger.info("\nStep 1: Creating test CSV files...")
    trainval_csv, test_csv = create_test_csv_files()
    logger.info(f"  Created: {trainval_csv}")
    logger.info(f"  Created: {test_csv}")

    # Initialize session
    logger.info("\nStep 2: Initialize InferenceSession...")
    session = InferenceSession()

    # Load datasets
    logger.info("\nStep 3: Load datasets...")
    try:
        train_df, val_df, test_df = session.load_datasets(
            trainval_csv=trainval_csv,
            test_csv=test_csv,
            val_size=0.30,
            random_seed=42
        )

        logger.info("\n✓ Datasets loaded successfully!")

    except Exception as e:
        logger.error(f"\n✗ Failed to load datasets: {e}")
        return False

    # Get summary
    logger.info("\nStep 4: Get dataset summary...")
    summary = session.get_dataset_summary()

    if summary['loaded']:
        logger.info("\n✓ Dataset Summary:")
        logger.info(f"  Train: {summary['train']['count']} samples")
        logger.info(f"    Distribution: {summary['train']['distribution']}")
        logger.info(f"  Val:   {summary['val']['count']} samples")
        logger.info(f"    Distribution: {summary['val']['distribution']}")
        logger.info(f"  Test:  {summary['test']['count']} samples")
        logger.info(f"    Distribution: {summary['test']['distribution']}")
    else:
        logger.error("\n✗ Datasets not loaded")
        return False

    # Verify splits
    logger.info("\nStep 5: Verify splits...")
    total_trainval = summary['train']['count'] + summary['val']['count']
    expected_trainval = 30

    if total_trainval == expected_trainval:
        logger.info(f"  ✓ Train + Val = {total_trainval} (expected {expected_trainval})")
    else:
        logger.error(f"  ✗ Train + Val = {total_trainval} (expected {expected_trainval})")
        return False

    if summary['test']['count'] == 10:
        logger.info(f"  ✓ Test = 10 (expected 10)")
    else:
        logger.error(f"  ✗ Test count mismatch")
        return False

    # Check stratification
    logger.info("\nStep 6: Check stratification...")
    train_dist = summary['train']['distribution']
    val_dist = summary['val']['distribution']

    # Should have both classes in train and val
    if len(train_dist) == 2 and len(val_dist) == 2:
        logger.info("  ✓ Both classes present in train and val")
    else:
        logger.error("  ✗ Stratification failed")
        return False

    return True


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Complete Dataset Workflow Test")
    logger.info("=" * 60)

    success = test_dataset_loading()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 All dataset workflow tests PASSED!")
    else:
        logger.error("❌ Some tests FAILED")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
