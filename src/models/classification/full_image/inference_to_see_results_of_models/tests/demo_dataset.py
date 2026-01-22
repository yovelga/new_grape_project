"""
Dataset handling demo script.

Demonstrates loading and splitting datasets from CSV files.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.settings import settings
from app.data.dataset import (
    load_dataset_csv, split_train_val, load_and_prepare_splits,
    get_class_distribution, validate_image_paths
)
from app.utils.logging import logger


def demo_load_csv():
    """Demo CSV loading."""
    logger.info("=" * 60)
    logger.info("Demo: Load CSV Dataset")
    logger.info("=" * 60)

    logger.info(f"Default TrainVal CSV: {settings.default_trainval_csv}")
    logger.info(f"Default Test CSV: {settings.default_test_csv}")
    logger.info(f"Val split size: {settings.val_split_size}")
    logger.info(f"Random seed: {settings.random_seed}")

    logger.info("\nNote: To test CSV loading, create CSV files with columns:")
    logger.info("  - grape_id (unique ID)")
    logger.info("  - image_path (path to image)")
    logger.info("  - label (class label)")
    logger.info("\nExample CSV:")
    logger.info("  grape_id,image_path,label")
    logger.info("  001,./images/grape_001.hdr,healthy")
    logger.info("  002,./images/grape_002.hdr,cracked")


def demo_split_validation():
    """Demo validation of split requirements."""
    import pandas as pd

    logger.info("\n" + "=" * 60)
    logger.info("Demo: Split Validation")
    logger.info("=" * 60)

    # Create mock dataframe
    data = {
        'grape_id': [f'g{i:03d}' for i in range(20)],
        'image_path': [f'./images/grape_{i:03d}.hdr' for i in range(20)],
        'label': ['healthy'] * 10 + ['cracked'] * 10
    }
    df = pd.DataFrame(data)

    logger.info(f"Mock dataset: {len(df)} samples")
    logger.info(f"Distribution: {get_class_distribution(df)}")

    try:
        train_df, val_df = split_train_val(df, val_size=0.30, random_state=42)

        logger.info("\n✓ Split successful!")
        logger.info(f"Train: {len(train_df)} samples - {get_class_distribution(train_df)}")
        logger.info(f"Val:   {len(val_df)} samples - {get_class_distribution(val_df)}")

    except ValueError as e:
        logger.error(f"\n✗ Split failed: {e}")


def demo_rare_class_error():
    """Demo error handling for rare classes."""
    import pandas as pd

    logger.info("\n" + "=" * 60)
    logger.info("Demo: Rare Class Error Handling")
    logger.info("=" * 60)

    # Create mock dataframe with rare class
    data = {
        'grape_id': [f'g{i:03d}' for i in range(12)],
        'image_path': [f'./images/grape_{i:03d}.hdr' for i in range(12)],
        'label': ['healthy'] * 10 + ['rare_disease'] * 2  # Only 2 samples of rare_disease
    }
    df = pd.DataFrame(data)

    logger.info(f"Mock dataset: {len(df)} samples")
    logger.info(f"Distribution: {get_class_distribution(df)}")

    try:
        train_df, val_df = split_train_val(df, val_size=0.30, random_state=42)
        logger.info("\n✓ Split successful (unexpected)")

    except ValueError as e:
        logger.info("\n✓ Expected error caught:")
        logger.info(f"  {str(e)[:200]}...")


def main():
    """Main demo function."""
    logger.info("=" * 60)
    logger.info("Dataset Handling Demo")
    logger.info("=" * 60)

    # Demo 1: Show configuration
    demo_load_csv()

    # Demo 2: Show successful split
    demo_split_validation()

    # Demo 3: Show rare class error
    demo_rare_class_error()

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
