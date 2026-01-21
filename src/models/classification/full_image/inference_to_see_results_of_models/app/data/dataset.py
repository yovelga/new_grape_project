"""
Dataset classes for loading and preprocessing images.

Handles CSV-based dataset loading with train/val/test splits.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ===== CSV Dataset Loading =====

def load_dataset_csv(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Required columns:
    - grape_id: Unique identifier for each sample
    - image_path: Path to image file
    - label: Class label

    Optional columns: Any additional metadata

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with validated columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate required columns
    required_cols = ['grape_id', 'image_path', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Required columns are: {required_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError(f"CSV file is empty: {path}")

    # Log info
    logger.info(f"Loaded dataset from {csv_path.name}:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Log class distribution
    label_counts = df['label'].value_counts().to_dict()
    logger.info(f"  Class distribution: {label_counts}")

    return df


def split_train_val(
    trainval_df: pd.DataFrame,
    val_size: float = 0.30,
    random_state: int = 42,
    label_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train/val dataframe into train and validation sets.

    Uses stratified split to maintain class distribution.

    Args:
        trainval_df: Combined train+val dataframe
        val_size: Fraction of data for validation (0.0-1.0)
        random_state: Random seed for reproducibility
        label_col: Name of label column

    Returns:
        Tuple of (train_df, val_df)

    Raises:
        ValueError: If stratified split fails due to rare classes
    """
    if val_size <= 0 or val_size >= 1:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    if len(trainval_df) == 0:
        raise ValueError("Empty dataframe provided")

    if label_col not in trainval_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    # Check class distribution
    label_counts = trainval_df[label_col].value_counts()
    min_count = label_counts.min()

    # Calculate minimum samples needed for stratified split
    min_samples_needed = int(1 / val_size) + 1  # e.g., val_size=0.3 needs at least 4 samples

    if min_count < min_samples_needed:
        rare_classes = label_counts[label_counts < min_samples_needed].index.tolist()
        raise ValueError(
            f"Cannot perform stratified split: Some classes have too few samples.\n"
            f"Classes with insufficient samples: {rare_classes}\n"
            f"Minimum samples needed per class: {min_samples_needed} (for val_size={val_size})\n"
            f"Current distribution:\n{label_counts.to_dict()}\n"
            f"Solution: Either increase samples for rare classes or reduce val_size."
        )

    try:
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_size,
            random_state=random_state,
            stratify=trainval_df[label_col]
        )

        logger.info("Train/Val split completed:")
        logger.info(f"  Train samples: {len(train_df)}")
        logger.info(f"  Train distribution: {train_df[label_col].value_counts().to_dict()}")
        logger.info(f"  Val samples: {len(val_df)}")
        logger.info(f"  Val distribution: {val_df[label_col].value_counts().to_dict()}")

        return train_df, val_df

    except ImportError:
        raise ImportError("scikit-learn is required for train/val split. Install: pip install scikit-learn")
    except Exception as e:
        raise ValueError(f"Failed to split train/val: {e}")


def load_and_prepare_splits(
    trainval_csv_path: str,
    test_csv_path: str,
    val_size: float = 0.30,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare train/val/test splits from CSV files.

    Args:
        trainval_csv_path: Path to train+val CSV file
        test_csv_path: Path to test CSV file
        val_size: Fraction of trainval to use for validation
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load CSVs
    logger.info("Loading datasets...")
    trainval_df = load_dataset_csv(trainval_csv_path)
    test_df = load_dataset_csv(test_csv_path)

    # Split train/val
    logger.info("Splitting train/val...")
    train_df, val_df = split_train_val(trainval_df, val_size, random_state)

    # Summary
    logger.info("=" * 60)
    logger.info("Dataset splits prepared successfully:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    logger.info("=" * 60)

    return train_df, val_df, test_df


def get_class_distribution(df: pd.DataFrame, label_col: str = "label") -> Dict[str, int]:
    """
    Get class distribution for a dataframe.

    Args:
        df: DataFrame with label column
        label_col: Name of label column

    Returns:
        Dictionary mapping class labels to counts
    """
    if label_col not in df.columns:
        return {}

    return df[label_col].value_counts().to_dict()


def validate_image_paths(df: pd.DataFrame, base_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Validate that image paths in dataframe exist.

    Args:
        df: DataFrame with image_path column
        base_dir: Optional base directory for relative paths

    Returns:
        Dictionary with 'missing' and 'found' lists
    """
    if 'image_path' not in df.columns:
        raise ValueError("DataFrame missing 'image_path' column")

    missing = []
    found = []

    for idx, row in df.iterrows():
        img_path = Path(row['image_path'])

        # Try absolute path first
        if not img_path.is_absolute() and base_dir is not None:
            img_path = base_dir / img_path

        if img_path.exists():
            found.append(str(row['image_path']))
        else:
            missing.append(str(row['image_path']))

    logger.info(f"Image path validation: {len(found)} found, {len(missing)} missing")

    return {
        'found': found,
        'missing': missing
    }


# ===== Inference Dataset Classes =====

class InferenceDataset:
    """Base dataset for inference."""

    def __init__(self, image_paths: list, transform: Optional[Callable] = None):
        """
        Initialize dataset.

        Args:
            image_paths: List of paths to images
            transform: Optional transform function
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Get item at index.

        Args:
            idx: Index

        Returns:
            Tuple of (image, path)
        """
        path = self.image_paths[idx]
        image = self._load_image(path)

        if self.transform:
            image = self.transform(image)

        return image, path

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_image")


class RGBDataset(InferenceDataset):
    """Dataset for RGB images."""

    def _load_image(self, path: str) -> np.ndarray:
        """Load RGB image."""
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return np.array(img)


class SpectralDataset(InferenceDataset):
    """Dataset for hyperspectral/multispectral images."""

    def _load_image(self, path: str) -> np.ndarray:
        """Load spectral image."""
        # Implementation depends on format (ENVI, etc.)
        # This is a placeholder - actual implementation in io modules
        raise NotImplementedError("Use ENVI loader for spectral data")
