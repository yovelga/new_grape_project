"""
Unified model loading utilities.

Supports both sklearn/joblib models and PyTorch models.
"""

import joblib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """
    Load model from file (sklearn/joblib or PyTorch).

    Automatically detects format based on file extension and content.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model object (sklearn model or PyTorch state dict)

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file format is unsupported or corrupted
    """
    model_path = Path(model_path)

    # Check if file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check the path and ensure the file exists."
        )

    # Get file extension
    ext = model_path.suffix.lower()

    logger.info(f"Loading model from: {model_path.name}")
    logger.info(f"File extension: {ext}")

    # Try loading based on extension
    if ext in ['.joblib', '.pkl', '.pickle']:
        return _load_sklearn_model(model_path)
    elif ext in ['.pt', '.pth']:
        return _load_torch_model(model_path)
    else:
        # Try to auto-detect format
        logger.warning(f"Unknown extension '{ext}', attempting auto-detection...")

        # Try joblib first (most common for sklearn)
        try:
            return _load_sklearn_model(model_path)
        except Exception as e1:
            logger.debug(f"Failed to load as sklearn model: {e1}")

            # Try torch
            try:
                return _load_torch_model(model_path)
            except Exception as e2:
                logger.debug(f"Failed to load as torch model: {e2}")

                raise ValueError(
                    f"Failed to load model from {model_path}.\n"
                    f"Supported formats:\n"
                    f"  - sklearn/joblib: .joblib, .pkl, .pickle\n"
                    f"  - PyTorch: .pt, .pth\n"
                    f"Sklearn load error: {e1}\n"
                    f"PyTorch load error: {e2}"
                )


def _load_sklearn_model(model_path: Path) -> Any:
    """
    Load sklearn/joblib model.

    Args:
        model_path: Path to model file

    Returns:
        Loaded sklearn model or pipeline

    Raises:
        ValueError: If loading fails
    """
    try:
        # Try joblib first (recommended for sklearn)
        model = joblib.load(model_path)
        logger.info(f"✓ Loaded sklearn model (joblib): {type(model).__name__}")
        return model
    except Exception as e1:
        logger.debug(f"joblib.load failed: {e1}")

        # Try pickle as fallback
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Loaded sklearn model (pickle): {type(model).__name__}")
            return model
        except Exception as e2:
            raise ValueError(
                f"Failed to load sklearn model:\n"
                f"  joblib error: {e1}\n"
                f"  pickle error: {e2}"
            )


def _load_torch_model(model_path: Path) -> Dict[str, Any]:
    """
    Load PyTorch model checkpoint.

    Args:
        model_path: Path to model file

    Returns:
        Loaded checkpoint dictionary

    Raises:
        ImportError: If torch is not installed
        ValueError: If loading fails
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is not installed but model appears to be a .pt/.pth file.\n"
            "Install PyTorch: pip install torch torchvision\n"
            "Or use a sklearn model instead."
        )

    try:
        # Load checkpoint (always to CPU initially)
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(f"✓ Loaded PyTorch checkpoint")

        # Log checkpoint structure
        if isinstance(checkpoint, dict):
            logger.info(f"  Checkpoint keys: {list(checkpoint.keys())}")
        else:
            logger.info(f"  Checkpoint type: {type(checkpoint).__name__}")

        return checkpoint
    except Exception as e:
        raise ValueError(f"Failed to load PyTorch model: {e}")


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Extract metadata from loaded model.

    Args:
        model: Loaded model object

    Returns:
        Dictionary with model metadata
    """
    info = {
        'type': type(model).__name__,
        'module': type(model).__module__,
    }

    # Sklearn model attributes
    if hasattr(model, 'classes_'):
        info['classes'] = list(model.classes_)
        info['n_classes'] = len(model.classes_)

    if hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_

    # Pipeline detection
    if 'Pipeline' in type(model).__name__:
        info['is_pipeline'] = True
        if hasattr(model, 'steps'):
            info['pipeline_steps'] = [name for name, _ in model.steps]

    # PyTorch checkpoint
    if isinstance(model, dict):
        info['checkpoint_keys'] = list(model.keys())
        if 'epoch' in model:
            info['epoch'] = model['epoch']
        if 'accuracy' in model:
            info['accuracy'] = model['accuracy']

    return info


def find_scaler_file(model_path: str) -> Optional[Path]:
    """
    Find associated scaler file for a model.

    Looks for scaler files in same directory:
    - {model_name}_scaler.joblib
    - {model_name}_scaler.pkl
    - scaler.joblib
    - StandardScaler.joblib

    Args:
        model_path: Path to model file

    Returns:
        Path to scaler file if found, None otherwise
    """
    model_path = Path(model_path)
    model_dir = model_path.parent
    model_stem = model_path.stem

    # Try different scaler naming conventions
    scaler_candidates = [
        model_dir / f"{model_stem}_scaler.joblib",
        model_dir / f"{model_stem}_scaler.pkl",
        model_dir / "scaler.joblib",
        model_dir / "scaler.pkl",
        model_dir / "StandardScaler.joblib",
        model_dir / "StandardScaler.pkl",
    ]

    for candidate in scaler_candidates:
        if candidate.exists():
            logger.info(f"✓ Found scaler: {candidate.name}")
            return candidate

    logger.debug(f"No scaler file found for {model_path.name}")
    return None


def load_scaler(scaler_path: Union[str, Path]) -> Any:
    """
    Load scaler from file.

    Args:
        scaler_path: Path to scaler file

    Returns:
        Loaded scaler object
    """
    scaler_path = Path(scaler_path)

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"✓ Loaded scaler: {type(scaler).__name__}")
        return scaler
    except Exception as e:
        raise ValueError(f"Failed to load scaler: {e}")
