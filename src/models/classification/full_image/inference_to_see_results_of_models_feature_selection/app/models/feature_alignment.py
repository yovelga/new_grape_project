"""
Feature Alignment for Reduced Model Inference.

This module provides a robust feature contract system to ensure that:
- Full-feature models receive all spectral bands (e.g., 159 features)
- Reduced-feature models (BFS subsets) receive only the required bands in the exact training order

The feature_names.json file serves as the "contract" specifying which features a reduced model expects.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import re

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Name Normalization
# ============================================================================

def normalize_feature_name(name: str) -> str:
    """
    Normalize feature name for consistent matching.
    
    Handles variations like:
    - "452.25nm" -> "452.25nm"
    - "452.25 nm" -> "452.25nm"
    - "452.250nm" -> "452.25nm" (trailing zeros)
    - 452.25 (float) -> "452.25nm"
    
    Args:
        name: Feature name (string or wavelength value)
        
    Returns:
        Normalized feature name in format "XXX.XXnm"
    """
    name = str(name).strip()
    
    # Remove any whitespace before 'nm'
    name = re.sub(r'\s*nm$', 'nm', name, flags=re.IGNORECASE)
    
    # If no 'nm' suffix, assume it's a wavelength value and add 'nm'
    if not name.lower().endswith('nm'):
        # Try to parse as float and format consistently
        try:
            wl = float(name)
            name = f"{wl:.2f}nm"
        except ValueError:
            pass  # Not a numeric value, keep as-is
    
    # Remove trailing zeros after decimal point but keep at least 2 decimal places
    match = re.match(r'^(\d+)\.(\d+)nm$', name, re.IGNORECASE)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        # Keep at least 2 decimal places
        decimal_part = decimal_part.rstrip('0')
        if len(decimal_part) < 2:
            decimal_part = decimal_part.ljust(2, '0')
        name = f"{integer_part}.{decimal_part}nm"
    
    return name


def wavelengths_to_feature_names(wavelengths: np.ndarray) -> List[str]:
    """
    Convert wavelength array to normalized feature names.
    
    Args:
        wavelengths: Array of wavelength values (e.g., [452.25, 536.82, ...])
        
    Returns:
        List of normalized feature names (e.g., ["452.25nm", "536.82nm", ...])
    """
    return [normalize_feature_name(f"{wl:.2f}") for wl in wavelengths]


# ============================================================================
# Reduced Model Package Resolution
# ============================================================================

@dataclass
class ReducedModelPackageInfo:
    """
    Information about a reduced-feature model package.
    
    A reduced model package is a folder containing:
    - model file: *.pkl or *.joblib
    - feature_names.json: List of required feature names in training order
    - Optionally: scaler, metrics, etc.
    """
    model_path: Path
    feature_names_path: Path
    feature_names: List[str]
    package_root: Path
    balance_type: Optional[str] = None  # "Balanced" or "Unbalanced"
    timestamp: Optional[str] = None


def find_latest_run_folder(package_root: Path) -> Optional[Path]:
    """
    Find the latest timestamp run folder in a reduced model package.
    
    Looks for folders matching YYYY-MM-DD_HH-MM-SS format.
    
    Args:
        package_root: Root of the reduced model package
        
    Returns:
        Path to latest run folder, or None if no timestamped folders found
    """
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    
    timestamp_folders = []
    for item in package_root.iterdir():
        if item.is_dir() and timestamp_pattern.match(item.name):
            timestamp_folders.append(item)
    
    if not timestamp_folders:
        return None
    
    # Sort by name (ISO format sorts chronologically)
    timestamp_folders.sort(key=lambda x: x.name, reverse=True)
    return timestamp_folders[0]


def find_model_and_features(search_root: Path, prefer_balanced: bool = True) -> Optional[Tuple[Path, Path]]:
    """
    Recursively find model file and feature_names.json within a directory.
    
    Handles nested structure like:
        package_root/
            2026-02-02_22-17-18/
                multiclass/
                    Balanced/
                        xgboost_model.pkl
                        feature_names.json
    
    Args:
        search_root: Directory to search
        prefer_balanced: If True, prefer Balanced over Unbalanced when both exist
        
    Returns:
        Tuple of (model_path, feature_names_path) or None if not found
    """
    # Direct check at current level
    model_file = None
    feature_file = None
    
    # Look for model file
    for pattern in ['*.pkl', '*.joblib']:
        files = list(search_root.glob(pattern))
        if files:
            model_file = files[0]
            break
    
    # Look for feature_names.json
    feature_path = search_root / 'feature_names.json'
    if feature_path.exists():
        feature_file = feature_path
    
    # If both found at this level, return them
    if model_file and feature_file:
        return (model_file, feature_file)
    
    # Check for Balanced/Unbalanced subdirs
    balanced_dir = search_root / 'Balanced'
    unbalanced_dir = search_root / 'Unbalanced'
    
    if prefer_balanced and balanced_dir.exists():
        result = find_model_and_features(balanced_dir, prefer_balanced)
        if result:
            return result
    
    if unbalanced_dir.exists():
        result = find_model_and_features(unbalanced_dir, prefer_balanced)
        if result:
            return result
    
    if not prefer_balanced and balanced_dir.exists():
        result = find_model_and_features(balanced_dir, prefer_balanced)
        if result:
            return result
    
    # Check other subdirs (like 'multiclass')
    for subdir in search_root.iterdir():
        if subdir.is_dir() and subdir.name not in ['Balanced', 'Unbalanced']:
            result = find_model_and_features(subdir, prefer_balanced)
            if result:
                return result
    
    return None


def resolve_reduced_package(package_root: Path, prefer_balanced: bool = True) -> ReducedModelPackageInfo:
    """
    Resolve a reduced model package folder to find model and feature files.
    
    Handles various nested structures:
    - Direct: package_root/model.pkl + feature_names.json
    - Timestamped: package_root/YYYY-MM-DD_HH-MM-SS/multiclass/Balanced/...
    
    Args:
        package_root: Root folder of the reduced model package
        prefer_balanced: If True, prefer Balanced models when both exist
        
    Returns:
        ReducedModelPackageInfo with resolved paths
        
    Raises:
        ValueError: If package structure is invalid (missing model or feature_names.json)
    """
    if not package_root.is_dir():
        raise ValueError(f"Package root is not a directory: {package_root}")
    
    # First, check if there are timestamped run folders
    latest_run = find_latest_run_folder(package_root)
    
    if latest_run:
        search_root = latest_run
        timestamp = latest_run.name
        logger.info(f"Found timestamp folder: {timestamp}")
    else:
        search_root = package_root
        timestamp = None
    
    # Find model and feature files
    result = find_model_and_features(search_root, prefer_balanced)
    
    if result is None:
        raise ValueError(
            f"Invalid reduced model package: {package_root}\n"
            f"Could not find both model file (*.pkl/*.joblib) and feature_names.json\n"
            f"Expected structure: package_root/[timestamp/][multiclass/][Balanced|Unbalanced/]model.pkl + feature_names.json"
        )
    
    model_path, feature_names_path = result
    
    # Determine balance type from path
    balance_type = None
    for part in feature_names_path.parts:
        if part in ['Balanced', 'Unbalanced']:
            balance_type = part
            break
    
    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    if not isinstance(feature_names, list):
        raise ValueError(
            f"feature_names.json must contain a list of feature names, "
            f"got {type(feature_names).__name__}"
        )
    
    # Normalize feature names
    feature_names = [normalize_feature_name(name) for name in feature_names]
    
    logger.info(
        f"Resolved reduced model package:\n"
        f"  Package root: {package_root}\n"
        f"  Model file: {model_path}\n"
        f"  Feature names: {feature_names_path}\n"
        f"  Balance type: {balance_type or 'Unknown'}\n"
        f"  Required features: {len(feature_names)}"
    )
    
    return ReducedModelPackageInfo(
        model_path=model_path,
        feature_names_path=feature_names_path,
        feature_names=feature_names,
        package_root=package_root,
        balance_type=balance_type,
        timestamp=timestamp
    )


def is_reduced_model_package(path: Path) -> bool:
    """
    Check if a path is a valid reduced model package folder.
    
    A valid package must contain (directly or nested):
    - A model file (*.pkl or *.joblib)
    - feature_names.json
    
    Args:
        path: Path to check
        
    Returns:
        True if valid reduced model package, False otherwise
    """
    if not path.is_dir():
        return False
    
    try:
        resolve_reduced_package(path)
        return True
    except ValueError:
        return False


# ============================================================================
# Feature Alignment for Inference
# ============================================================================

def align_features_for_model(
    X_full: np.ndarray,
    full_feature_names: List[str],
    required_feature_names: Optional[List[str]]
) -> np.ndarray:
    """
    Align features for model inference by selecting and ordering required features.
    
    This is the CRITICAL function that ensures reduced models receive only their
    required features in the exact training order.
    
    Args:
        X_full: Full feature matrix (N, D_full) where D_full is all available features
        full_feature_names: List of feature names for X_full columns (e.g., wavelengths)
        required_feature_names: List of required feature names in training order.
            If None, returns X_full unchanged (full-feature model).
            
    Returns:
        X_aligned: Aligned feature matrix (N, D_required)
        - If required_feature_names is None: returns X_full unchanged
        - Otherwise: returns X_full sliced to required features in correct order
        
    Raises:
        ValueError: If any required feature is not found in full_feature_names
        
    Example:
        >>> X_full = np.random.rand(100, 159)  # 100 samples, 159 features
        >>> full_names = ["400.00nm", "402.50nm", ..., "1000.00nm"]
        >>> required_names = ["452.25nm", "536.82nm", "892.95nm"]  # 3 features
        >>> X_aligned = align_features_for_model(X_full, full_names, required_names)
        >>> assert X_aligned.shape == (100, 3)
    """
    # Full-feature model: no alignment needed
    if required_feature_names is None:
        logger.debug(f"Full-feature model: using all {X_full.shape[1]} features")
        return X_full
    
    # Validate inputs
    if X_full.ndim != 2:
        raise ValueError(f"X_full must be 2D array (N, D), got shape {X_full.shape}")
    
    if len(full_feature_names) != X_full.shape[1]:
        raise ValueError(
            f"full_feature_names length ({len(full_feature_names)}) must match "
            f"X_full columns ({X_full.shape[1]})"
        )
    
    # Normalize all feature names for matching
    full_names_normalized = [normalize_feature_name(n) for n in full_feature_names]
    required_names_normalized = [normalize_feature_name(n) for n in required_feature_names]
    
    # Build name-to-index mapping
    name_to_idx = {name: idx for idx, name in enumerate(full_names_normalized)}
    
    # Find indices for required features
    required_indices = []
    missing_features = []
    
    for req_name in required_names_normalized:
        if req_name in name_to_idx:
            required_indices.append(name_to_idx[req_name])
        else:
            missing_features.append(req_name)
    
    # Check for missing features
    if missing_features:
        # Log available features for debugging
        logger.error(
            f"Missing required features:\n"
            f"  Missing: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}\n"
            f"  Available (first 10): {full_names_normalized[:10]}\n"
            f"  Available (last 10): {full_names_normalized[-10:]}\n"
            f"  Total available: {len(full_names_normalized)}"
        )
        raise ValueError(
            f"Cannot align features: {len(missing_features)} required feature(s) not found.\n"
            f"Missing features: {missing_features}\n"
            f"This may indicate a mismatch between the model's training data and inference data."
        )
    
    # Extract aligned features in required order (CRITICAL: order must match training)
    required_indices = np.array(required_indices)
    X_aligned = X_full[:, required_indices]
    
    # Validate output shape
    expected_features = len(required_feature_names)
    if X_aligned.shape[1] != expected_features:
        raise ValueError(
            f"Feature alignment failed: expected {expected_features} features, "
            f"got {X_aligned.shape[1]}"
        )
    
    logger.info(
        f"Feature alignment complete:\n"
        f"  Input: {X_full.shape[1]} features -> Output: {X_aligned.shape[1]} features\n"
        f"  Required features: {required_names_normalized[:5]}...{required_names_normalized[-2:] if len(required_names_normalized) > 5 else ''}"
    )
    
    return X_aligned


def log_feature_alignment_summary(
    full_feature_count: int,
    required_feature_names: Optional[List[str]],
    model_name: str = "model"
) -> None:
    """
    Log a summary of feature alignment configuration for sanity checks.
    
    Args:
        full_feature_count: Number of available features from the HSI cube
        required_feature_names: List of required features (None for full-feature models)
        model_name: Name of the model for logging
    """
    if required_feature_names is None:
        logger.info(
            f"[{model_name}] FULL-FEATURE MODEL\n"
            f"  Expected features: {full_feature_count}"
        )
    else:
        reduced_count = len(required_feature_names)
        first_5 = required_feature_names[:5]
        last_2 = required_feature_names[-2:] if reduced_count > 5 else []
        
        logger.info(
            f"[{model_name}] REDUCED-FEATURE MODEL\n"
            f"  Full feature count: {full_feature_count}\n"
            f"  Required feature count: {reduced_count}\n"
            f"  First 5 required: {first_5}\n"
            f"  Last 2 required: {last_2}"
        )


# ============================================================================
# Package Detection Helpers (for UI)
# ============================================================================

def list_reduced_packages(models_dir: Path) -> List[Tuple[str, Path]]:
    """
    List all valid reduced model packages in a directory.
    
    Looks for subfolders that are valid reduced model packages.
    
    Args:
        models_dir: Directory containing model packages
        
    Returns:
        List of (display_name, path) tuples for each valid package
    """
    packages = []
    
    if not models_dir.exists():
        return packages
    
    for item in models_dir.iterdir():
        if item.is_dir():
            try:
                pkg_info = resolve_reduced_package(item)
                # Create display name
                display_name = item.name
                if pkg_info.balance_type:
                    display_name += f" ({pkg_info.balance_type})"
                display_name += f" [reduced: {len(pkg_info.feature_names)} features]"
                packages.append((display_name, item))
            except ValueError:
                # Not a valid reduced package, skip
                pass
    
    return packages
