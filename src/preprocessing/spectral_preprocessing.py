# src/preprocessing/spectral_preprocessing.py
"""
Reusable preprocessing utilities for pixel-level spectral datasets.

Main function:
    preprocess_pixel_level_dataset(df, ...) -> (X, y, groups, feature_names)
    preprocess_multiclass_dataset(df, ...) -> PreprocessedData (namedtuple with extended info)

Usage example:
    from src.preprocessing.spectral_preprocessing import preprocess_pixel_level_dataset, preprocess_multiclass_dataset

    df = pd.read_csv("my_spectral_data.csv")

    # Legacy binary usage:
    X, y, groups, feature_names = preprocess_pixel_level_dataset(
        df,
        wl_min=450,
        wl_max=925,
        apply_snv=True,
        balanced=False,
    )

    # Multi-class usage:
    data = preprocess_multiclass_dataset(
        df,
        label_col="label",
        balanced=True,
        cap_class="CRACK",
    )
    # data.X, data.y, data.groups, data.segment_ids, data.image_ids,
    # data.label_encoder, data.class_names, data.feature_names
"""
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, NamedTuple, Union

import numpy as np
import pandas as pd

# Optional imports
try:
    from sklearn.covariance import MinCovDet
    _SKLEARN_COV_AVAILABLE = True
except Exception:
    _SKLEARN_COV_AVAILABLE = False

try:
    from sklearn.preprocessing import LabelEncoder
    _LABEL_ENCODER_AVAILABLE = True
except Exception:
    _LABEL_ENCODER_AVAILABLE = False


# Default label mapping for string labels -> int (legacy binary)
DEFAULT_LABEL_MAP: Dict[str, int] = {
    "REGULAR": 0,
    "CRACK": 1,
    "CRACKED": 1,
}

# Grape-related classes for domain-aware splitting
GRAPE_CLASSES = {"REGULAR", "CRACK", "CRACKED"}
GRAPE_CLASS_IDS_3CLASS = {1, 2}  # For 3-class: REGULAR=1, CRACK=2 (original labels)


class PreprocessedData(NamedTuple):
    """Container for preprocessed multi-class dataset."""
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray  # cluster_id for LOGO
    segment_ids: np.ndarray  # mask_path for leakage prevention
    image_ids: np.ndarray  # hs_dir for image-level grouping
    label_encoder: object  # LabelEncoder or None
    class_names: List[str]
    class_mapping: Dict[str, int]
    feature_names: List[str]
    original_labels: np.ndarray  # String/int labels before encoding
    grape_class_indices: set  # Encoded indices for grape classes (for domain-aware split)
    original_labels: np.ndarray  # String labels before encoding

# Regex to parse wavelength from column name (e.g., "450nm", "450.5nm", "450 nm")
_WL_RE = re.compile(r"([\d.]+)\s*nm$", re.IGNORECASE)


def extract_cluster_id(hs_dir: str) -> str:
    """
    Extract cluster folder name (three levels up) from hs_dir path.
    Matches your current logic in compare_models.py.
    """
    parts = Path(str(hs_dir)).parts
    return parts[-3] if len(parts) >= 3 else "unknown"


def _parse_wavelength(col_name: str) -> Optional[float]:
    """
    Parse wavelength from column name ending with 'nm' (e.g., '450nm', '450.5nm', '450 nm').
    Returns float wavelength or None if not parseable.
    """
    m = _WL_RE.search(col_name.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _snv(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standard Normal Variate (SNV) per spectrum:
    For each row: (x - mean(row)) / std(row)
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def _undersample_to_minority(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    """
    Undersample majority class to match minority class count.
    """
    counts = df[label_col].value_counts(dropna=False)
    if len(counts) < 2:
        return df

    minority_label = counts.idxmin()
    n_min = int(counts.min())

    df_min = df[df[label_col] == minority_label]
    df_maj = df[df[label_col] != minority_label].sample(n=n_min, random_state=seed)

    return pd.concat([df_min, df_maj], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _undersample_cap_based_segment_proportional(
    df: pd.DataFrame,
    label_col: str,
    segment_col: str,
    cap_class: Union[str, int, None] = None,
    max_samples_per_class: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cap-based undersampling with segment-proportional sampling.

    You can specify the cap in two ways:
    1. `cap_class`: Use the count of a reference class (e.g., "CRACK") as the maximum
    2. `max_samples_per_class`: Use a specific number as the maximum
    
    If both are provided, the minimum of the two is used.
    Sampling is done proportionally across segments to maintain segment diversity.

    Args:
        df: DataFrame with samples.
        label_col: Column name for labels.
        segment_col: Column name for segment IDs (e.g., mask_path).
        cap_class: The class whose count is used as the cap (e.g., "CRACK" or 2).
                   Set to None to only use max_samples_per_class.
        max_samples_per_class: Maximum number of samples per class (absolute cap).
                               Set to None to only use cap_class count.
        seed: Random seed for reproducibility.

    Returns:
        Balanced DataFrame with segment-proportional sampling.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    # Determine the cap value
    n_cap = None
    cap_source = []
    
    # Option 1: Use cap_class count
    if cap_class is not None:
        if cap_class not in df[label_col].values:
            print(f"[BALANCE] WARNING: Cap class '{cap_class}' not found in data, ignoring cap_class")
        else:
            n_cap_from_class = int((df[label_col] == cap_class).sum())
            cap_source.append(f"cap_class '{cap_class}' = {n_cap_from_class}")
            n_cap = n_cap_from_class
    
    # Option 2: Use max_samples_per_class
    if max_samples_per_class is not None and max_samples_per_class > 0:
        cap_source.append(f"max_samples_per_class = {max_samples_per_class}")
        if n_cap is None:
            n_cap = max_samples_per_class
        else:
            n_cap = min(n_cap, max_samples_per_class)
    
    # If no cap specified, return original data
    if n_cap is None:
        print(f"[BALANCE] No cap specified (cap_class={cap_class}, max_samples_per_class={max_samples_per_class}), returning original data")
        return df
    
    print(f"[BALANCE] Cap sources: {', '.join(cap_source)}")
    print(f"[BALANCE] Final cap value: {n_cap} samples per class")

    balanced_dfs = []

    for label in df[label_col].unique():
        class_df = df[df[label_col] == label].copy()
        n_class = len(class_df)
        target_n = min(n_class, n_cap)

        if n_class <= target_n:
            # Keep all samples for this class
            balanced_dfs.append(class_df)
            print(f"[BALANCE] Class '{label}': kept all {n_class} samples (target={target_n})")
            continue

        # Need to undersample - do it proportionally across segments
        # Group by segment
        segment_groups = class_df.groupby(segment_col)
        segment_sizes = segment_groups.size()
        total_class_size = segment_sizes.sum()

        # Compute proportional allocation per segment
        # k_i = round(target_n * (len(segment_i) / total_class_size))
        allocations = {}
        for seg_id, seg_size in segment_sizes.items():
            k_i = round(target_n * (seg_size / total_class_size))
            allocations[seg_id] = max(1, k_i)  # At least 1 sample per segment

        # Adjust to hit exact target
        total_allocated = sum(allocations.values())
        diff = target_n - total_allocated

        if diff != 0:
            # Distribute difference across segments proportionally
            sorted_segs = sorted(allocations.keys(), key=lambda s: segment_sizes[s], reverse=True)
            for i, seg_id in enumerate(sorted_segs):
                if diff == 0:
                    break
                if diff > 0:
                    # Need more samples - add to larger segments
                    if allocations[seg_id] < segment_sizes[seg_id]:
                        allocations[seg_id] += 1
                        diff -= 1
                else:
                    # Need fewer samples - reduce from larger segments (but keep at least 1)
                    if allocations[seg_id] > 1:
                        allocations[seg_id] -= 1
                        diff += 1

        # Sample from each segment
        sampled_dfs = []
        for seg_id, k_i in allocations.items():
            seg_df = segment_groups.get_group(seg_id)
            k_actual = min(k_i, len(seg_df))
            if k_actual > 0:
                sampled = seg_df.sample(n=k_actual, random_state=rng.randint(0, 2**31))
                sampled_dfs.append(sampled)

        if sampled_dfs:
            class_balanced = pd.concat(sampled_dfs, axis=0)
            balanced_dfs.append(class_balanced)
            print(f"[BALANCE] Class '{label}': sampled {len(class_balanced)} from {n_class} samples across {len(allocations)} segments")

    if balanced_dfs:
        result = pd.concat(balanced_dfs, axis=0)
        # Shuffle the result
        result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"[BALANCE] Total after balancing: {len(result)} samples")
        return result
    else:
        return df


def _remove_outliers_mahalanobis(X: np.ndarray, p_loss: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Remove top p_loss fraction of samples by robust Mahalanobis distance (MinCovDet).
    Returns a boolean mask of kept samples.
    """
    if not _SKLEARN_COV_AVAILABLE:
        raise ImportError("scikit-learn MinCovDet not available; cannot remove outliers.")

    if not (0.0 < p_loss < 0.5):
        raise ValueError("p_loss must be in (0, 0.5).")

    # Robust covariance fit
    mcd = MinCovDet(random_state=seed).fit(X)
    d2 = mcd.mahalanobis(X)  # squared distances

    # Keep lowest (1 - p_loss) fraction
    thresh = np.quantile(d2, 1.0 - p_loss)
    keep = d2 <= thresh
    return keep


def preprocess_pixel_level_dataset(
    df: pd.DataFrame,
    *,
    wl_min: float = 450,
    wl_max: float = 925,
    apply_snv: bool = True,
    remove_outliers: bool = False,
    p_loss: float = 0.01,
    balanced: bool = False,
    label_col: str = "label",
    hs_dir_col: str = "hs_dir",
    label_map: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess pixel-level spectral dataset for modeling.

    Steps:
      1) Validate required columns
      2) Map string labels to int (if needed)
      3) Extract groups (cluster_id) from hs_dir
      4) Optional undersampling balance (balanced=True)
      5) Select wavelength feature columns ending with 'nm'
      6) Filter wavelengths to [wl_min, wl_max] (based on parsed wavelength)
      7) Build X, y, groups
      8) Optional SNV normalization per spectrum
      9) Optional outlier removal using robust Mahalanobis distance

    Args:
        df: DataFrame with spectral data.
        wl_min: Minimum wavelength to include (nm).
        wl_max: Maximum wavelength to include (nm).
        apply_snv: Apply SNV normalization per spectrum.
        remove_outliers: Remove outliers using Mahalanobis distance.
        p_loss: Fraction of outliers to remove if remove_outliers=True.
        balanced: Undersample majority class to balance.
        label_col: Column name for labels.
        hs_dir_col: Column name for hs_dir (used to extract cluster groups).
        label_map: Optional dict mapping string labels to int (default: DEFAULT_LABEL_MAP).
        seed: Random seed for reproducibility.

    Returns:
        X (np.float32), y (np.int), groups (np.str), feature_names (list[str])
    """
    # --- Validate ---
    for col in (hs_dir_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # --- Copy and prepare ---
    df = df.copy()

    # --- Map string labels to int if necessary ---
    if df[label_col].dtype == object or df[label_col].dtype.name == 'str':
        if label_map is None:
            label_map = DEFAULT_LABEL_MAP
        df[label_col] = df[label_col].str.upper().map(label_map)
        if df[label_col].isna().any():
            unmapped = df[df[label_col].isna()][label_col].unique()
            raise ValueError(f"Unmapped label values found: {unmapped}. Provide a label_map.")
        df[label_col] = df[label_col].astype(int)

    # --- Add groups ---
    df["cluster_id"] = df[hs_dir_col].apply(extract_cluster_id)

    # --- Optional balance ---
    if balanced:
        df = _undersample_to_minority(df, label_col=label_col, seed=seed)

    # --- Select spectral columns ---
    nm_cols = [c for c in df.columns if str(c).strip().lower().endswith("nm")]
    if not nm_cols:
        raise ValueError("No spectral feature columns ending with 'nm' were found.")

    # Parse wavelengths and filter range
    wl_map = []
    for c in nm_cols:
        wl = _parse_wavelength(str(c))
        if wl is not None:
            wl_map.append((wl, c))

    if not wl_map:
        raise ValueError("Found 'nm' columns, but none were parseable as '<number>nm'.")

    wl_map.sort(key=lambda t: t[0])
    selected = [c for (wl, c) in wl_map if wl_min <= wl <= wl_max]
    if not selected:
        raise ValueError(f"No wavelengths found in the range [{wl_min}, {wl_max}] nm.")

    # --- Build arrays ---
    X = df[selected].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=int)
    groups = df["cluster_id"].to_numpy(dtype=str)

    # --- SNV ---
    if apply_snv:
        X = _snv(X)

    # --- Optional outlier removal ---
    if remove_outliers:
        keep_mask = _remove_outliers_mahalanobis(X, p_loss=p_loss, seed=seed)
        X = X[keep_mask]
        y = y[keep_mask]
        groups = groups[keep_mask]

    return X, y, groups, selected


def preprocess_multiclass_dataset(
    df: pd.DataFrame,
    *,
    wl_min: float = 450,
    wl_max: float = 925,
    apply_snv: bool = True,
    remove_outliers: bool = False,
    p_loss: float = 0.01,
    balanced: bool = False,
    label_col: str = "label",
    hs_dir_col: str = "hs_dir",
    segment_col: str = "mask_path",
    cap_class: Optional[Union[str, int]] = None,
    max_samples_per_class: Optional[int] = None,
    seed: int = 42,
) -> PreprocessedData:
    """
    Preprocess multi-class pixel-level spectral dataset for modeling.

    Extended version of preprocess_pixel_level_dataset that:
    - Supports true multi-class (not just binary)
    - Uses LabelEncoder for string labels
    - Returns segment_ids and image_ids for domain-aware splitting
    - Uses cap-based segment-proportional undersampling when balanced=True

    Args:
        df: DataFrame with spectral data.
        wl_min: Minimum wavelength to include (nm).
        wl_max: Maximum wavelength to include (nm).
        apply_snv: Apply SNV normalization per spectrum.
        remove_outliers: Remove outliers using Mahalanobis distance.
        p_loss: Fraction of outliers to remove if remove_outliers=True.
        balanced: Apply cap-based segment-proportional undersampling.
        label_col: Column name for labels.
        hs_dir_col: Column name for hs_dir (image-level grouping).
        segment_col: Column name for segment IDs (mask_path).
        cap_class: The class whose count is used as the cap for balancing
                   (default: "CRACK" for string labels, 2 for integer labels).
                   Set to None to only use max_samples_per_class.
        max_samples_per_class: Maximum number of samples per class (absolute cap).
                               If both cap_class and max_samples_per_class are set,
                               the minimum of the two is used.
                               Set to None to only use cap_class count.
        seed: Random seed for reproducibility.

    Returns:
        PreprocessedData namedtuple with:
            X, y, groups, segment_ids, image_ids, label_encoder,
            class_names, class_mapping, feature_names, original_labels
    """
    if not _LABEL_ENCODER_AVAILABLE:
        raise ImportError("sklearn.preprocessing.LabelEncoder not available")

    # --- Validate required columns ---
    required_cols = [hs_dir_col, label_col]
    if segment_col:
        required_cols.append(segment_col)
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # --- Copy and prepare ---
    df = df.copy()

    # Store original string labels before any transformation
    original_labels_series = df[label_col].copy()

    # Determine if labels are strings or already integers
    labels_are_strings = df[label_col].dtype == object or df[label_col].dtype.name == 'str'

    if labels_are_strings:
        # Uppercase for consistency
        df[label_col] = df[label_col].str.upper()
        original_labels_series = df[label_col].copy()

        # Use LabelEncoder
        label_encoder = LabelEncoder()
        df["_encoded_label"] = label_encoder.fit_transform(df[label_col])
        class_names = list(label_encoder.classes_)
        class_mapping = {name: int(idx) for idx, name in enumerate(class_names)}

        # Determine cap_class
        if cap_class is None:
            cap_class = "CRACK" if "CRACK" in class_names else class_names[0]
        elif isinstance(cap_class, str):
            cap_class = cap_class.upper()

        working_label_col = "_encoded_label"
        cap_class_for_balance = cap_class  # String for balancing (uses original label col)

        # Compute grape_class_indices from GRAPE_CLASSES (encoded indices)
        grape_class_indices = set()
        for grape_name in GRAPE_CLASSES:
            if grape_name in class_mapping:
                grape_class_indices.add(class_mapping[grape_name])
        print(f"[PREPROCESS] Grape class indices (encoded): {grape_class_indices}")
    else:
        # Labels are already integers but may not be 0-based (e.g., [1,2,3])
        # We need to re-encode them to [0,1,2,...] for XGBoost compatibility
        unique_labels = sorted(df[label_col].unique())
        class_names = [str(lbl) for lbl in unique_labels]

        # Create mapping from original int labels to 0-based indices
        # e.g., {1: 0, 2: 1, 3: 2} for labels [1,2,3]
        original_to_encoded = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        class_mapping = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}

        # Use LabelEncoder to re-encode integer labels to 0-based
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_labels)
        df["_encoded_label"] = label_encoder.transform(df[label_col])

        # Determine cap_class - need to use original label value for balancing
        if cap_class is None:
            cap_class = 2 if 2 in unique_labels else unique_labels[0]  # Default for 3-class

        working_label_col = "_encoded_label"
        cap_class_for_balance = cap_class  # Original integer for balancing (uses original label col)

        print(f"[PREPROCESS] Integer labels re-encoded: {dict(zip(unique_labels, range(len(unique_labels))))}")

        # Compute grape_class_indices from GRAPE_CLASS_IDS_3CLASS (re-encoded)
        # Original grape classes are {1, 2}, map to encoded indices
        grape_class_indices = set()
        for orig_label in GRAPE_CLASS_IDS_3CLASS:
            if orig_label in original_to_encoded:
                grape_class_indices.add(original_to_encoded[orig_label])
        print(f"[PREPROCESS] Grape class indices (encoded from {GRAPE_CLASS_IDS_3CLASS}): {grape_class_indices}")

    print(f"[PREPROCESS] Classes found: {class_names}")
    print(f"[PREPROCESS] Class mapping: {class_mapping}")
    if cap_class_for_balance is not None:
        print(f"[PREPROCESS] Cap class for balancing: {cap_class_for_balance}")
    if max_samples_per_class is not None:
        print(f"[PREPROCESS] Max samples per class: {max_samples_per_class}")

    # --- Add groups ---
    df["cluster_id"] = df[hs_dir_col].apply(extract_cluster_id)

    # --- Apply sample limit / balancing ---
    # If balanced=True: use cap_class AND/OR max_samples_per_class
    # If balanced=False but max_samples_per_class is set: still apply the limit (for fast testing)
    # Always use original label column for balancing, then re-encode
    if balanced:
        # Use original label column for balancing
        df = _undersample_cap_based_segment_proportional(
            df,
            label_col=label_col,  # Always use original label column
            segment_col=segment_col,
            cap_class=cap_class_for_balance,
            max_samples_per_class=max_samples_per_class,
            seed=seed,
        )
        # Re-encode after balancing
        df["_encoded_label"] = label_encoder.transform(df[label_col])
        # Update original_labels_series to match balanced df
        original_labels_series = df[label_col].copy()
    elif max_samples_per_class is not None:
        # Unbalanced mode but with sample limit (for fast testing)
        print(f"[PREPROCESS] Applying max_samples_per_class={max_samples_per_class} to unbalanced data")
        df = _undersample_cap_based_segment_proportional(
            df,
            label_col=label_col,  # Always use original label column
            segment_col=segment_col,
            cap_class=None,  # No cap class, only use max_samples_per_class
            max_samples_per_class=max_samples_per_class,
            seed=seed,
        )
        # Re-encode after sampling
        df["_encoded_label"] = label_encoder.transform(df[label_col])
        # Update original_labels_series to match sampled df
        original_labels_series = df[label_col].copy()

    # --- Select spectral columns ---
    nm_cols = [c for c in df.columns if str(c).strip().lower().endswith("nm")]
    if not nm_cols:
        raise ValueError("No spectral feature columns ending with 'nm' were found.")

    # Parse wavelengths and filter range
    wl_map = []
    for c in nm_cols:
        wl = _parse_wavelength(str(c))
        if wl is not None:
            wl_map.append((wl, c))

    if not wl_map:
        raise ValueError("Found 'nm' columns, but none were parseable as '<number>nm'.")

    wl_map.sort(key=lambda t: t[0])
    selected = [c for (wl, c) in wl_map if wl_min <= wl <= wl_max]
    if not selected:
        raise ValueError(f"No wavelengths found in the range [{wl_min}, {wl_max}] nm.")

    # --- Build arrays ---
    X = df[selected].to_numpy(dtype=np.float32)
    y = df[working_label_col].to_numpy(dtype=int)
    groups = df["cluster_id"].to_numpy(dtype=str)
    segment_ids = df[segment_col].to_numpy(dtype=str) if segment_col else np.array([""] * len(df))
    image_ids = df[hs_dir_col].to_numpy(dtype=str)
    original_labels = original_labels_series.to_numpy()

    # --- SNV ---
    if apply_snv:
        X = _snv(X)

    # --- Optional outlier removal ---
    if remove_outliers:
        keep_mask = _remove_outliers_mahalanobis(X, p_loss=p_loss, seed=seed)
        X = X[keep_mask]
        y = y[keep_mask]
        groups = groups[keep_mask]
        segment_ids = segment_ids[keep_mask]
        image_ids = image_ids[keep_mask]
        original_labels = original_labels[keep_mask]

    print(f"[PREPROCESS] Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[PREPROCESS] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return PreprocessedData(
        X=X,
        y=y,
        groups=groups,
        segment_ids=segment_ids,
        image_ids=image_ids,
        label_encoder=label_encoder,
        class_names=class_names,
        class_mapping=class_mapping,
        feature_names=selected,
        original_labels=original_labels,
        grape_class_indices=grape_class_indices,
    )


def save_class_mapping(class_mapping: Dict[str, int], output_path: Path) -> None:
    """Save class mapping to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"[SAVED] Class mapping to {output_path}")


# --- Minimal smoke test ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for preprocess_pixel_level_dataset")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_2026-01-13.csv"),
        help="Path to CSV file with spectral data",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[INFO] DataFrame shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns[:10])} ... ({len(df.columns)} total)")

    X, y, groups, feature_names = preprocess_pixel_level_dataset(
        df,
        wl_min=450,
        wl_max=925,
        apply_snv=True,
        remove_outliers=False,
        balanced=False,
    )

    print(f"\n[RESULT] X.shape: {X.shape}, dtype: {X.dtype}")
    print(f"[RESULT] y.shape: {y.shape}, dtype: {y.dtype}, unique: {np.unique(y)}")
    print(f"[RESULT] groups.shape: {groups.shape}, dtype: {groups.dtype}, unique count: {len(np.unique(groups))}")
    print(f"[RESULT] feature_names: {len(feature_names)} features, first 5: {feature_names[:5]}")
    print("\n[SUCCESS] Smoke test passed!")

