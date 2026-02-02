"""
run_bfs_80_20_grapeid.py

Backward Feature Selection (BFS) with 80/20 split for fast thesis experiments.

Key Design:
- Single 80/20 split (NOT LOGO, NOT CV) for speed
- Trains on BOTH rows (Row1 + Row2)
- Split by grape_id for CRACK+REGULAR, by segment group for REST classes
- Two runs: maximize CRACK F1, maximize CRACK PR-AUC
- Generates CSV logs + JSON with best features
- Plots generated separately by make_bfs_plots_from_csv.py

Leakage Prevention:
- Grape classes (CRACK, REGULAR): split by grape_id (all samples from same grape in train OR test)
- REST classes: split by segment/hs_dir group (no spatial overlap)
- Explicit assertions to verify no leakage

Author: Feature Selection Pipeline
Date: February 2026
"""

import sys
import json
import re
import warnings
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set, Union
from tqdm import tqdm
from dataclasses import dataclass, asdict

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== CONFIGURATION ====================

CSV_PATH_MULTICLASS = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_multiclass_2026-01-16.csv")
EXPERIMENTS_BASE = Path(r"C:\Users\yovel\Desktop\Grape_Project\experiments\feature_selection")

# Grape-related classes (split by grape_id)
GRAPE_CLASSES = {"CRACK", "REGULAR"}

# Column candidates for grape_id extraction
GRAPE_ID_COL_CANDIDATES = ["grape_id", "grapeid", "grape", "fruit_id", "berry_id", "id_grape"]

# Regex for extracting grape_id from mask_path
# Pattern: "sample_1_2024-09-25_..." -> captures sample number as grape identifier
GRAPE_ID_PATTERN = re.compile(r'sample_(\d+)_', re.IGNORECASE)

# Regex for extracting cluster/segment group from paths
# Pattern: data\raw\{row}_{cluster_num}\...
CLUSTER_PATTERN = re.compile(r'data[\\/]raw[\\/](\d+)_(\d+)')


# ==================== SNV ====================

def apply_snv(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard Normal Variate - per sample normalization."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + eps)


# ==================== EXTRACTION HELPERS ====================

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column from candidates (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def extract_grape_id_from_path(path: str) -> Optional[str]:
    """Extract grape_id from mask_path (sample_X pattern) or hs_dir."""
    if pd.isna(path):
        return None
    match = GRAPE_ID_PATTERN.search(str(path))
    return f"sample_{match.group(1)}" if match else None


def extract_cluster_from_path(path: str) -> Optional[str]:
    """Extract cluster_id from hs_dir or mask_path."""
    if pd.isna(path):
        return None
    match = CLUSTER_PATTERN.search(str(path))
    return f"{match.group(1)}_{match.group(2)}" if match else None


def get_wavelength_columns(df: pd.DataFrame, wl_min: float = 450, wl_max: float = 925) -> List[str]:
    """Get wavelength columns within range. Handles both '450.0' and '450.0nm' formats."""
    wl_cols = []
    for col in df.columns:
        try:
            # Remove 'nm' suffix if present
            col_clean = col.replace('nm', '').strip() if isinstance(col, str) else str(col)
            wl = float(col_clean)
            if wl_min <= wl <= wl_max:
                wl_cols.append(col)
        except (ValueError, TypeError):
            continue
    return sorted(wl_cols, key=lambda x: float(x.replace('nm', '').strip() if isinstance(x, str) else x))


# ==================== DATA LOADING AND SPLITTING ====================

@dataclass
class SplitManifest:
    """Manifest for the train/test split."""
    seed: int
    timestamp: str
    
    # Grape split info
    train_grape_ids: List[str]
    test_grape_ids: List[str]
    n_train_grape_samples: int
    n_test_grape_samples: int
    grape_id_extraction_method: str
    
    # REST split info
    train_rest_groups: List[str]
    test_rest_groups: List[str]
    n_train_rest_samples: int
    n_test_rest_samples: int
    rest_group_extraction_method: str
    
    # Total counts
    n_train_total: int
    n_test_total: int
    train_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]
    
    # Class info
    class_names: List[str]
    class_mapping: Dict[str, int]
    crack_class_idx: int
    n_features: int
    feature_names: List[str]


def load_and_prepare_data(
    csv_path: Path,
    label_col: str = "label",
    hs_dir_col: str = "hs_dir",
    segment_col: str = "mask_path",
    wl_min: float = 450,
    wl_max: float = 925,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], int]:
    """
    Load and prepare data for BFS experiment.
    
    Returns:
        df: DataFrame with all required columns
        feature_names: List of wavelength feature names
        class_mapping: Dict mapping class names to indices
        crack_idx: Index of CRACK class
    """
    print(f"\n[DATA] Loading from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df):,} rows")
    
    # Sample if needed (for fast_dev_run)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        print(f"[DATA] Sampled to {len(df):,} rows")
    
    # Validate required columns
    for col in [hs_dir_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found!")
    
    # Uppercase labels
    df[label_col] = df[label_col].astype(str).str.upper()
    
    # Get wavelength columns
    feature_names = get_wavelength_columns(df, wl_min, wl_max)
    if len(feature_names) == 0:
        raise ValueError(f"No wavelength columns found in range [{wl_min}, {wl_max}]")
    print(f"[DATA] Found {len(feature_names)} wavelength features in [{wl_min}, {wl_max}] nm")
    
    # Encode labels
    le = LabelEncoder()
    df["_encoded_label"] = le.fit_transform(df[label_col])
    class_names = list(le.classes_)
    class_mapping = {name: int(idx) for idx, name in enumerate(class_names)}
    
    print(f"[DATA] Classes: {class_names}")
    print(f"[DATA] Class distribution:")
    for cls in class_names:
        count = (df[label_col] == cls).sum()
        print(f"       {cls}: {count:,}")
    
    # Get CRACK index
    if "CRACK" not in class_mapping:
        raise ValueError("CRACK class not found in labels!")
    crack_idx = class_mapping["CRACK"]
    
    return df, feature_names, class_mapping, crack_idx


def create_80_20_split(
    df: pd.DataFrame,
    feature_names: List[str],
    class_mapping: Dict[str, int],
    label_col: str = "label",
    hs_dir_col: str = "hs_dir",
    segment_col: str = "mask_path",
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SplitManifest]:
    """
    Create 80/20 train/test split with proper leakage prevention.
    
    - Grape classes (CRACK, REGULAR): split by grape_id
    - REST classes: split by segment/cluster group
    """
    print(f"\n[SPLIT] Creating 80/20 train/test split (seed={seed})...")
    
    # Identify grape samples
    grape_mask = df[label_col].isin(GRAPE_CLASSES)
    df_grape = df[grape_mask].copy()
    df_rest = df[~grape_mask].copy()
    
    print(f"[SPLIT] Grape samples (CRACK+REGULAR): {len(df_grape):,}")
    print(f"[SPLIT] REST samples: {len(df_rest):,}")
    
    # ========== GRAPE SPLIT BY GRAPE_ID ==========
    grape_id_method = "N/A"
    if len(df_grape) > 0:
        # Try to find grape_id column
        grape_id_col = find_column(df_grape, GRAPE_ID_COL_CANDIDATES)
        
        if grape_id_col is not None:
            print(f"[SPLIT] Found grape_id column: '{grape_id_col}'")
            grape_id_method = f"column:{grape_id_col}"
        else:
            # Parse from paths - try mask_path first (has sample_X pattern), then hs_dir
            print(f"[SPLIT] Parsing grape_id from paths...")
            if segment_col in df_grape.columns:
                df_grape["_grape_id"] = df_grape[segment_col].apply(extract_grape_id_from_path)
                source = segment_col
            elif hs_dir_col in df_grape.columns:
                df_grape["_grape_id"] = df_grape[hs_dir_col].apply(extract_grape_id_from_path)
                source = hs_dir_col
            else:
                raise ValueError("Cannot extract grape_id: no mask_path or hs_dir column!")
            
            grape_id_col = "_grape_id"
            grape_id_method = f"parsed_from:{source}"
            
            # Check for failures
            n_failed = df_grape[grape_id_col].isna().sum()
            if n_failed > 0:
                print(f"[WARN] {n_failed} grape samples have no parseable grape_id")
                if n_failed == len(df_grape):
                    raise ValueError("Cannot extract grape_id for ANY grape samples! Check path format.")
                df_grape = df_grape[df_grape[grape_id_col].notna()].copy()
        
        unique_grape_ids = df_grape[grape_id_col].unique()
        print(f"[SPLIT] Unique grape_ids: {len(unique_grape_ids)}")
        
        # GroupShuffleSplit by grape_id
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        groups = df_grape[grape_id_col].values
        train_idx, test_idx = next(gss.split(df_grape, groups=groups))
        
        df_grape_train = df_grape.iloc[train_idx].copy()
        df_grape_test = df_grape.iloc[test_idx].copy()
        
        train_grape_ids = sorted(df_grape_train[grape_id_col].unique().tolist())
        test_grape_ids = sorted(df_grape_test[grape_id_col].unique().tolist())
        
        # LEAKAGE CHECK
        overlap = set(train_grape_ids) & set(test_grape_ids)
        assert not overlap, f"LEAKAGE: {len(overlap)} grape_ids in both train and test!"
        print(f"[SPLIT] Grape train: {len(df_grape_train):,} samples, {len(train_grape_ids)} grape_ids")
        print(f"[SPLIT] Grape test: {len(df_grape_test):,} samples, {len(test_grape_ids)} grape_ids")
    else:
        df_grape_train = pd.DataFrame()
        df_grape_test = pd.DataFrame()
        train_grape_ids = []
        test_grape_ids = []
    
    # ========== REST SPLIT BY SEGMENT/CLUSTER GROUP ==========
    rest_group_method = "N/A"
    if len(df_rest) > 0:
        # Parse cluster from paths
        print(f"[SPLIT] Parsing cluster group for REST samples...")
        if hs_dir_col in df_rest.columns:
            df_rest["_cluster_group"] = df_rest[hs_dir_col].apply(extract_cluster_from_path)
            source = hs_dir_col
        elif segment_col in df_rest.columns:
            df_rest["_cluster_group"] = df_rest[segment_col].apply(extract_cluster_from_path)
            source = segment_col
        else:
            # Fallback to segment_col as group
            df_rest["_cluster_group"] = df_rest[segment_col].astype(str)
            source = segment_col + " (direct)"
        
        rest_group_method = f"parsed_from:{source}"
        
        # Fill NAs with segment_col
        if df_rest["_cluster_group"].isna().any():
            if segment_col in df_rest.columns:
                df_rest.loc[df_rest["_cluster_group"].isna(), "_cluster_group"] = \
                    df_rest.loc[df_rest["_cluster_group"].isna(), segment_col].astype(str)
        
        unique_rest_groups = df_rest["_cluster_group"].unique()
        print(f"[SPLIT] Unique REST groups: {len(unique_rest_groups)}")
        
        # GroupShuffleSplit by cluster group
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        groups = df_rest["_cluster_group"].values
        train_idx, test_idx = next(gss.split(df_rest, groups=groups))
        
        df_rest_train = df_rest.iloc[train_idx].copy()
        df_rest_test = df_rest.iloc[test_idx].copy()
        
        train_rest_groups = sorted(df_rest_train["_cluster_group"].dropna().unique().tolist())
        test_rest_groups = sorted(df_rest_test["_cluster_group"].dropna().unique().tolist())
        
        # LEAKAGE CHECK
        overlap = set(train_rest_groups) & set(test_rest_groups)
        assert not overlap, f"LEAKAGE: {len(overlap)} REST groups in both train and test!"
        print(f"[SPLIT] REST train: {len(df_rest_train):,} samples, {len(train_rest_groups)} groups")
        print(f"[SPLIT] REST test: {len(df_rest_test):,} samples, {len(test_rest_groups)} groups")
    else:
        df_rest_train = pd.DataFrame()
        df_rest_test = pd.DataFrame()
        train_rest_groups = []
        test_rest_groups = []
    
    # ========== MERGE ==========
    df_train = pd.concat([df_grape_train, df_rest_train], ignore_index=True)
    df_test = pd.concat([df_grape_test, df_rest_test], ignore_index=True)
    
    print(f"\n[SPLIT] FINAL: Train={len(df_train):,}, Test={len(df_test):,}")
    
    # Extract X, y
    X_train = df_train[feature_names].values.astype(np.float32)
    X_test = df_test[feature_names].values.astype(np.float32)
    y_train = df_train["_encoded_label"].values.astype(int)
    y_test = df_test["_encoded_label"].values.astype(int)
    
    # Apply SNV (fit on train concept - but SNV is per-sample, no leakage)
    print("[SPLIT] Applying SNV normalization...")
    X_train = apply_snv(X_train)
    X_test = apply_snv(X_test)
    
    # Class distributions
    class_names = sorted(class_mapping.keys(), key=lambda x: class_mapping[x])
    train_dist = {}
    test_dist = {}
    for cls in class_names:
        idx = class_mapping[cls]
        train_dist[cls] = int((y_train == idx).sum())
        test_dist[cls] = int((y_test == idx).sum())
    
    print("[SPLIT] Train class distribution:")
    for cls, cnt in train_dist.items():
        print(f"       {cls}: {cnt:,}")
    print("[SPLIT] Test class distribution:")
    for cls, cnt in test_dist.items():
        print(f"       {cls}: {cnt:,}")
    
    # Create manifest
    manifest = SplitManifest(
        seed=seed,
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        train_grape_ids=train_grape_ids,
        test_grape_ids=test_grape_ids,
        n_train_grape_samples=len(df_grape_train),
        n_test_grape_samples=len(df_grape_test),
        grape_id_extraction_method=grape_id_method,
        train_rest_groups=train_rest_groups,
        test_rest_groups=test_rest_groups,
        n_train_rest_samples=len(df_rest_train),
        n_test_rest_samples=len(df_rest_test),
        rest_group_extraction_method=rest_group_method,
        n_train_total=len(df_train),
        n_test_total=len(df_test),
        train_class_distribution=train_dist,
        test_class_distribution=test_dist,
        class_names=class_names,
        class_mapping=class_mapping,
        crack_class_idx=class_mapping["CRACK"],
        n_features=len(feature_names),
        feature_names=feature_names,
    )
    
    return X_train, X_test, y_train, y_test, manifest


# ==================== XGBOOST HELPERS ====================

def get_xgb_params(n_classes: int, seed: int = 42, use_gpu: bool = True) -> dict:
    """Get XGBoost parameters."""
    params = {
        "n_estimators": 200,  # Reduced for BFS speed
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "verbosity": 0,
        "num_class": n_classes,
    }
    if use_gpu:
        params["device"] = "cuda"
    return params


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    crack_idx: int,
) -> Dict:
    """Compute all metrics for BFS logging."""
    # CRACK-specific metrics
    y_binary = (y_true == crack_idx).astype(int)
    y_prob_crack = y_prob[:, crack_idx]
    
    # Precision, Recall, F1 for CRACK
    crack_prec = precision_score(y_true, y_pred, labels=[crack_idx], average='micro', zero_division=0)
    crack_rec = recall_score(y_true, y_pred, labels=[crack_idx], average='micro', zero_division=0)
    crack_f1 = f1_score(y_true, y_pred, labels=[crack_idx], average='micro', zero_division=0)
    
    # PR-AUC and ROC-AUC for CRACK
    try:
        crack_pr_auc = average_precision_score(y_binary, y_prob_crack)
    except Exception:
        crack_pr_auc = 0.0
    
    try:
        crack_roc_auc = roc_auc_score(y_binary, y_prob_crack)
    except Exception:
        crack_roc_auc = 0.0
    
    # Global metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "crack_precision": crack_prec,
        "crack_recall": crack_rec,
        "crack_f1": crack_f1,
        "crack_pr_auc": crack_pr_auc,
        "roc_auc": crack_roc_auc,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# ==================== BACKWARD FEATURE SELECTION ====================

def run_bfs(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    crack_idx: int,
    n_classes: int,
    objective: str,  # "f1" or "prauc"
    min_features: int = 5,
    seed: int = 42,
    use_gpu: bool = True,
    output_dir: Path = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run Backward Feature Selection.
    
    Args:
        objective: "f1" (maximize crack_f1) or "prauc" (maximize crack_pr_auc)
    
    Returns:
        log_df: DataFrame with BFS log
        best_info: Dict with best feature set info
    """
    objective_col = "crack_f1" if objective == "f1" else "crack_pr_auc"
    print(f"\n{'='*70}")
    print(f"BACKWARD FEATURE SELECTION - Objective: {objective.upper()} ({objective_col})")
    print(f"{'='*70}")
    
    current_features = list(range(len(feature_names)))
    current_names = list(feature_names)
    
    log_rows = []
    best_score = -1
    best_iteration = 0
    best_n_features = len(feature_names)
    best_features = list(feature_names)
    
    iteration = 0
    total_iterations = len(feature_names) - min_features + 1
    
    pbar = tqdm(total=total_iterations, desc=f"BFS-{objective.upper()}")
    
    while len(current_features) >= min_features:
        # Get current feature subset
        X_train_sub = X_train[:, current_features]
        X_test_sub = X_test[:, current_features]
        
        # Train XGBoost
        xgb_params = get_xgb_params(n_classes, seed, use_gpu)
        xgb_params.pop("num_class", None)  # sklearn API doesn't use this
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_sub, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_sub)
        y_prob = model.predict_proba(X_test_sub)
        
        metrics = compute_metrics(y_test, y_pred, y_prob, crack_idx)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Determine removed feature (from previous iteration)
        if iteration == 0:
            removed_feature = ""
            removed_importance = np.nan
        else:
            # Already set below in previous iteration
            pass
        
        # Log this iteration
        log_row = {
            "objective_name": objective,
            "iteration": iteration,
            "n_features": len(current_features),
            "removed_feature": removed_feature if iteration > 0 else "",
            "removed_feature_importance": removed_importance if iteration > 0 else np.nan,
            **metrics,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
        }
        log_rows.append(log_row)
        
        # Track best
        current_score = metrics[objective_col]
        if current_score > best_score:
            best_score = current_score
            best_iteration = iteration
            best_n_features = len(current_features)
            best_features = list(current_names)
        
        pbar.set_postfix({
            "n_feat": len(current_features),
            objective_col: f"{current_score:.4f}",
            "best": f"{best_score:.4f}@{best_n_features}",
        })
        pbar.update(1)
        
        # Stop if at min_features
        if len(current_features) <= min_features:
            break
        
        # Find least important feature
        least_imp_idx = np.argmin(importances)
        removed_feature = current_names[least_imp_idx]
        removed_importance = float(importances[least_imp_idx])
        
        # Remove it
        del current_features[least_imp_idx]
        del current_names[least_imp_idx]
        
        iteration += 1
    
    pbar.close()
    
    # Create DataFrame
    log_df = pd.DataFrame(log_rows)
    
    # Save log
    if output_dir:
        log_path = output_dir / f"bfs_log_{objective}.csv"
        log_df.to_csv(log_path, index=False)
        print(f"[BFS] Saved log: {log_path}")
    
    # Best info
    best_info = {
        "objective_name": objective,
        "objective_metric": objective_col,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "best_n_features": best_n_features,
        "selected_features": best_features,
        "log_file": str(output_dir / f"bfs_log_{objective}.csv") if output_dir else None,
    }
    
    print(f"\n[BFS-{objective.upper()}] Best: {objective_col}={best_score:.4f} at n_features={best_n_features}")
    
    return log_df, best_info


# ==================== MAIN ====================

# ========== DEFAULT CONFIG (for VS Code Play button) ==========
# Modify these values to run directly without CLI args
DEFAULT_CONFIG = {
    "experiment_name": "bfs_full_run",  # Name for the experiment
    "seed": 42,                          # Random seed
    "min_features": 1,                   # Minimum features to keep (1 = full sweep)
    "fast_dev_run": False,               # Set True for quick test (5K samples, 20 iterations)
    "csv_path": str(CSV_PATH_MULTICLASS),
    "wl_min": 450,
    "wl_max": 925,
    "use_gpu": True,                     # Set False to disable GPU
}
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="BFS Feature Selection with 80/20 Split")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_CONFIG["experiment_name"], help="Name for the experiment")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed")
    parser.add_argument("--min_features", type=int, default=DEFAULT_CONFIG["min_features"], help="Minimum features to keep")
    parser.add_argument("--fast_dev_run", action="store_true", default=DEFAULT_CONFIG["fast_dev_run"], help="Fast development run with limited data")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CONFIG["csv_path"], help="Path to CSV data")
    parser.add_argument("--wl_min", type=float, default=DEFAULT_CONFIG["wl_min"], help="Minimum wavelength")
    parser.add_argument("--wl_max", type=float, default=DEFAULT_CONFIG["wl_max"], help="Maximum wavelength")
    parser.add_argument("--no_gpu", action="store_true", default=not DEFAULT_CONFIG["use_gpu"], help="Disable GPU")
    parser.add_argument("--resume", action="store_true", help="Resume from existing experiment (not implemented)")
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = EXPERIMENTS_BASE / args.experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BACKWARD FEATURE SELECTION - 80/20 SPLIT")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output: {experiment_dir}")
    print(f"Seed: {args.seed}")
    print(f"Min features: {args.min_features}")
    print(f"Fast dev run: {args.fast_dev_run}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load data
    max_samples = 5000 if args.fast_dev_run else None
    df, feature_names, class_mapping, crack_idx = load_and_prepare_data(
        csv_path=Path(args.csv_path),
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        max_samples=max_samples,
        seed=args.seed,
    )
    
    # Create split
    X_train, X_test, y_train, y_test, manifest = create_80_20_split(
        df=df,
        feature_names=feature_names,
        class_mapping=class_mapping,
        seed=args.seed,
    )
    
    n_classes = len(class_mapping)
    
    # Save split manifest
    manifest_path = experiment_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, default=str)
    print(f"\n[SAVE] Split manifest: {manifest_path}")
    
    # Adjust min_features for fast_dev_run
    min_features = args.min_features
    if args.fast_dev_run:
        min_features = max(50, len(feature_names) - 20)  # Only do 20 iterations
        print(f"[FAST] Adjusted min_features to {min_features}")
    
    # Run BFS for F1
    log_f1, best_f1 = run_bfs(
        X_train, X_test, y_train, y_test,
        feature_names, crack_idx, n_classes,
        objective="f1",
        min_features=min_features,
        seed=args.seed,
        use_gpu=not args.no_gpu,
        output_dir=experiment_dir,
    )
    
    # Run BFS for PR-AUC
    log_prauc, best_prauc = run_bfs(
        X_train, X_test, y_train, y_test,
        feature_names, crack_idx, n_classes,
        objective="prauc",
        min_features=min_features,
        seed=args.seed,
        use_gpu=not args.no_gpu,
        output_dir=experiment_dir,
    )
    
    # Save best features
    best_f1_path = experiment_dir / "best_features_f1.json"
    with open(best_f1_path, "w") as f:
        json.dump(best_f1, f, indent=2)
    print(f"[SAVE] Best features (F1): {best_f1_path}")
    
    best_prauc_path = experiment_dir / "best_features_prauc.json"
    with open(best_prauc_path, "w") as f:
        json.dump(best_prauc, f, indent=2)
    print(f"[SAVE] Best features (PR-AUC): {best_prauc_path}")
    
    # Create README
    readme_content = f"""BFS Feature Selection Experiment
================================

Experiment: {args.experiment_name}
Timestamp: {timestamp}
Seed: {args.seed}

Data:
- CSV: {args.csv_path}
- Wavelength range: [{args.wl_min}, {args.wl_max}] nm
- Train samples: {manifest.n_train_total:,}
- Test samples: {manifest.n_test_total:,}
- Features: {manifest.n_features}

Split:
- Grape (CRACK+REGULAR): split by grape_id
- REST: split by segment/cluster group
- No leakage verified

Results:
- Best F1: {best_f1['best_score']:.4f} at {best_f1['best_n_features']} features
- Best PR-AUC: {best_prauc['best_score']:.4f} at {best_prauc['best_n_features']} features

To generate plots (without retraining):
    python make_bfs_plots_from_csv.py --experiment_dir "{experiment_dir}"

Files:
- split_manifest.json: Train/test split details
- bfs_log_f1.csv: BFS log for F1 objective
- bfs_log_prauc.csv: BFS log for PR-AUC objective
- best_features_f1.json: Best feature set for F1
- best_features_prauc.json: Best feature set for PR-AUC
- plots/: Generated plots (after running make_bfs_plots_from_csv.py)
"""
    readme_path = experiment_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"[SAVE] README: {readme_path}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED in {elapsed/60:.1f} min")
    print(f"Output: {experiment_dir}")
    print(f"{'='*70}")
    
    return experiment_dir


if __name__ == "__main__":
    main()
