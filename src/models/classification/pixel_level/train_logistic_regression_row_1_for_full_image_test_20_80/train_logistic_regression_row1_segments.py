"""

train_logistic_regression_row1_segments.py



Logistic Regression training script for pixel-level classification using ONLY ROW 1 data.

Segment-level samples (mask_path) with group-based train/val split by cluster_id.



Features:

- Filters data to ROW 1 only before preprocessing

- Uses GroupShuffleSplit by cluster_id for train/val split (90/10)

- Uses ALL 159 spectral features (450-925 nm after SNV)

- StandardScaler applied before Logistic Regression

- Supports both multiclass and 3class modes (same as benchmark)

- Saves model, scaler, class_mapping, split_manifest, predictions, and metrics

- Compatible with final test on ROW 2 (same preprocessing, label encoding)



NO LOGO, NO CV - just one train/val split for final model training.

"""



import json

import re

import sys

import time

import warnings

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

from dataclasses import dataclass



# Add project root to Python path for imports

# Path: src/models/classification/pixel_level/train_logistic_regression_row_1_for_full_image_test_20_80/train_logistic_regression_row1_segments.py

# parents[0]=train_logistic_regression_row_1_for_full_image_test_20_80, [1]=pixel_level, [2]=classification, [3]=models, [4]=src, [5]=Grape_Project

_PROJECT_ROOT = Path(__file__).resolve().parents[5]  # Navigate up to Grape_Project

if str(_PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(_PROJECT_ROOT))

    

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    accuracy_score, f1_score, precision_score, recall_score,

    roc_auc_score, average_precision_score, confusion_matrix, classification_report

)

from sklearn.utils.class_weight import compute_sample_weight



# Import the reusable preprocessing function (same as benchmark)

from src.preprocessing.spectral_preprocessing import (

    preprocess_multiclass_dataset,

    PreprocessedData,

)



try:

    import joblib

except ImportError:

    joblib = None

    print("[WARN] joblib not available; model saving will use pickle fallback.")



from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=ConvergenceWarning)



# ==================== CONFIGURATION ====================



@dataclass

class DatasetConfig:

    """Configuration for a dataset."""

    name: str

    csv_path: Path

    target_col: str

    crack_identifier: Union[str, int]  # "CRACK" for multiclass, 2 for 3class

    grape_classes: set  # Classes that are grape-related

    is_3class: bool = False





# Dataset configurations (same paths as benchmark)

CSV_PATH_MULTICLASS = Path(str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_multiclass_2026-01-16.csv"))



CSV_PATH_3CLASS = Path(str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_3class_2026-01-16.csv"))



DATASET_CONFIGS = [

    DatasetConfig(

        name="multiclass",

        csv_path=CSV_PATH_MULTICLASS,

        target_col="label",

        crack_identifier="CRACK",

        grape_classes={"REGULAR", "CRACK"},

        is_3class=False,

    ),

    # 3class config disabled - using only multiclass

    # DatasetConfig(

    #     name="3class",

    #     csv_path=CSV_PATH_3CLASS,

    #     target_col="label_3class_id",

    #     crack_identifier=2,

    #     grape_classes={1, 2},  # REGULAR=1, CRACK=2 in 3-class

    #     is_3class=True,

    # ),

]



# ==================== ROW AND CLUSTER EXTRACTION ====================



# Regex patterns for extracting row and cluster from paths

# Pattern: data\raw\{row}_{cluster_num}\... or data/raw/{row}_{cluster_num}/...

ROW_CLUSTER_PATTERN = re.compile(r'data[\\/]raw[\\/](\d+)_(\d+)')



# Column name candidates for row identification

ROW_COL_CANDIDATES = ["row", "vine_row", "row_id", "ROW"]

CLUSTER_COL_CANDIDATES = ["cluster_id", "CLUSTER_ID", "cluster", "cluster_name"]





def extract_row_from_path(path: str) -> Optional[int]:

    """

    Extract row number from hs_dir or mask_path.

    Pattern: data\raw\{row}_{cluster_num}\...

    """

    match = ROW_CLUSTER_PATTERN.search(str(path))

    return int(match.group(1)) if match else None





def extract_cluster_from_path(path: str) -> Optional[str]:

    """

    Extract cluster_id from hs_dir or mask_path.

    Pattern: data\raw\{row}_{cluster_num}\... -> "{row}_{cluster_num}"

    """

    match = ROW_CLUSTER_PATTERN.search(str(path))

    return f"{match.group(1)}_{match.group(2)}" if match else None





def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:

    """Find the first matching column from candidates (case-insensitive)."""

    df_cols_lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:

        if candidate.lower() in df_cols_lower:

            return df_cols_lower[candidate.lower()]

    return None





# ==================== EXPERIMENT SETTINGS ====================



# Create timestamped experiment directory

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EXPERIMENT_DIR = Path(str(_PROJECT_ROOT / f"experiments/logistic_regression_row1_only_train_val_{TIMESTAMP}"))



RANDOM_STATE = 42

VAL_SIZE = 0.10  # 10% validation



# ==================== BALANCING SETTINGS ====================

# Maximum samples per class (same as benchmark)

MAX_SAMPLES_PER_CLASS = 50000



# ==================== Logistic Regression HYPERPARAMETERS ====================

# Easy to tweak - all Logistic Regression settings in one place

LOGISTIC_REGRESSION_PARAMS = {

    "C": 1.0,                    # Inverse of regularization strength

    "solver": "saga",            # Supports L1 + multinomial

    "max_iter": 5000,            # Maximum iterations for convergence

    "penalty": "l1",             # L1 regularization (sparse coefficients)

    "class_weight": "balanced",  # Handle class imbalance

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "verbose": 1,

}



# ==================== HELPER FUNCTIONS ====================



def ensure_dir(path: Path) -> Path:

    """Ensure directory exists, creating it if necessary."""

    path.mkdir(parents=True, exist_ok=True)

    return path





def slugify(name: str) -> str:

    """Convert name to filesystem-friendly slug."""

    slug = name.lower().strip()

    slug = re.sub(r'[^a-z0-9]+', '_', slug)

    return slug.strip('_')





# ==================== DATA LOADING AND FILTERING ====================



def load_and_filter_row1(

    config: DatasetConfig,

    hs_dir_col: str = "hs_dir",

    segment_col: str = "mask_path",

) -> Tuple[pd.DataFrame, Dict]:

    """

    Load CSV and filter to ROW 1 only.

    

    Returns:

        df_row1: Filtered DataFrame with only ROW 1 samples

        filter_info: Dict with filtering information for manifest

    """

    print(f"[LOAD] Loading data from {config.csv_path}...")

    df = pd.read_csv(config.csv_path)

    print(f"[LOAD] Loaded {len(df):,} total rows")

    

    filter_info = {

        "original_rows": len(df),

        "row_detection_method": None,

        "row_regex": None,

    }

    

    # Try to find existing row column

    row_col = find_column(df, ROW_COL_CANDIDATES)

    

    if row_col is not None:

        print(f"[ROW] Found existing row column: '{row_col}'")

        filter_info["row_detection_method"] = f"column:{row_col}"

    else:

        # Parse row from hs_dir or mask_path

        print(f"[ROW] No row column found, parsing from paths...")

        

        if hs_dir_col in df.columns:

            df["_parsed_row"] = df[hs_dir_col].apply(extract_row_from_path)

            parse_source = hs_dir_col

        elif segment_col in df.columns:

            df["_parsed_row"] = df[segment_col].apply(extract_row_from_path)

            parse_source = segment_col

        else:

            raise ValueError(f"Cannot extract row: no '{hs_dir_col}' or '{segment_col}' column found.")

        

        row_col = "_parsed_row"

        filter_info["row_detection_method"] = f"parsed_from:{parse_source}"

        filter_info["row_regex"] = str(ROW_CLUSTER_PATTERN.pattern)

        

        # Check for unparseable rows

        n_unparsed = df[row_col].isna().sum()

        if n_unparsed > 0:

            print(f"[WARN] {n_unparsed} rows could not be parsed (will be excluded)")

            df = df[df[row_col].notna()].copy()

            df[row_col] = df[row_col].astype(int)

    

    # Show unique rows

    unique_rows = sorted(df[row_col].dropna().unique())

    print(f"[ROW] Unique rows found: {unique_rows}")

    

    # Filter to ROW 1

    df_row1 = df[df[row_col] == 1].copy()

    print(f"[ROW] Filtered to ROW 1: {len(df_row1):,} samples")

    

    filter_info["row1_samples"] = len(df_row1)

    filter_info["rows_excluded"] = [r for r in unique_rows if r != 1]

    

    if len(df_row1) == 0:

        raise ValueError("No samples found for ROW 1!")

    

    return df_row1, filter_info





def add_cluster_id_column(

    df: pd.DataFrame,

    hs_dir_col: str = "hs_dir",

    segment_col: str = "mask_path",

) -> Tuple[str, Dict]:

    """

    Ensure cluster_id column exists in DataFrame.

    

    Returns:

        cluster_col: Name of the cluster column

        cluster_info: Dict with cluster detection info

    """

    cluster_info = {

        "cluster_detection_method": None,

        "cluster_regex": None,

    }

    

    # Try to find existing cluster column

    cluster_col = find_column(df, CLUSTER_COL_CANDIDATES)

    

    if cluster_col is not None:

        print(f"[CLUSTER] Found existing cluster column: '{cluster_col}'")

        cluster_info["cluster_detection_method"] = f"column:{cluster_col}"

    else:

        # Parse cluster from hs_dir or mask_path

        print(f"[CLUSTER] No cluster column found, parsing from paths...")

        

        if hs_dir_col in df.columns:

            df["cluster_id"] = df[hs_dir_col].apply(extract_cluster_from_path)

            parse_source = hs_dir_col

        elif segment_col in df.columns:

            df["cluster_id"] = df[segment_col].apply(extract_cluster_from_path)

            parse_source = segment_col

        else:

            raise ValueError(f"Cannot extract cluster: no '{hs_dir_col}' or '{segment_col}' column found.")

        

        cluster_col = "cluster_id"

        cluster_info["cluster_detection_method"] = f"parsed_from:{parse_source}"

        cluster_info["cluster_regex"] = str(ROW_CLUSTER_PATTERN.pattern)

        

        # Check for unparseable clusters

        n_unparsed = df[cluster_col].isna().sum()

        if n_unparsed > 0:

            print(f"[WARN] {n_unparsed} rows have unparseable cluster_id")

    

    # Show unique clusters

    unique_clusters = sorted(df[cluster_col].dropna().unique())

    print(f"[CLUSTER] Unique clusters in ROW 1: {len(unique_clusters)}")

    print(f"[CLUSTER] Clusters: {unique_clusters}")

    

    cluster_info["n_clusters"] = len(unique_clusters)

    cluster_info["cluster_list"] = unique_clusters

    

    return cluster_col, cluster_info





# ==================== TRAIN/VAL SPLIT ====================



def create_group_train_val_split(

    X: np.ndarray,

    y: np.ndarray,

    groups: np.ndarray,

    segment_ids: np.ndarray,

    val_size: float = 0.10,

    random_state: int = 42,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:

    """

    Create train/val split grouped by cluster_id (no cluster in both train and val).

    

    Returns:

        X_train, X_val, y_train, y_val, split_info

    """

    print(f"[SPLIT] Creating group-based train/val split...")

    print(f"[SPLIT] Val size: {val_size:.0%}, Random state: {random_state}")

    

    unique_groups = np.unique(groups)

    n_groups = len(unique_groups)

    print(f"[SPLIT] Total unique clusters: {n_groups}")

    

    # GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    

    X_train, X_val = X[train_idx], X[val_idx]

    y_train, y_val = y[train_idx], y[val_idx]

    

    # Get cluster_ids for train and val

    train_clusters = sorted(set(groups[train_idx]))

    val_clusters = sorted(set(groups[val_idx]))

    

    # Verify no cluster overlap

    cluster_overlap = set(train_clusters) & set(val_clusters)

    if cluster_overlap:

        raise ValueError(f"LEAKAGE: {len(cluster_overlap)} clusters in both train and val!")

    

    # Verify no segment_id overlap (mask_path)

    train_segments = set(segment_ids[train_idx])

    val_segments = set(segment_ids[val_idx])

    segment_overlap = train_segments & val_segments

    if segment_overlap:

        raise ValueError(f"LEAKAGE: {len(segment_overlap)} segments in both train and val!")

    

    print(f"[SPLIT] Train: {len(train_idx):,} samples from {len(train_clusters)} clusters")

    print(f"[SPLIT] Val: {len(val_idx):,} samples from {len(val_clusters)} clusters")

    print(f"[SPLIT] Cluster overlap: 0 (verified)")

    print(f"[SPLIT] Segment overlap: 0 (verified)")

    

    split_info = {

        "train_samples": len(train_idx),

        "val_samples": len(val_idx),

        "train_clusters": train_clusters,

        "val_clusters": val_clusters,

        "n_train_clusters": len(train_clusters),

        "n_val_clusters": len(val_clusters),

        "n_train_segments": len(train_segments),

        "n_val_segments": len(val_segments),

        "val_size": val_size,

        "random_state": random_state,

        "split_method": "GroupShuffleSplit",

        "group_column": "cluster_id",

    }

    

    return X_train, X_val, y_train, y_val, split_info





# ==================== TRAINING ====================



def train_logistic_regression(

    X_train: np.ndarray,

    y_train: np.ndarray,

    X_val: np.ndarray,

    y_val: np.ndarray,

    n_classes: int,

    crack_class_idx: int,

    class_names: List[str] = None,

) -> Tuple[LogisticRegression, StandardScaler, Dict]:

    """

    Train Logistic Regression classifier with StandardScaler preprocessing.

    

    Uses balanced class_weight to handle CRACK class imbalance.

    

    Args:

        X_train: Training features

        y_train: Training labels (multi-class)

        X_val: Validation features (held-out clusters)

        y_val: Validation labels (multi-class)

        n_classes: Number of classes

        crack_class_idx: Index of the CRACK class for metric reporting

        class_names: List of class names (for per-class PR-AUC reporting)

    

    Returns:

        model: Trained LogisticRegression

        scaler: Fitted StandardScaler

        train_info: Dict with training information

    """

    print(f"\n[TRAIN] Training Logistic Regression...")

    print(f"[TRAIN] Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")

    print(f"[TRAIN] Features: {X_train.shape[1]}")

    print(f"[TRAIN] Classes: {n_classes}")

    print(f"[TRAIN] CRACK class index: {crack_class_idx}")

    print(f"[TRAIN] Using class_weight='balanced' for imbalance handling")

    

    # Standardize features (critical for Logistic Regression)

    print(f"[TRAIN] Fitting StandardScaler on training data...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    

    print(f"[TRAIN] Scaled features: mean≈{X_train_scaled.mean():.4f}, std≈{X_train_scaled.std():.4f}")

    

    # Build model

    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    

    print(f"[TRAIN] Hyperparameters: {LOGISTIC_REGRESSION_PARAMS}")

    

    # Train

    start_time = time.time()

    print(f"\n[TRAIN] Starting training...")

    

    model.fit(X_train_scaled, y_train)

    

    train_time = time.time() - start_time

    

    # Report convergence

    n_iter_used = model.n_iter_[0] if hasattr(model, 'n_iter_') else None

    print(f"\n[TRAIN] Training completed in {train_time:.1f}s")

    if n_iter_used is not None:

        print(f"[TRAIN] Iterations used: {n_iter_used} / {LOGISTIC_REGRESSION_PARAMS['max_iter']}")

    

    # Quick val metrics for reporting

    y_pred_val = model.predict(X_val_scaled)

    y_prob_val = model.predict_proba(X_val_scaled)

    

    acc = accuracy_score(y_val, y_pred_val)

    macro_f1 = f1_score(y_val, y_pred_val, average='macro', zero_division=0)

    

    # Build class name labels

    _class_names = class_names if class_names is not None else [f"Class_{i}" for i in range(n_classes)]

    

    # Per-class PR-AUC (one-vs-rest)

    per_class_prauc = {}

    print(f"\n[TRAIN] Val Accuracy: {acc:.4f}")

    print(f"[TRAIN] Val Macro-F1: {macro_f1:.4f}")

    print(f"[TRAIN] Per-class PR-AUC (one-vs-rest):")

    for cls_idx in range(n_classes):

        cls_name = _class_names[cls_idx]

        y_true_binary = (y_val == cls_idx).astype(int)

        n_pos = y_true_binary.sum()

        if n_pos == 0 or n_pos == len(y_val):

            prauc = float('nan')

        else:

            try:

                prauc = average_precision_score(y_true_binary, y_prob_val[:, cls_idx])

            except Exception:

                prauc = float('nan')

        per_class_prauc[cls_name] = round(prauc, 4) if not np.isnan(prauc) else None

        cls_f1 = f1_score(y_val, y_pred_val, labels=[cls_idx], average='micro', zero_division=0)

        marker = " ◄ CRACK" if cls_idx == crack_class_idx else ""

        prauc_str = f"{prauc:.4f}" if not np.isnan(prauc) else "N/A"

        print(f"  [{cls_idx}] {cls_name:20s}  F1={cls_f1:.4f}  PR-AUC={prauc_str}{marker}")

    

    # L1 sparsity analysis: how many features were selected (non-zero coefficients)

    total_features = model.coef_.shape[1]

    # A feature is "selected" if at least one class has a non-zero coefficient for it

    nonzero_per_feature = np.any(model.coef_ != 0, axis=0)

    n_selected_features = int(nonzero_per_feature.sum())

    n_zeroed_features = total_features - n_selected_features

    

    print(f"\n[L1 SPARSITY] Total features: {total_features}")

    print(f"[L1 SPARSITY] Features selected (non-zero coef in at least 1 class): {n_selected_features}")

    print(f"[L1 SPARSITY] Features zeroed out: {n_zeroed_features}")

    print(f"[L1 SPARSITY] Sparsity: {n_zeroed_features/total_features*100:.1f}%")

    

    # Per-class sparsity

    for cls_idx in range(model.coef_.shape[0]):

        n_nonzero = int(np.count_nonzero(model.coef_[cls_idx]))

        print(f"[L1 SPARSITY]   Class {cls_idx}: {n_nonzero}/{total_features} non-zero coefficients")

    

    train_info = {

        "train_time_seconds": train_time,

        "n_iterations_used": int(n_iter_used) if n_iter_used is not None else None,

        "max_iter": LOGISTIC_REGRESSION_PARAMS["max_iter"],

        "crack_class_idx": crack_class_idx,

        "val_accuracy": float(acc),

        "val_macro_f1": float(macro_f1),

        "per_class_prauc": per_class_prauc,

        "l1_sparsity": {

            "total_features": total_features,

            "selected_features": n_selected_features,

            "zeroed_features": n_zeroed_features,

            "sparsity_pct": round(n_zeroed_features / total_features * 100, 2),

        },

        "hyperparameters": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 

                           for k, v in LOGISTIC_REGRESSION_PARAMS.items()},

        "scaler": "StandardScaler",

        "scaler_mean": scaler.mean_.tolist(),

        "scaler_scale": scaler.scale_.tolist(),

    }

    

    return model, scaler, train_info





# ==================== EVALUATION ====================



def evaluate_model(

    model: LogisticRegression,

    scaler: StandardScaler,

    X_val: np.ndarray,

    y_val: np.ndarray,

    class_names: List[str],

    crack_class_idx: int,

) -> Tuple[Dict, pd.DataFrame]:

    """

    Evaluate model on validation set.

    

    Returns:

        metrics: Dict with all evaluation metrics

        predictions_df: DataFrame with y_true, y_pred, y_prob_crack

    """

    print(f"\n[EVAL] Evaluating on validation set...")

    

    # Scale validation data

    X_val_scaled = scaler.transform(X_val)

    

    # Predictions

    y_pred = model.predict(X_val_scaled)

    y_prob = model.predict_proba(X_val_scaled)

    

    # Get CRACK class probabilities

    if crack_class_idx < y_prob.shape[1]:

        y_prob_crack = y_prob[:, crack_class_idx]

    else:

        y_prob_crack = np.zeros(len(y_val))

    

    # Standard metrics

    acc = accuracy_score(y_val, y_pred)

    macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

    weighted_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

    

    # Per-class metrics

    n_classes = len(class_names)

    labels = list(range(n_classes))

    prec_per_class = precision_score(y_val, y_pred, average=None, labels=labels, zero_division=0)

    rec_per_class = recall_score(y_val, y_pred, average=None, labels=labels, zero_division=0)

    f1_per_class = f1_score(y_val, y_pred, average=None, labels=labels, zero_division=0)

    

    # CRACK-specific metrics

    crack_prec = prec_per_class[crack_class_idx] if crack_class_idx < len(prec_per_class) else 0.0

    crack_rec = rec_per_class[crack_class_idx] if crack_class_idx < len(rec_per_class) else 0.0

    crack_f1 = f1_per_class[crack_class_idx] if crack_class_idx < len(f1_per_class) else 0.0

    

    # CRACK ROC-AUC and PR-AUC (one-vs-rest)

    y_true_binary = (y_val == crack_class_idx).astype(int)

    crack_count = y_true_binary.sum()

    

    crack_roc_auc = np.nan

    crack_pr_auc = np.nan

    

    if crack_count > 0 and y_prob_crack.sum() > 0:

        try:

            crack_roc_auc = roc_auc_score(y_true_binary, y_prob_crack)

            crack_pr_auc = average_precision_score(y_true_binary, y_prob_crack)

        except Exception as e:

            print(f"[EVAL] Warning: AUC calculation failed: {e}")

    

    # Print summary

    print(f"[EVAL] Accuracy: {acc:.4f}")

    print(f"[EVAL] Macro-F1: {macro_f1:.4f}")

    print(f"[EVAL] Weighted-F1: {weighted_f1:.4f}")

    print(f"[EVAL] CRACK Precision: {crack_prec:.4f}")

    print(f"[EVAL] CRACK Recall: {crack_rec:.4f}")

    print(f"[EVAL] CRACK F1: {crack_f1:.4f}")

    print(f"[EVAL] CRACK ROC-AUC: {crack_roc_auc:.4f}")

    print(f"[EVAL] CRACK PR-AUC: {crack_pr_auc:.4f}")

    

    # Class distribution in predictions

    print(f"\n[EVAL] Prediction distribution:")

    for i, name in enumerate(class_names):

        true_count = (y_val == i).sum()

        pred_count = (y_pred == i).sum()

        print(f"  {name}: true={true_count:,}, pred={pred_count:,}")

    

    # Logistic Regression coefficients summary

    if hasattr(model, 'coef_'):

        print(f"\n[EVAL] Model coefficients shape: {model.coef_.shape}")

        print(f"[EVAL] CRACK class coefficient L2 norm: {np.linalg.norm(model.coef_[crack_class_idx]):.4f}")

    

    metrics = {

        "accuracy": float(acc),

        "macro_f1": float(macro_f1),

        "weighted_f1": float(weighted_f1),

        "CRACK_precision": float(crack_prec),

        "CRACK_recall": float(crack_rec),

        "CRACK_f1": float(crack_f1),

        "CRACK_roc_auc": float(crack_roc_auc) if not np.isnan(crack_roc_auc) else None,

        "CRACK_pr_auc": float(crack_pr_auc) if not np.isnan(crack_pr_auc) else None,

        "crack_count_val": int(crack_count),

        "per_class_precision": {class_names[i]: float(prec_per_class[i]) for i in range(n_classes)},

        "per_class_recall": {class_names[i]: float(rec_per_class[i]) for i in range(n_classes)},

        "per_class_f1": {class_names[i]: float(f1_per_class[i]) for i in range(n_classes)},

    }

    

    # Create predictions DataFrame

    predictions_df = pd.DataFrame({

        "y_true": y_val,

        "y_pred": y_pred,

        "y_prob_crack": y_prob_crack,

    })

    

    return metrics, predictions_df





# ==================== SAVE ARTIFACTS ====================



def save_artifacts(

    out_dir: Path,

    model: LogisticRegression,

    scaler: StandardScaler,

    class_mapping: Dict[str, int],

    class_names: List[str],

    split_info: Dict,

    filter_info: Dict,

    cluster_info: Dict,

    train_info: Dict,

    metrics: Dict,

    predictions_df: pd.DataFrame,

    config: DatasetConfig,

    balance_mode: str,

    feature_names: List[str],

) -> None:

    """Save all artifacts for reproducibility and inference."""

    

    save_dir = ensure_dir(out_dir / config.name / balance_mode)

    print(f"\n[SAVE] Saving artifacts to: {save_dir}")

    

    # 1. Save model

    model_path = save_dir / "logistic_regression_model.pkl"

    if joblib is not None:

        joblib.dump(model, model_path)

    else:

        import pickle

        with open(model_path, 'wb') as f:

            pickle.dump(model, f)

    print(f"[SAVE] Model saved: {model_path}")

    

    # 2. Save scaler

    scaler_path = save_dir / "standard_scaler.pkl"

    if joblib is not None:

        joblib.dump(scaler, scaler_path)

    else:

        import pickle

        with open(scaler_path, 'wb') as f:

            pickle.dump(scaler, f)

    print(f"[SAVE] Scaler saved: {scaler_path}")

    

    # 3. Save class_mapping

    class_mapping_path = save_dir / "class_mapping.json"

    with open(class_mapping_path, 'w') as f:

        json.dump(class_mapping, f, indent=2)

    print(f"[SAVE] Class mapping saved: {class_mapping_path}")

    

    # 4. Save split_manifest

    manifest = {

        "timestamp": TIMESTAMP,

        "dataset": config.name,

        "balance_mode": balance_mode,

        "model_type": "LogisticRegression",

        "row_filter": filter_info,

        "cluster_info": cluster_info,

        "train_val_split": split_info,

        "training": train_info,

        "preprocessing": {

            "wl_min": 450,

            "wl_max": 925,

            "apply_snv": True,

            "remove_outliers": False,

            "n_features": len(feature_names),

            "scaler": "StandardScaler",

        },

        "class_names": class_names,

        "class_mapping": class_mapping,

    }

    manifest_path = save_dir / "split_manifest.json"

    with open(manifest_path, 'w') as f:

        json.dump(manifest, f, indent=2, default=str)

    print(f"[SAVE] Split manifest saved: {manifest_path}")

    

    # 5. Save validation predictions

    predictions_path = save_dir / "val_predictions.csv"

    predictions_df.to_csv(predictions_path, index=False)

    print(f"[SAVE] Validation predictions saved: {predictions_path}")

    

    # 6. Save metrics

    metrics_path = save_dir / "metrics.json"

    with open(metrics_path, 'w') as f:

        json.dump(metrics, f, indent=2)

    print(f"[SAVE] Metrics saved: {metrics_path}")

    

    # 7. Save feature names

    features_path = save_dir / "feature_names.json"

    with open(features_path, 'w') as f:

        json.dump(feature_names, f, indent=2)

    print(f"[SAVE] Feature names saved: {features_path}")

    

    # 8. Save classification report

    all_labels = list(range(len(class_names)))

    report = classification_report(

        predictions_df["y_true"],

        predictions_df["y_pred"],

        labels=all_labels,

        target_names=class_names,

        output_dict=True,

        zero_division=0,

    )

    report_path = save_dir / "classification_report.json"

    with open(report_path, 'w') as f:

        json.dump(report, f, indent=2)

    print(f"[SAVE] Classification report saved: {report_path}")

    

    # 9. Save model coefficients (useful for interpretability)

    if hasattr(model, 'coef_'):

        coef_df = pd.DataFrame(

            model.coef_,

            index=class_names,

            columns=feature_names,

        )

        coef_path = save_dir / "model_coefficients.csv"

        coef_df.to_csv(coef_path)

        print(f"[SAVE] Model coefficients saved: {coef_path}")

        

        # Also save intercept

        intercept_df = pd.DataFrame({

            "class": class_names,

            "intercept": model.intercept_,

        })

        intercept_path = save_dir / "model_intercepts.csv"

        intercept_df.to_csv(intercept_path, index=False)

        print(f"[SAVE] Model intercepts saved: {intercept_path}")





# ==================== MAIN ====================



def main():

    """Main training function."""

    print("=" * 80)

    print("Logistic Regression ROW 1 Training Script")

    print("All 159 spectral features (450-925 nm)")

    print("=" * 80)

    print(f"Timestamp: {TIMESTAMP}")

    print(f"Output directory: {EXPERIMENT_DIR}")

    print()

    

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    

    # Calculate total configurations

    total_configs = len(DATASET_CONFIGS) * 2  # 2 balance modes per dataset

    config_idx = 0

    

    # Create main progress bar

    pbar_main = tqdm(total=total_configs, desc="Overall Progress", position=0, leave=True,

                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    

    for config in DATASET_CONFIGS:

        print(f"\n{'='*80}")

        print(f"DATASET: {config.name.upper()}")

        print(f"{'='*80}")

        

        # Check if CSV exists

        if not config.csv_path.exists():

            print(f"[ERROR] CSV not found: {config.csv_path}")

            pbar_main.update(2)  # Skip both balance modes

            continue

        

        # Load and filter to ROW 1

        try:

            df_row1, filter_info = load_and_filter_row1(config)

        except Exception as e:

            print(f"[ERROR] Failed to load/filter data: {e}")

            import traceback

            traceback.print_exc()

            pbar_main.update(2)  # Skip both balance modes

            continue

        

        # Add cluster_id column

        try:

            cluster_col, cluster_info = add_cluster_id_column(df_row1)

        except Exception as e:

            print(f"[ERROR] Failed to add cluster_id: {e}")

            import traceback

            traceback.print_exc()

            pbar_main.update(2)  # Skip both balance modes

            continue

        

        # Run for both balanced and unbalanced

        for is_balanced in [False, True]:

            balance_mode = "Balanced" if is_balanced else "Unbalanced"

            config_idx += 1

            

            # Update progress bar description

            pbar_main.set_description(f"[{config_idx}/{total_configs}] {config.name}/{balance_mode}")

            

            print(f"\n{'-'*60}")

            print(f"--- Training: {config.name} / {balance_mode} ({config_idx}/{total_configs}) ---")

            print(f"{'-'*60}")

            

            # Preprocess data (same as benchmark)

            try:

                data = preprocess_multiclass_dataset(

                    df_row1.copy(),

                    wl_min=450,

                    wl_max=925,

                    apply_snv=True,

                    remove_outliers=False,

                    balanced=is_balanced,

                    label_col=config.target_col,

                    hs_dir_col="hs_dir",

                    segment_col="mask_path",

                    cap_class=config.crack_identifier,

                    max_samples_per_class=MAX_SAMPLES_PER_CLASS,

                    seed=RANDOM_STATE,

                )

            except Exception as e:

                print(f"[ERROR] Preprocessing failed: {e}")

                import traceback

                traceback.print_exc()

                pbar_main.update(1)

                continue

            

            X, y = data.X, data.y

            class_names = data.class_names

            class_mapping = data.class_mapping

            n_classes = len(class_names)

            

            print(f"[INFO] Preprocessed: {X.shape[0]:,} samples, {X.shape[1]} features, {n_classes} classes")

            print(f"[INFO] Classes: {class_names}")

            print(f"[INFO] Class mapping: {class_mapping}")

            

            # Class distribution

            print(f"[INFO] Class distribution:")

            for name, idx in class_mapping.items():

                count = (y == idx).sum()

                print(f"  {name}: {count:,} ({count/len(y)*100:.1f}%)")

            

            # Determine CRACK class index

            if config.is_3class:

                crack_class_name = str(config.crack_identifier)

                crack_class_idx = class_mapping.get(crack_class_name, 0)

            else:

                crack_class_name = str(config.crack_identifier).upper()

                crack_class_idx = class_mapping.get(crack_class_name, 0)

            

            print(f"[INFO] CRACK class: '{crack_class_name}' (idx={crack_class_idx})")

            

            # Create train/val split by cluster_id

            try:

                X_train, X_val, y_train, y_val, split_info = create_group_train_val_split(

                    X=X,

                    y=y,

                    groups=data.groups,

                    segment_ids=data.segment_ids,

                    val_size=VAL_SIZE,

                    random_state=RANDOM_STATE,

                )

            except Exception as e:

                print(f"[ERROR] Split failed: {e}")

                import traceback

                traceback.print_exc()

                pbar_main.update(1)

                continue

            

            # Print class distribution in train/val

            print(f"\n[INFO] Train class distribution:")

            for name, idx in class_mapping.items():

                count = (y_train == idx).sum()

                print(f"  {name}: {count:,}")

            

            print(f"[INFO] Val class distribution:")

            for name, idx in class_mapping.items():

                count = (y_val == idx).sum()

                print(f"  {name}: {count:,}")

            

            # Train Logistic Regression

            try:

                model, scaler, train_info = train_logistic_regression(

                    X_train, y_train,

                    X_val, y_val,

                    n_classes=n_classes,

                    crack_class_idx=crack_class_idx,

                    class_names=class_names,

                )

            except Exception as e:

                print(f"[ERROR] Training failed: {e}")

                import traceback

                traceback.print_exc()

                pbar_main.update(1)

                continue

            

            # Evaluate

            try:

                metrics, predictions_df = evaluate_model(

                    model, scaler, X_val, y_val,

                    class_names=class_names,

                    crack_class_idx=crack_class_idx,

                )

            except Exception as e:

                print(f"[ERROR] Evaluation failed: {e}")

                import traceback

                traceback.print_exc()

                pbar_main.update(1)

                continue

            

            # Save artifacts

            try:

                save_artifacts(

                    out_dir=EXPERIMENT_DIR,

                    model=model,

                    scaler=scaler,

                    class_mapping=class_mapping,

                    class_names=class_names,

                    split_info=split_info,

                    filter_info=filter_info,

                    cluster_info=cluster_info,

                    train_info=train_info,

                    metrics=metrics,

                    predictions_df=predictions_df,

                    config=config,

                    balance_mode=balance_mode,

                    feature_names=data.feature_names,

                )

            except Exception as e:

                print(f"[ERROR] Saving artifacts failed: {e}")

                import traceback

                traceback.print_exc()

                pbar_main.update(1)

                continue

            

            print(f"\n[SUCCESS] Completed: {config.name} / {balance_mode}")

            pbar_main.update(1)

    

    pbar_main.close()

    print(f"\n{'='*80}")

    print(f"Training complete!")

    print(f"Results saved to: {EXPERIMENT_DIR}")

    print(f"{'='*80}")





if __name__ == "__main__":

    main()

