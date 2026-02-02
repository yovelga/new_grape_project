"""
train_xgboost_row1_segments_30_channels_best.py

XGBoost training script for pixel-level classification using ONLY ROW 1 data.
*** BEST MODEL: 30 channels selected via BFS (PR-AUC: 0.9948) ***

Pixel-level samples with group-based train/val split to prevent data leakage.

Split Level Configuration (SPLIT_LEVEL):
- "segment": Groups by mask_path (segment_id) - ensures all pixels from the same
  segment stay together. Segment overlap is ALWAYS zero (enforced).
  Cluster overlap MAY be >0 (clusters can have segments in both train and val).
- "cluster": Groups by cluster_id (row_cluster) - ensures all segments from the
  same cluster stay together. Both segment and cluster overlap are ALWAYS zero.

Leakage Prevention Rules:
- Segment overlap must ALWAYS be zero regardless of SPLIT_LEVEL.
- Cluster overlap must be zero when SPLIT_LEVEL="cluster".
- Cluster overlap is allowed (but reported) when SPLIT_LEVEL="segment".

Features:
- Filters data to ROW 1 only before preprocessing
- Uses GroupShuffleSplit with configurable grouping level for train/val split
- Supports both multiclass and 3class modes (same as benchmark)
- Saves model, class_mapping, split_manifest, predictions, and metrics
- Compatible with final test on ROW 2 (same preprocessing, label encoding)
- *** USES TOP 30 CHANNELS FROM BFS RESULTS ***

Based on BFS results from 2026-02-01_20-53-40:
Iteration 129: 30 features, CRACK PR-AUC = 0.9948461751921699 (BEST)

NO LOGO, NO CV - just one train/val split for final model training.

Output Structure:
  experiments/xgboost_row1/bfs_30_channels_best/<TIMESTAMP>/<dataset>/<balance_mode>/
"""

import sys
import os
import json
import pickle
import time
import traceback
import warnings
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Add project root to Python path for imports
# Path: src/models/classification/pixel_level/train_xgboost_row_1_for_full_image_test_20_80/train_xgboost_row1_segments_30_channels_best.py
# parents[0]=train_xgboost_row_1_for_full_image_test_20_80, [1]=pixel_level, [2]=classification, [3]=models, [4]=src, [5]=Grape_Project
_PROJECT_ROOT = Path(__file__).resolve().parents[5]  # Navigate up to Grape_Project
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== LOGGING SETUP ====================

class TeeLogger:
    """
    A class that duplicates stdout/stderr to both console and a log file.
    This ensures all print statements and errors are captured for debugging.
    """
    def __init__(self, log_file: Path, stream_type: str = "stdout"):
        self.log_file = log_file
        self.stream_type = stream_type
        self.terminal = sys.stdout if stream_type == "stdout" else sys.stderr
        self.log = None
        
    def __enter__(self):
        self.log = open(self.log_file, 'a', encoding='utf-8')
        if self.stream_type == "stdout":
            sys.stdout = self
        else:
            sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream_type == "stdout":
            sys.stdout = self.terminal
        else:
            sys.stderr = self.terminal
        if self.log:
            self.log.close()
        return False
    
    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()  # Ensure immediate write for debugging
    
    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()


def setup_logging(experiment_dir: Path) -> Tuple[Path, Path]:
    """
    Set up logging to capture all stdout and stderr to log files.
    
    Args:
        experiment_dir: The experiment output directory
        
    Returns:
        Tuple of (stdout_log_path, stderr_log_path)
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    stdout_log = experiment_dir / "run_stdout.log"
    stderr_log = experiment_dir / "run_stderr.log"
    
    # Write header to log files
    with open(stdout_log, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"XGBoost ROW 1 Training Log - BFS 30 CHANNELS (BEST)\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")
    
    with open(stderr_log, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"XGBoost ROW 1 Training Error Log - BFS 30 CHANNELS (BEST)\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")
    
    return stdout_log, stderr_log


# ==================== BFS CHANNEL SELECTION ====================

# Selected channels from BFS results (iteration 129 - best performance)
# CRACK PR-AUC = 0.9948461751921699
# Exact column names from best_features_prauc.json
BFS_SELECTED_CHANNELS = [
    "452.25nm", "455.16nm", "475.50nm", "513.40nm", "536.82nm", "545.62nm",
    "548.55nm", "551.49nm", "572.07nm", "580.90nm", "592.70nm", "607.46nm",
    "631.15nm", "634.11nm", "643.01nm", "726.53nm", "729.53nm", "732.53nm",
    "738.53nm", "741.53nm", "756.56nm", "759.56nm", "771.61nm", "777.64nm",
    "841.18nm", "892.95nm", "902.12nm", "911.30nm", "917.42nm", "920.48nm"
]

print(f"[BFS] Using {len(BFS_SELECTED_CHANNELS)} channels selected via BFS")
print(f"[BFS] Channels: {BFS_SELECTED_CHANNELS[:3]}...{BFS_SELECTED_CHANNELS[-3:]}")


# ==================== STARTUP LOGGING ====================

def print_global_run_config():
    """Print global run configuration for sanity checking."""
    print()
    print("=" * 50)
    print("            GLOBAL RUN CONFIG")
    print("=" * 50)
    print(f"Timestamp:      {TIMESTAMP}")
    print(f"Experiment dir: {EXPERIMENT_DIR.resolve()}")
    print(f"Split level:    {SPLIT_LEVEL}")
    print(f"Val size:       {VAL_SIZE:.0%}")
    print(f"Random state:   {RANDOM_STATE}")
    print(f"Datasets:       {[c.name for c in DATASET_CONFIGS]}")
    print(f"Balance modes:  ['Unbalanced', 'Balanced']")
    print(f"BFS channels:   {len(BFS_SELECTED_CHANNELS)}")
    bfs_preview = BFS_SELECTED_CHANNELS[:5] + ["..."] + BFS_SELECTED_CHANNELS[-5:]
    print(f"BFS preview:    {bfs_preview}")
    print("=" * 50)
    print()


def print_row_filter_summary(filter_info: Dict):
    """Print summary after row filtering."""
    print()
    print("-" * 50)
    print("       AFTER ROW FILTERING")
    print("-" * 50)
    print(f"Row detection:    {filter_info.get('row_detection_method', 'N/A')}")
    print(f"Original samples: {filter_info.get('original_rows', 'N/A'):,}")
    print(f"Row 1 samples:    {filter_info.get('row1_samples', 'N/A'):,}")
    print(f"Rows excluded:    {filter_info.get('rows_excluded', [])}")
    if filter_info.get('row_regex'):
        print(f"Row regex:        {filter_info.get('row_regex')}")
    print("-" * 50)
    print()


def print_segment_cluster_summary(
    segment_ids: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_info: Dict,
):
    """Print summary after segment/cluster extraction."""
    unique_segments = np.unique(segment_ids)
    unique_clusters = np.unique(cluster_ids)
    
    print()
    print("-" * 50)
    print("   AFTER CLUSTER / SEGMENT EXTRACTION")
    print("-" * 50)
    print(f"Unique segment_ids: {len(unique_segments)}")
    print(f"Unique cluster_ids: {len(unique_clusters)}")
    print(f"Segment sample (first 10): {list(unique_segments[:10])}")
    print(f"Cluster sample (first 10): {list(unique_clusters[:10])}")
    print(f"Cluster detection: {cluster_info.get('cluster_detection_method', 'N/A')}")
    print("-" * 50)
    print()


def print_pre_split_summary(
    X: np.ndarray,
    y: np.ndarray,
    segment_ids: np.ndarray,
    cluster_ids: np.ndarray,
    split_level: str,
    class_names: List[str],
):
    """Print summary before train/val split for final verification."""
    unique_segments = np.unique(segment_ids)
    unique_clusters = np.unique(cluster_ids)
    
    print()
    print("=" * 50)
    print("        BEFORE TRAIN/VAL SPLIT")
    print("=" * 50)
    print(f"Split level:        {split_level}")
    print(f"Grouping by:        {'segment_id' if split_level == 'segment' else 'cluster_id'}")
    print(f"Total samples:      {len(X):,}")
    print(f"Total features:     {X.shape[1]}")
    print(f"Unique segments:    {len(unique_segments)}")
    print(f"Unique clusters:    {len(unique_clusters)}")
    print()
    print("Class distribution:")
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        pct = 100.0 * count / len(y)
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    print()
    print("Leakage prevention rules:")
    print("  - Segment overlap: MUST be 0 (always enforced)")
    if split_level == "cluster":
        print("  - Cluster overlap: MUST be 0 (enforced for cluster split)")
    else:
        print("  - Cluster overlap: ALLOWED (not enforced for segment split)")
    print("=" * 50)
    print()


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
CSV_PATH_MULTICLASS = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_multiclass_2026-01-16.csv")
CSV_PATH_3CLASS = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_3class_2026-01-16.csv")

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

# Organized output directory structure:
# experiments/xgboost_row1/bfs_30_channels_best/<TIMESTAMP>/<dataset>/<balance_mode>/
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENTS_ROOT = Path(r"C:\Users\yovel\Desktop\Grape_Project\experiments")
RUN_NAME = "xgboost_row1/bfs_30_channels_best"
EXPERIMENT_DIR = EXPERIMENTS_ROOT / RUN_NAME / TIMESTAMP

RANDOM_STATE = 42
VAL_SIZE = 0.20  # 20% validation (80/20 split)

# ==================== SPLIT LEVEL CONFIGURATION ====================
# Controls how samples are grouped during train/val split to prevent leakage:
# - "segment": Group by segment_id (mask_path) - segment overlap=0 enforced,
#              cluster overlap allowed but reported.
# - "cluster": Group by cluster_id (row_cluster) - both segment and cluster
#              overlap=0 enforced.
SPLIT_LEVEL = "segment"  # Options: "segment" or "cluster"

# ==================== BALANCING SETTINGS ====================
# Maximum samples per class (same as benchmark)
MAX_SAMPLES_PER_CLASS = 50000

# ==================== XGBoost HYPERPARAMETERS ====================
# Easy to tweak - all XGBoost settings in one place
# Optimized for Row 1 training with PR-AUC monitoring
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
}

EARLY_STOPPING_ROUNDS = 50


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


# ==================== CHANNEL SELECTION ====================

def select_bfs_channels(features: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Select only the channels identified by BFS as optimal.
    
    Args:
        features: Full feature matrix (n_samples, n_features)
        feature_names: List of all feature names (wavelengths)
        
    Returns:
        selected_features: Feature matrix with selected channels only
        selected_feature_names: Names of selected features
    """
    print(f"[CHANNEL_SELECT] Input features shape: {features.shape}")
    print(f"[CHANNEL_SELECT] Total feature names: {len(feature_names)}")
    
    # Create mapping from feature name to index
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    # Find indices of BFS selected channels using exact string matching
    selected_indices = []
    selected_names = []
    missing_channels = []
    
    for channel_name in BFS_SELECTED_CHANNELS:
        if channel_name in name_to_idx:
            selected_indices.append(name_to_idx[channel_name])
            selected_names.append(channel_name)
        else:
            missing_channels.append(channel_name)
    
    if missing_channels:
        print(f"[CHANNEL_SELECT] WARNING: Could not find channels: {missing_channels}")
        print(f"[CHANNEL_SELECT] Available channels (sample): {feature_names[:10]}")
    
    print(f"[CHANNEL_SELECT] Successfully matched {len(selected_indices)}/{len(BFS_SELECTED_CHANNELS)} channels")
    
    if len(selected_indices) == 0:
        raise ValueError("No BFS channels could be matched to feature names!")
    
    # Select features
    selected_features = features[:, selected_indices]
    
    print(f"[CHANNEL_SELECT] Output features shape: {selected_features.shape}")
    print(f"[CHANNEL_SELECT] Selected channels: {selected_names[:5]}...")
    
    return selected_features, selected_names


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

# Maximum number of IDs to store in manifest samples (for debugging)
_MAX_SAMPLE_IDS = 50


def create_group_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    segment_ids: np.ndarray,
    cluster_ids: np.ndarray,
    split_level: str = "segment",
    val_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create train/val split grouped by either segment_id or cluster_id.
    
    Leakage Prevention Rules:
    - Segment overlap must ALWAYS be zero (a segment cannot appear in both sets).
    - Cluster overlap must be zero when split_level="cluster".
    - Cluster overlap is allowed when split_level="segment" (but reported).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        segment_ids: Segment identifiers for each sample (mask_path based)
        cluster_ids: Cluster identifiers for each sample (row_cluster based)
        split_level: "segment" or "cluster" - determines grouping level
        val_size: Fraction of data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, y_train, y_val, split_info
    
    Raises:
        ValueError: If split_level is invalid, segment overlap > 0, or
                   cluster overlap > 0 when split_level="cluster"
    """
    # Validate split_level and select grouping variable
    if split_level == "segment":
        group_ids = segment_ids
        group_name = "segment_id"
    elif split_level == "cluster":
        group_ids = cluster_ids
        group_name = "cluster_id"
    else:
        raise ValueError(f"Invalid split_level='{split_level}'. Must be 'segment' or 'cluster'.")
    
    print(f"[SPLIT] Creating group-based train/val split BY {split_level.upper()}...")
    print(f"[SPLIT] Grouping by: {group_name}")
    print(f"[SPLIT] Val size: {val_size:.0%}, Random state: {random_state}")
    
    # Count unique groups
    unique_groups = np.unique(group_ids)
    unique_segments = np.unique(segment_ids)
    unique_clusters = np.unique(cluster_ids)
    print(f"[SPLIT] Total unique groups ({group_name}): {len(unique_groups)}")
    print(f"[SPLIT] Total unique segments: {len(unique_segments)}")
    print(f"[SPLIT] Total unique clusters: {len(unique_clusters)}")
    
    # GroupShuffleSplit using the selected group_ids
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=group_ids))
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Compute segment overlap (MUST always be zero)
    train_segments = set(segment_ids[train_idx])
    val_segments = set(segment_ids[val_idx])
    segment_overlap = train_segments & val_segments
    n_segment_overlap = len(segment_overlap)
    
    # Compute cluster overlap
    train_clusters = set(cluster_ids[train_idx])
    val_clusters = set(cluster_ids[val_idx])
    cluster_overlap = train_clusters & val_clusters
    n_cluster_overlap = len(cluster_overlap)
    
    # Print split statistics
    print(f"[SPLIT] Train: {len(train_idx):,} samples")
    print(f"[SPLIT] Val: {len(val_idx):,} samples")
    print(f"[SPLIT] Train segments: {len(train_segments)}, Val segments: {len(val_segments)}")
    print(f"[SPLIT] Train clusters: {len(train_clusters)}, Val clusters: {len(val_clusters)}")
    print(f"[SPLIT] Segment overlap: {n_segment_overlap} (must be 0)")
    print(f"[SPLIT] Cluster overlap: {n_cluster_overlap} {'(must be 0)' if split_level == 'cluster' else '(allowed)'}")
    
    # ========== LEAKAGE CHECKS ==========
    
    # 1. Segment overlap must ALWAYS be zero (critical - no segment in both sets)
    if n_segment_overlap > 0:
        overlap_sample = sorted(segment_overlap)[:_MAX_SAMPLE_IDS]
        raise ValueError(
            f"SEGMENT LEAKAGE: {n_segment_overlap} segments appear in both train and val! "
            f"This is NEVER allowed. Sample overlapping segments: {overlap_sample}"
        )
    
    # 2. Cluster overlap check depends on split_level
    if split_level == "cluster" and n_cluster_overlap > 0:
        overlap_sample = sorted(cluster_overlap)[:_MAX_SAMPLE_IDS]
        raise ValueError(
            f"CLUSTER LEAKAGE: {n_cluster_overlap} clusters appear in both train and val! "
            f"This is not allowed when split_level='cluster'. Sample overlapping clusters: {overlap_sample}"
        )
    
    # Build compact split_info (no huge ID lists)
    # Store counts + small samples for debugging
    train_segments_sorted = sorted(train_segments)
    val_segments_sorted = sorted(val_segments)
    train_clusters_sorted = sorted(train_clusters)
    val_clusters_sorted = sorted(val_clusters)
    
    split_info = {
        "split_level": split_level,
        "group_column": group_name,
        "split_method": "GroupShuffleSplit",
        "val_size": val_size,
        "random_state": random_state,
        # Sample counts
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        # Segment info (counts + samples)
        "n_train_segments": len(train_segments),
        "n_val_segments": len(val_segments),
        "train_segments_sample": train_segments_sorted[:_MAX_SAMPLE_IDS],
        "val_segments_sample": val_segments_sorted[:_MAX_SAMPLE_IDS],
        "n_segment_overlap": n_segment_overlap,  # Always 0 (enforced)
        # Cluster info (counts + samples)
        "n_train_clusters": len(train_clusters),
        "n_val_clusters": len(val_clusters),
        "train_clusters_sample": train_clusters_sorted[:_MAX_SAMPLE_IDS],
        "val_clusters_sample": val_clusters_sorted[:_MAX_SAMPLE_IDS],
        "n_cluster_overlap": n_cluster_overlap,
    }
    
    # Add note about cluster overlap when split_level="segment"
    if split_level == "segment" and n_cluster_overlap > 0:
        cluster_overlap_sample = sorted(cluster_overlap)[:_MAX_SAMPLE_IDS]
        split_info["cluster_overlap_note"] = (
            f"{n_cluster_overlap} clusters have segments in both train and val. "
            f"This is allowed when split_level='segment' because we group by segment, not cluster."
        )
        split_info["cluster_overlap_sample"] = cluster_overlap_sample
        print(f"[SPLIT] Note: {n_cluster_overlap} clusters span train/val (allowed with segment-level split)")
    
    print(f"[SPLIT] Leakage checks passed: segment_overlap=0, cluster_overlap={'0 (enforced)' if split_level == 'cluster' else f'{n_cluster_overlap} (allowed)'}")
    
    return X_train, X_val, y_train, y_val, split_info


# ==================== TRAINING ====================

class BoosterWrapper:
    """Wrapper to make xgb.Booster work with sklearn-like interface.
    Defined at module level for pickle compatibility.
    """
    def __init__(self, booster, n_classes, best_iteration, best_score):
        self._Booster = booster
        self.n_classes_ = n_classes
        self._classes = np.arange(n_classes)
        self.best_iteration = best_iteration
        self.best_score = best_score
        
    @property
    def classes_(self):
        return self._classes
    
    def predict(self, X):
        """Predict class labels."""
        dmatrix = xgb.DMatrix(X)
        probs = self._Booster.predict(dmatrix)
        if probs.ndim == 1:
            probs = probs.reshape(-1, self.n_classes_)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        dmatrix = xgb.DMatrix(X)
        probs = self._Booster.predict(dmatrix)
        if probs.ndim == 1:
            probs = probs.reshape(-1, self.n_classes_)
        return probs
    
    def get_booster(self):
        return self._Booster


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    crack_class_idx: int,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> Tuple[XGBClassifier, Dict]:
    """
    Train XGBoost classifier with early stopping based on CRACK PR-AUC.
    
    Uses the native XGBoost API with a custom evaluation function to maximize
    CRACK class PR-AUC for early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels (multi-class)
        X_val: Validation features (held-out clusters)
        y_val: Validation labels (multi-class)
        n_classes: Number of classes
        crack_class_idx: Index of the CRACK class for PR-AUC calculation
        early_stopping_rounds: Rounds for early stopping
    
    Returns:
        model: Trained XGBClassifier
        train_info: Dict with training information
    """
    from sklearn.metrics import average_precision_score
    
    print(f"\n[TRAIN] Training XGBoost...")
    print(f"[TRAIN] Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
    print(f"[TRAIN] Features: {X_train.shape[1]} (BFS selected)")
    print(f"[TRAIN] Classes: {n_classes}")
    print(f"[TRAIN] CRACK class index: {crack_class_idx}")
    print(f"[TRAIN] Early stopping: MAXIMIZE CRACK PR-AUC (patience={early_stopping_rounds})")
    
    # Compute balanced sample weights to handle CRACK class imbalance
    sample_weights = compute_sample_weight('balanced', y_train)
    print(f"[TRAIN] Using balanced sample weights (min={sample_weights.min():.4f}, max={sample_weights.max():.4f})")
    
    # Create DMatrix for native XGBoost API
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Prepare parameters for native API
    params = {
        "max_depth": XGBOOST_PARAMS["max_depth"],
        "learning_rate": XGBOOST_PARAMS["learning_rate"],  # 'eta' in native API
        "subsample": XGBOOST_PARAMS["subsample"],
        "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
        "reg_lambda": XGBOOST_PARAMS["reg_lambda"],
        "min_child_weight": XGBOOST_PARAMS["min_child_weight"],
        "gamma": XGBOOST_PARAMS["gamma"],
        "tree_method": XGBOOST_PARAMS["tree_method"],
        "seed": XGBOOST_PARAMS["random_state"],
        "objective": "multi:softprob",
        "num_class": n_classes,
        "nthread": -1,
    }
    
    print(f"[TRAIN] Hyperparameters: {params}")
    
    # Custom evaluation function for CRACK PR-AUC
    def crack_prauc_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Custom evaluation metric: PR-AUC for CRACK class.
        Returns (metric_name, value, higher_is_better)
        """
        labels = dtrain.get_label().astype(int)
        
        # preds shape: (n_samples, n_classes) for multi:softprob
        if preds.ndim == 1:
            # Reshape if flattened
            preds = preds.reshape(-1, n_classes)
        
        # Extract CRACK class probabilities
        y_prob_crack = preds[:, crack_class_idx]
        
        # Binary labels: 1 if CRACK, 0 otherwise
        y_true_binary = (labels == crack_class_idx).astype(int)
        
        # Calculate PR-AUC
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            score = 0.0
        else:
            try:
                score = average_precision_score(y_true_binary, y_prob_crack)
            except Exception:
                score = 0.0
        
        return "crack_prauc", float(score)
    
    # Custom callback for detailed progress logging with all metrics
    class ProgressCallback(xgb.callback.TrainingCallback):
        def __init__(self, total_rounds, X_val, y_val, crack_idx, print_every=10):
            self.total_rounds = total_rounds
            self.X_val = X_val
            self.y_val = y_val
            self.crack_idx = crack_idx
            self.print_every = print_every
            self.best_prauc = 0.0
            self.best_iter = 0
            self.start_time = time.time()
            self.dval = xgb.DMatrix(X_val)
            
        def after_iteration(self, model, epoch, evals_log):
            if epoch == 0 or (epoch + 1) % self.print_every == 0 or epoch == self.total_rounds - 1:
                # Get predictions on validation set
                val_probs = model.predict(self.dval)
                if val_probs.ndim == 1:
                    val_probs = val_probs.reshape(-1, model.num_class())
                
                val_preds = np.argmax(val_probs, axis=1)
                
                # Calculate metrics
                acc = accuracy_score(self.y_val, val_preds)
                
                # CRACK-specific metrics
                y_true_crack = (self.y_val == self.crack_idx).astype(int)
                y_prob_crack = val_probs[:, self.crack_idx]
                
                # Handle edge cases
                if y_true_crack.sum() == 0:
                    crack_prec = crack_rec = crack_f1 = crack_prauc = 0.0
                else:
                    crack_mask = val_preds == self.crack_idx
                    if crack_mask.sum() == 0:
                        crack_prec = 0.0
                    else:
                        crack_prec = (y_true_crack[crack_mask]).mean()
                    
                    crack_rec = (val_preds[y_true_crack == 1] == self.crack_idx).mean() if y_true_crack.sum() > 0 else 0.0
                    
                    if crack_prec + crack_rec == 0:
                        crack_f1 = 0.0
                    else:
                        crack_f1 = 2 * crack_prec * crack_rec / (crack_prec + crack_rec)
                    
                    try:
                        crack_prauc = average_precision_score(y_true_crack, y_prob_crack)
                    except:
                        crack_prauc = 0.0
                
                # Update best
                if crack_prauc > self.best_prauc:
                    self.best_prauc = crack_prauc
                    self.best_iter = epoch
                
                elapsed = time.time() - self.start_time
                print(f"[{epoch+1:4d}/{self.total_rounds}] "
                      f"Acc={acc:.4f} | "
                      f"CRACK: P={crack_prec:.4f}, R={crack_rec:.4f}, F1={crack_f1:.4f}, PR-AUC={crack_prauc:.6f} | "
                      f"Best: {self.best_prauc:.6f}@{self.best_iter+1} | "
                      f"{elapsed:.1f}s")
            
            return False  # Don't stop training
    
    # Train with native API
    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}
    
    start_time = time.time()
    
    print(f"\n[TRAIN] Starting training with CRACK PR-AUC early stopping...")
    print(f"[TRAIN] Will stop if no improvement in {early_stopping_rounds} rounds")
    print(f"[TRAIN] Progress updates every 10 rounds")
    print(f"[TRAIN] Metrics: Acc=Accuracy, P=Precision, R=Recall, F1=F1-Score, PR-AUC=Area Under PR Curve")
    print()
    
    # Create progress callback
    progress_cb = ProgressCallback(
        total_rounds=XGBOOST_PARAMS["n_estimators"],
        X_val=X_val,
        y_val=y_val,
        crack_idx=crack_class_idx,
        print_every=10
    )
    
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=XGBOOST_PARAMS["n_estimators"],
        evals=evals,
        custom_metric=crack_prauc_eval,
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,  # We want to MAXIMIZE PR-AUC
        evals_result=evals_result,
        verbose_eval=False,  # We use our custom callback instead
        callbacks=[progress_cb],
    )
    
    train_time = time.time() - start_time
    
    # Get best iteration and score
    best_iteration = booster.best_iteration
    best_score = booster.best_score
    
    print(f"\n[TRAIN] Training completed in {train_time:.1f}s")
    print(f"[TRAIN] Best iteration: {best_iteration}")
    print(f"[TRAIN] Best CRACK PR-AUC: {best_score:.6f}")
    
    # Create wrapper using module-level class (for pickle compatibility)
    model = BoosterWrapper(booster, n_classes, best_iteration, best_score)
    
    train_info = {
        "train_time_seconds": train_time,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "best_score_metric": "crack_prauc",
        "n_estimators_used": best_iteration,
        "early_stopping_rounds": early_stopping_rounds,
        "crack_class_idx": crack_class_idx,
        "hyperparameters": params,
        "evals_result": {
            "train_prauc": evals_result.get("train", {}).get("crack_prauc", []),
            "val_prauc": evals_result.get("val", {}).get("crack_prauc", []),
        },
        "bfs_info": {
            "n_selected_channels": len(BFS_SELECTED_CHANNELS),
            "selected_channels": BFS_SELECTED_CHANNELS,
            "bfs_performance": "PR-AUC: 0.9948461751921699 (BEST)",
        },
    }
    
    return model, train_info


# ==================== EVALUATION ====================

def evaluate_model(
    model: XGBClassifier,
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
    
    # Predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    
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
        "bfs_info": {
            "n_selected_channels": len(BFS_SELECTED_CHANNELS),
            "bfs_expected_performance": "PR-AUC: 0.9948461751921699 (BEST)",
        },
    }
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "y_true": y_val,
        "y_pred": y_pred,
        "y_prob_crack": y_prob_crack,
    })
    
    return metrics, predictions_df


# ==================== SAVE ARTIFACTS ====================

def write_readme(out_dir: Path, config: DatasetConfig, balance_mode: str, split_info: Dict) -> None:
    """
    Write a README.txt with run summary for easy reference.
    """
    readme_path = out_dir / "README.txt"
    
    lines = [
        "="*60,
        "XGBoost ROW 1 Training - BFS 30 Channels (BEST)",
        "="*60,
        "",
        f"Timestamp: {TIMESTAMP}",
        f"Dataset: {config.name}",
        f"Balance Mode: {balance_mode}",
        "",
        "--- Split Configuration ---",
        f"SPLIT_LEVEL: {SPLIT_LEVEL}",
        f"Group Column: {split_info.get('group_column', 'N/A')}",
        f"Val Size: {VAL_SIZE:.0%}",
        f"Random State: {RANDOM_STATE}",
        "",
        "--- Sample Counts ---",
        f"Train Samples: {split_info.get('train_samples', 'N/A'):,}",
        f"Val Samples: {split_info.get('val_samples', 'N/A'):,}",
        f"Train Segments: {split_info.get('n_train_segments', 'N/A')}",
        f"Val Segments: {split_info.get('n_val_segments', 'N/A')}",
        f"Train Clusters: {split_info.get('n_train_clusters', 'N/A')}",
        f"Val Clusters: {split_info.get('n_val_clusters', 'N/A')}",
        "",
        "--- Leakage Checks ---",
        f"Segment Overlap: {split_info.get('n_segment_overlap', 'N/A')} (must be 0)",
        f"Cluster Overlap: {split_info.get('n_cluster_overlap', 'N/A')} {'(must be 0)' if SPLIT_LEVEL == 'cluster' else '(allowed)'}",
        "",
        "--- BFS Channel Selection ---",
        f"Number of Channels: {len(BFS_SELECTED_CHANNELS)}",
        f"Expected PR-AUC: 0.9948461751921699 (BEST)",
        "",
        "--- Files ---",
        "xgboost_model.pkl - Trained model",
        "class_mapping.json - Label encoding",
        "split_manifest.json - Full split details",
        "metrics.json - Evaluation metrics",
        "val_predictions.csv - Validation predictions",
        "feature_names.json - Selected channel names",
        "bfs_channel_selection.json - BFS details",
        "classification_report.json - Per-class metrics",
        "="*60,
    ]
    
    with open(readme_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"[SAVE] README saved: {readme_path}")


def save_artifacts(
    out_dir: Path,
    model: XGBClassifier,
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
    
    # 0. Write README.txt with run summary (including metrics)
    write_readme(save_dir, config, balance_mode, split_info, metrics)
    
    # 1. Save model
    model_path = save_dir / "xgboost_model.pkl"
    if joblib is not None:
        joblib.dump(model, model_path)
    else:
        # Fallback to pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print(f"[SAVE] Model saved: {model_path}")
    
    # 2. Save class_mapping
    class_mapping_path = save_dir / "class_mapping.json"
    with open(class_mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"[SAVE] Class mapping saved: {class_mapping_path}")
    
    # 3. Save split_manifest (compact - split_info already has samples not full lists)
    # Remove cluster_list from cluster_info to keep manifest small
    cluster_info_compact = {k: v for k, v in cluster_info.items() if k != "cluster_list"}
    cluster_info_compact["cluster_sample"] = cluster_info.get("cluster_list", [])[:_MAX_SAMPLE_IDS]
    
    manifest = {
        "timestamp": TIMESTAMP,
        "dataset": config.name,
        "balance_mode": balance_mode,
        "model_type": "BFS_30_channels_best",
        "split_level": SPLIT_LEVEL,
        "row_filter": filter_info,
        "cluster_info": cluster_info_compact,
        "train_val_split": split_info,  # Already compact with samples not full lists
        "training": train_info,
        "preprocessing": {
            "wl_min": 450,
            "wl_max": 925,
            "apply_snv": True,
            "remove_outliers": False,
            "n_features": len(feature_names),
            "bfs_channel_selection": {
                "n_selected": len(BFS_SELECTED_CHANNELS),
                "selected_channels": BFS_SELECTED_CHANNELS,
                "bfs_iteration": 129,
                "expected_performance": "PR-AUC: 0.9948461751921699 (BEST)",
            },
        },
        "class_names": class_names,
        "class_mapping": class_mapping,
    }
    manifest_path = save_dir / "split_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[SAVE] Split manifest saved: {manifest_path}")
    
    # 4. Save validation predictions
    predictions_path = save_dir / "val_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[SAVE] Validation predictions saved: {predictions_path}")
    
    # 5. Save metrics
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics saved: {metrics_path}")
    
    # 6. Save feature names
    features_path = save_dir / "feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"[SAVE] Feature names saved: {features_path}")
    
    # 7. Save BFS channel selection details
    bfs_path = save_dir / "bfs_channel_selection.json"
    bfs_info = {
        "source": "BFS results from 2026-02-01_20-53-40",
        "iteration": 129,
        "n_features": 30,
        "expected_crack_pr_auc": 0.9948461751921699,
        "performance_rank": "BEST (100%)",
        "selected_channels_nm": BFS_SELECTED_CHANNELS,
        "channel_range": {
            "min": min(BFS_SELECTED_CHANNELS),
            "max": max(BFS_SELECTED_CHANNELS),
        },
    }
    with open(bfs_path, 'w') as f:
        json.dump(bfs_info, f, indent=2)
    print(f"[SAVE] BFS channel selection saved: {bfs_path}")
    
    # 8. Save classification report
    # Use labels parameter to handle cases where not all classes are present in val
    all_labels = list(range(len(class_names)))
    report = classification_report(
        predictions_df["y_true"],
        predictions_df["y_pred"],
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,  # Handle classes with no samples
    )
    report_path = save_dir / "classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[SAVE] Classification report saved: {report_path}")


# ==================== MAIN ====================

def main():
    """Main training function."""
    # Set up logging FIRST to capture all output
    stdout_log, stderr_log = setup_logging(EXPERIMENT_DIR)
    
    with TeeLogger(stdout_log, "stdout"), TeeLogger(stderr_log, "stderr"):
        _run_training()


def _run_training():
    """Internal training function - runs within logging context."""
    print("=" * 80)
    print("XGBoost ROW 1 Training Script - BFS 30 CHANNELS (BEST)")
    print("=" * 80)
    
    # Print global run config for sanity checking
    print_global_run_config()
    
    print(f"Creating output directory: {EXPERIMENT_DIR}")
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
            print_row_filter_summary(filter_info)
        except Exception as e:
            print(f"[ERROR] Failed to load/filter data: {e}")
            # traceback.print_exc()
            pbar_main.update(2)  # Skip both balance modes
            continue
        
        # Add cluster_id column
        try:
            cluster_col, cluster_info = add_cluster_id_column(df_row1)
        except Exception as e:
            print(f"[ERROR] Failed to add cluster_id: {e}")
            # traceback.print_exc()
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
                print(f"[PREPROCESS] Starting preprocessing...")
                
                # Use preprocess_multiclass_dataset to get preprocessed data
                preprocessed: PreprocessedData = preprocess_multiclass_dataset(
                    df_row1,
                    wl_min=450,
                    wl_max=925,
                    apply_snv=True,
                    remove_outliers=False,
                    balanced=is_balanced,
                    label_col=config.target_col,
                    max_samples_per_class=MAX_SAMPLES_PER_CLASS if is_balanced else None,
                    seed=RANDOM_STATE,
                )
                
                X = preprocessed.X
                y = preprocessed.y
                feature_names = preprocessed.feature_names
                class_mapping = preprocessed.class_mapping
                class_names = preprocessed.class_names
                
                print(f"[PREPROCESS] Preprocessed shape: {X.shape}")
                print(f"[PREPROCESS] Classes: {class_names}")
                print(f"[PREPROCESS] Features: {len(feature_names)} ({feature_names[0]} to {feature_names[-1]})")
                
                # Apply BFS channel selection
                X_selected, selected_feature_names = select_bfs_channels(X, feature_names)
                
                # Use segment_ids and groups from preprocessed data
                segment_ids = preprocessed.segment_ids
                cluster_ids = preprocessed.groups
                
                # Verify same length after preprocessing
                if len(X_selected) != len(segment_ids):
                    raise ValueError(f"Length mismatch: X={len(X_selected)}, segments={len(segment_ids)}")
                
                # Print segment/cluster extraction summary
                print_segment_cluster_summary(segment_ids, cluster_ids, cluster_info)
                
                # Find CRACK class index
                crack_class_idx = class_mapping.get(config.crack_identifier, class_mapping.get("CRACK", -1))
                if crack_class_idx == -1:
                    raise ValueError(f"CRACK class not found in class_mapping: {class_mapping}")
                
                print(f"[PREPROCESS] CRACK class index: {crack_class_idx}")
                print(f"[PREPROCESS] Class distribution:")
                for cls_name, cls_idx in class_mapping.items():
                    count = (y == cls_idx).sum()
                    print(f"  {cls_name}: {count:,} samples")
                
                # Print pre-split summary for final verification
                print_pre_split_summary(
                    X_selected, y, segment_ids, cluster_ids, SPLIT_LEVEL, class_names
                )
                
                # Train/Val split (using SPLIT_LEVEL to determine grouping)
                X_train, X_val, y_train, y_val, split_info = create_group_train_val_split(
                    X=X_selected,
                    y=y,
                    segment_ids=segment_ids,
                    cluster_ids=cluster_ids,
                    split_level=SPLIT_LEVEL,
                    val_size=VAL_SIZE,
                    random_state=RANDOM_STATE,
                )
                
                # Train model
                model, train_info = train_xgboost(
                    X_train, y_train, X_val, y_val, len(class_names), crack_class_idx
                )
                
                # Evaluate model
                metrics, predictions_df = evaluate_model(
                    model, X_val, y_val, class_names, crack_class_idx
                )
                
                # Save artifacts
                save_artifacts(
                    EXPERIMENT_DIR, model, class_mapping, class_names,
                    split_info, filter_info, cluster_info, train_info, metrics, predictions_df,
                    config, balance_mode, selected_feature_names
                )
                
                # Print CRACK metrics summary for easy reference
                print(f"\n" + "="*60)
                print(f"  CRACK METRICS SUMMARY - {config.name} / {balance_mode}")
                print(f"="*60)
                print(f"  CRACK Precision:  {metrics.get('CRACK_precision', 0):.4f}")
                print(f"  CRACK Recall:     {metrics.get('CRACK_recall', 0):.4f}")
                print(f"  CRACK F1:         {metrics.get('CRACK_f1', 0):.4f}")
                print(f"  CRACK PR-AUC:     {metrics.get('CRACK_pr_auc', 'N/A') if metrics.get('CRACK_pr_auc') else 'N/A'}")
                print(f"  CRACK ROC-AUC:    {metrics.get('CRACK_roc_auc', 'N/A') if metrics.get('CRACK_roc_auc') else 'N/A'}")
                print(f"="*60)
                
                print(f"[SUCCESS] Completed: {config.name} / {balance_mode}")
                
            except Exception as e:
                print(f"[ERROR] Failed processing {config.name}/{balance_mode}: {e}")
                traceback.print_exc()
            
            pbar_main.update(1)
    
    pbar_main.close()
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Results saved to: {EXPERIMENT_DIR}")
    print(f"Model: BFS 30 channels (BEST) - Expected PR-AUC: 0.9948")
    print(f"Log files saved to:")
    print(f"  - stdout: {EXPERIMENT_DIR / 'run_stdout.log'}")
    print(f"  - stderr: {EXPERIMENT_DIR / 'run_stderr.log'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()