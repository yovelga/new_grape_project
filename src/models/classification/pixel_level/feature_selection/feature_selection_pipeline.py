"""
feature_selection_pipeline.py

Feature Selection Pipeline for Hyperspectral Channel Reduction.

This script performs a two-step feature selection process to reduce 150 hyperspectral
channels to 10 optimal channels for the XGBoost model:

Step 1: SHAP-based Initial Filtering (Ranking)
    - Train XGBoost on all 150 channels
    - Use SHAP TreeExplainer to calculate Mean Absolute SHAP values
    - Select top 40 channels

Step 2: RFECV-based Active Reduction
    - Run sklearn RFECV with XGBoost on the 40 selected channels
    - Use LOGO CV splits to prevent information leakage
    - Identify optimal number of features (targeting ~10)

Outputs:
    - SHAP Summary Plot for top 40 channels
    - RFECV Curve showing performance vs number of channels
    - CSV file with final top 10 channels

================================================================================
NOTE FOR THESIS:
    For the thesis multiclass-only feature selection with SNV normalization,
    use the dedicated script: feature_selection_multiclass.py
    
    That script provides:
    - MULTICLASS label setup ONLY (no binary or 3-class variants)
    - SNV (Standard Normal Variate) normalization applied per-sample
    - LOGO folds from unified_experiment_pipeline_acc.py
    - K-sweep evaluation (100 → 1) with extensive artifact logging
    - PR-AUC(CRACK) as primary metric (one-vs-rest)
    
    Run quick test:
        python quick_test_feature_selection.py --mode multiclass
    
    Run full experiment:
        python feature_selection_multiclass.py --top_k_shap 100 --k_max 100
================================================================================

Author: Feature Selection Pipeline
Date: January 2026
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Add project root to Python path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[5]  # Navigate up to Grape_Project
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
print(f"[DEBUG] Project root: {_PROJECT_ROOT}")

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Import preprocessing and CV splits from unified pipeline
from src.preprocessing.spectral_preprocessing import (
    preprocess_multiclass_dataset,
    PreprocessedData,
)

# Import SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Install with: pip install shap")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== CONFIGURATION ====================

# Dataset path
CSV_PATH_MULTICLASS = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_multiclass_2026-01-16.csv")

# Output directory - dedicated folder for feature selection experiments
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path(rf"C:\Users\yovel\Desktop\Grape_Project\results\feature_selection\shap_rfecv_{TIMESTAMP}")

# Feature selection parameters
RANDOM_STATE = 42
TOP_K_SHAP = 40  # Number of channels to select in Step 1 (SHAP ranking)
MIN_FEATURES_TO_SELECT = 1  # Minimum features for RFECV
RFECV_STEP = 1  # Step size for RFECV
TARGET_FEATURES = 10  # Target number of final features

# XGBoost parameters - GPU accelerated
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",  # Use GPU for training
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# ==================== HELPER FUNCTIONS ====================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_wavelength_columns(df: pd.DataFrame) -> List[str]:
    """Extract wavelength column names from DataFrame."""
    wl_cols = [c for c in df.columns if c.replace('.', '').replace('-', '').isdigit()]
    return sorted(wl_cols, key=lambda x: float(x))


# ==================== DOMAIN-AWARE CV SPLITS ====================
# (Copied from unified_experiment_pipeline_acc.py for consistency)

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from typing import NamedTuple


class FoldInfo(NamedTuple):
    """Information about a CV fold."""
    fold_idx: int
    grape_group_held_out: str
    n_train: int
    n_test: int
    train_grape_groups: List[str]
    test_grape_group: str
    non_grape_train_images: List[str]
    non_grape_test_images: List[str]


def create_domain_aware_cv_splits(
    y: np.ndarray,
    original_labels: np.ndarray,
    segment_ids: np.ndarray,
    image_ids: np.ndarray,
    grape_classes: set,
    crack_class_idx: int,
    regular_class_idx: int,
    random_state: int = 42,
    non_grape_holdout_frac: float = 0.20,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, FoldInfo]], Dict]:
    """
    Create domain-aware CV splits with LOGO on grape samples and fixed holdout for non-grape.
    
    This function ensures no information leakage between pixels from the same object (Group).
    
    Returns:
        List of (train_idx, test_idx, fold_info) tuples
        Split manifest dict for reproducibility
    """
    n_samples = len(y)
    all_indices = np.arange(n_samples)

    # Determine grape samples
    if isinstance(list(grape_classes)[0], str):
        is_grape = np.array([str(lbl).upper() in grape_classes for lbl in original_labels])
    else:
        is_grape = np.isin(y, list(grape_classes))

    grape_indices = all_indices[is_grape]
    non_grape_indices = all_indices[~is_grape]

    print(f"[SPLIT] Total samples: {n_samples}")
    print(f"[SPLIT] Grape samples: {len(grape_indices)}")
    print(f"[SPLIT] Non-grape samples: {len(non_grape_indices)}")

    # Fixed non-grape holdout
    if len(non_grape_indices) > 0:
        non_grape_image_ids = image_ids[non_grape_indices]
        non_grape_segment_ids = segment_ids[non_grape_indices]
        
        gss = GroupShuffleSplit(n_splits=1, test_size=non_grape_holdout_frac, random_state=random_state)
        dummy_y = np.zeros(len(non_grape_indices))

        try:
            ng_train_idx_local, ng_test_idx_local = next(gss.split(
                non_grape_indices, dummy_y, groups=non_grape_image_ids
            ))
            grouping_used = "image (hs_dir)"
        except ValueError:
            gss_segment = GroupShuffleSplit(n_splits=1, test_size=non_grape_holdout_frac, random_state=random_state)
            ng_train_idx_local, ng_test_idx_local = next(gss_segment.split(
                non_grape_indices, dummy_y, groups=non_grape_segment_ids
            ))
            grouping_used = "segment (mask_path)"

        non_grape_train_global = non_grape_indices[ng_train_idx_local]
        non_grape_test_global = non_grape_indices[ng_test_idx_local]
        
        ng_train_images = set(image_ids[non_grape_train_global])
        ng_test_images = set(image_ids[non_grape_test_global])
        
        # Verify no segment leakage
        ng_train_segments = set(segment_ids[non_grape_train_global])
        ng_test_segments = set(segment_ids[non_grape_test_global])
        segment_intersection = ng_train_segments & ng_test_segments
        
        if segment_intersection:
            raise ValueError(f"LEAKAGE DETECTED: {len(segment_intersection)} segments in both non-grape train and test!")
    else:
        non_grape_train_global = np.array([], dtype=int)
        non_grape_test_global = np.array([], dtype=int)
        ng_train_images = set()
        ng_test_images = set()
        grouping_used = "N/A"

    # Separate grape samples into CRACK and REGULAR
    crack_mask = y[grape_indices] == crack_class_idx
    regular_mask = y[grape_indices] == regular_class_idx
    
    crack_indices = grape_indices[crack_mask]
    regular_indices = grape_indices[regular_mask]
    
    crack_image_ids = image_ids[crack_indices]
    regular_image_ids = image_ids[regular_indices]
    
    unique_crack_images = np.unique(crack_image_ids)
    unique_regular_images = np.unique(regular_image_ids)
    
    print(f"[SPLIT] CRACK samples: {len(crack_indices)} from {len(unique_crack_images)} images")
    print(f"[SPLIT] REGULAR samples: {len(regular_indices)} from {len(unique_regular_images)} images")
    
    # 80/20 split for REGULAR samples (fixed across folds)
    if len(regular_indices) > 0:
        regular_segment_ids = segment_ids[regular_indices]
        
        gss_regular = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        dummy_y = np.zeros(len(regular_indices))
        reg_train_local, reg_test_local = next(gss_regular.split(
            regular_indices, dummy_y, groups=regular_segment_ids
        ))
        
        regular_train_global = regular_indices[reg_train_local]
        regular_test_global = regular_indices[reg_test_local]
        
        # Verify no segment leakage
        reg_train_segments = set(segment_ids[regular_train_global])
        reg_test_segments = set(segment_ids[regular_test_global])
        if reg_train_segments & reg_test_segments:
            raise ValueError("LEAKAGE DETECTED: REGULAR segments overlap train/test!")
        
        print(f"[SPLIT] REGULAR 80/20 split: {len(regular_train_global)} train, {len(regular_test_global)} test")
    else:
        regular_train_global = np.array([], dtype=int)
        regular_test_global = np.array([], dtype=int)
    
    # LOGO on CRACK samples only
    logo = LeaveOneGroupOut()
    
    folds = []
    fold_manifest = {
        "non_grape_holdout": {
            "grouping": grouping_used,
            "train_images": sorted(list(ng_train_images)) if ng_train_images else [],
            "test_images": sorted(list(ng_test_images)) if ng_test_images else [],
            "train_count": len(non_grape_train_global),
            "test_count": len(non_grape_test_global),
        },
        "regular_holdout": {
            "train_count": len(regular_train_global),
            "test_count": len(regular_test_global),
        },
        "folds": [],
    }

    for fold_idx, (crack_train_local, crack_test_local) in enumerate(
        logo.split(crack_indices, y[crack_indices], groups=crack_image_ids)
    ):
        crack_train_global = crack_indices[crack_train_local]
        crack_test_global = crack_indices[crack_test_local]

        # Combine: CRACK (LOGO) + REGULAR (fixed 80/20) + non-grape (fixed 80/20)
        train_idx = np.concatenate([crack_train_global, regular_train_global, non_grape_train_global])
        test_idx = np.concatenate([crack_test_global, regular_test_global, non_grape_test_global])

        # Verify no segment leakage
        train_segments = set(segment_ids[train_idx])
        test_segments = set(segment_ids[test_idx])
        segment_leak = train_segments & test_segments

        if segment_leak:
            raise ValueError(f"LEAKAGE DETECTED in fold {fold_idx}: {len(segment_leak)} segments overlap!")

        held_out_group = str(np.unique(crack_image_ids[crack_test_local])[0])
        train_crack_groups = list(np.unique(crack_image_ids[crack_train_local]))

        fold_info = FoldInfo(
            fold_idx=fold_idx,
            grape_group_held_out=held_out_group,
            n_train=len(train_idx),
            n_test=len(test_idx),
            train_grape_groups=train_crack_groups,
            test_grape_group=held_out_group,
            non_grape_train_images=sorted(list(ng_train_images)),
            non_grape_test_images=sorted(list(ng_test_images)),
        )

        folds.append((train_idx, test_idx, fold_info))
        
        fold_manifest["folds"].append({
            "fold_idx": fold_idx,
            "crack_image_held_out": held_out_group,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    print(f"[SPLIT] Created {len(folds)} LOGO folds on CRACK samples only")
    return folds, fold_manifest


def create_logo_cv_generator(folds: List[Tuple[np.ndarray, np.ndarray, FoldInfo]]):
    """
    Create a CV generator compatible with sklearn's cross-validation functions.
    
    Yields (train_indices, test_indices) tuples.
    """
    for train_idx, test_idx, fold_info in folds:
        yield train_idx, test_idx


# ==================== STEP 1: SHAP-BASED FEATURE RANKING ====================

def step1_shap_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray, FoldInfo]],
    top_k: int = 40,
    output_dir: Path = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Step 1: Train XGBoost on all features and use SHAP to rank them.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels
        feature_names: List of feature/channel names
        folds: CV folds from create_domain_aware_cv_splits
        top_k: Number of top features to select
        output_dir: Directory to save SHAP plots
    
    Returns:
        selected_indices: Indices of top-k features
        selected_names: Names of top-k features
        shap_importance_df: DataFrame with SHAP importance for all features
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required for Step 1. Install with: pip install shap")
    
    print("\n" + "=" * 60)
    print("STEP 1: SHAP-BASED FEATURE RANKING")
    print("=" * 60)
    
    # Train XGBoost on full data to get SHAP values
    print(f"[SHAP] Training XGBoost on all {X.shape[1]} features...")
    
    # Use first fold's training data for SHAP analysis (representative)
    train_idx, test_idx, fold_info = folds[0]
    X_train, y_train = X[train_idx], y[train_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train XGBoost
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    print("[SHAP] Computing SHAP values...")
    
    # Use a subsample for SHAP computation if dataset is large
    max_shap_samples = 5000
    if X_train_scaled.shape[0] > max_shap_samples:
        np.random.seed(RANDOM_STATE)
        shap_sample_idx = np.random.choice(X_train_scaled.shape[0], max_shap_samples, replace=False)
        X_shap = X_train_scaled[shap_sample_idx]
    else:
        X_shap = X_train_scaled
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # For multi-class, shap_values is a list of arrays (one per class)
    # or a 3D array (samples, features, classes)
    # Compute mean absolute SHAP across all classes and samples
    if isinstance(shap_values, list):
        # List of arrays: one array per class, each is (samples, features)
        shap_values_stacked = np.abs(np.stack(shap_values, axis=0)).mean(axis=(0, 1))
    elif len(shap_values.shape) == 3:
        # 3D array: (samples, features, classes)
        shap_values_stacked = np.abs(shap_values).mean(axis=(0, 2))
    else:
        # 2D array: (samples, features) - binary or regression
        shap_values_stacked = np.abs(shap_values).mean(axis=0)
    
    # Ensure it's 1D
    shap_values_stacked = np.array(shap_values_stacked).flatten()
    
    print(f"[DEBUG] SHAP values shape: {shap_values_stacked.shape}, features: {len(feature_names)}")
    
    # Create importance DataFrame
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': shap_values_stacked,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    shap_importance_df['rank'] = range(1, len(shap_importance_df) + 1)
    
    print(f"\n[SHAP] Top 10 features by Mean Absolute SHAP:")
    print(shap_importance_df.head(10).to_string(index=False))
    
    # Select top-k features
    top_k_features = shap_importance_df.head(top_k)['feature'].tolist()
    selected_indices = [feature_names.index(f) for f in top_k_features]
    
    print(f"\n[SHAP] Selected top {top_k} features for RFECV")
    
    # Save SHAP Summary Plot
    if output_dir:
        ensure_dir(output_dir)
        
        # SHAP Summary Plot (beeswarm)
        print("[SHAP] Saving SHAP summary plot...")
        plt.figure(figsize=(12, 10))
        
        # Get indices for top-k features
        top_k_idx = [feature_names.index(f) for f in top_k_features]
        
        if isinstance(explainer.shap_values(X_shap[:100]), list):
            # Multi-class: use first class or aggregate
            shap_values_plot = explainer.shap_values(X_shap[:min(500, len(X_shap))])
            # Take absolute mean across classes for plotting
            shap_values_agg = np.mean([np.abs(sv) for sv in shap_values_plot], axis=0)
        else:
            shap_values_agg = np.abs(explainer.shap_values(X_shap[:min(500, len(X_shap))]))
        
        # Create bar plot of mean absolute SHAP values
        fig, ax = plt.subplots(figsize=(10, 12))
        top_k_importance = shap_importance_df.head(top_k)
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_k_importance)))
        
        bars = ax.barh(
            range(len(top_k_importance)), 
            top_k_importance['mean_abs_shap'].values[::-1],
            color=colors[::-1]
        )
        ax.set_yticks(range(len(top_k_importance)))
        ax.set_yticklabels(top_k_importance['feature'].values[::-1])
        ax.set_xlabel('Mean Absolute SHAP Value')
        ax.set_title(f'Top {top_k} Features by SHAP Importance')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'shap_top40_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save SHAP importance to CSV
        shap_importance_df.to_csv(output_dir / 'shap_all_features_importance.csv', index=False)
        top_k_importance.to_csv(output_dir / 'shap_top40_features.csv', index=False)
        
        print(f"[SHAP] Saved plots and CSVs to {output_dir}")
    
    return np.array(selected_indices), top_k_features, shap_importance_df


# ==================== STEP 2: RFECV-BASED FEATURE REDUCTION ====================

def step2_rfecv_reduction(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray, FoldInfo]],
    min_features: int = 1,
    step: int = 1,
    scoring: str = 'f1_macro',
    output_dir: Path = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Step 2: Use RFECV to find optimal feature subset.
    
    Args:
        X: Feature matrix (already subset to top-k from Step 1)
        y: Target labels
        feature_names: List of feature/channel names for the subset
        folds: CV folds from create_domain_aware_cv_splits
        min_features: Minimum number of features to select
        step: Number of features to remove at each iteration
        scoring: Scoring metric for RFECV
        output_dir: Directory to save RFECV plots
    
    Returns:
        selected_indices: Indices of optimal features (relative to input X)
        selected_names: Names of optimal features
        rfecv_results_df: DataFrame with RFECV scores per number of features
    """
    print("\n" + "=" * 60)
    print("STEP 2: RFECV-BASED FEATURE REDUCTION")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create XGBoost estimator
    model = XGBClassifier(**XGBOOST_PARAMS)
    
    # Create CV generator from folds
    cv_splits = list(create_logo_cv_generator(folds))
    
    print(f"[RFECV] Running RFECV with {len(cv_splits)} LOGO folds...")
    print(f"[RFECV] Starting features: {X.shape[1]}")
    print(f"[RFECV] Min features: {min_features}, Step: {step}")
    print(f"[RFECV] Scoring: {scoring}")
    
    # Define scorer
    if scoring == 'f1_macro':
        scorer = make_scorer(f1_score, average='macro')
    elif scoring == 'accuracy':
        scorer = make_scorer(accuracy_score)
    else:
        scorer = scoring
    
    # Run RFECV
    rfecv = RFECV(
        estimator=model,
        step=step,
        cv=cv_splits,
        scoring=scorer,
        min_features_to_select=min_features,
        n_jobs=-1,
        verbose=1,
    )
    
    rfecv.fit(X_scaled, y)
    
    # Get results
    n_features_optimal = rfecv.n_features_
    selected_mask = rfecv.support_
    selected_indices = np.where(selected_mask)[0]
    selected_names = [feature_names[i] for i in selected_indices]
    
    # Get feature rankings
    feature_ranking = rfecv.ranking_
    
    print(f"\n[RFECV] Optimal number of features: {n_features_optimal}")
    print(f"[RFECV] Selected features: {selected_names}")
    
    # Create results DataFrame
    # cv_results_ contains mean and std test scores for each number of features
    n_features_range = range(min_features, X.shape[1] + 1)
    
    # Handle different sklearn versions
    if hasattr(rfecv, 'cv_results_'):
        mean_scores = rfecv.cv_results_['mean_test_score']
        std_scores = rfecv.cv_results_['std_test_score']
    else:
        # Older sklearn versions
        mean_scores = rfecv.grid_scores_ if hasattr(rfecv, 'grid_scores_') else np.array([])
        std_scores = np.zeros_like(mean_scores)
    
    rfecv_results_df = pd.DataFrame({
        'n_features': list(n_features_range)[:len(mean_scores)],
        'mean_score': mean_scores,
        'std_score': std_scores,
    })
    
    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'ranking': feature_ranking,
        'selected': selected_mask,
    }).sort_values('ranking').reset_index(drop=True)
    
    # Save RFECV curve plot
    if output_dir:
        ensure_dir(output_dir)
        
        # RFECV Curve
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_values = rfecv_results_df['n_features'].values
        y_values = rfecv_results_df['mean_score'].values
        y_std = rfecv_results_df['std_score'].values
        
        ax.plot(x_values, y_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(x_values, y_values - y_std, y_values + y_std, alpha=0.2)
        
        # Mark optimal point
        optimal_idx = np.argmax(y_values)
        ax.axvline(x=x_values[optimal_idx], color='r', linestyle='--', 
                   label=f'Optimal: {x_values[optimal_idx]} features')
        ax.scatter([x_values[optimal_idx]], [y_values[optimal_idx]], 
                   color='r', s=100, zorder=5, marker='*')
        
        # Mark target (10 features) if different from optimal
        if TARGET_FEATURES in x_values and TARGET_FEATURES != x_values[optimal_idx]:
            target_idx = list(x_values).index(TARGET_FEATURES)
            ax.axvline(x=TARGET_FEATURES, color='g', linestyle=':', 
                       label=f'Target: {TARGET_FEATURES} features')
            ax.scatter([TARGET_FEATURES], [y_values[target_idx]], 
                       color='g', s=80, zorder=5, marker='s')
        
        ax.set_xlabel('Number of Features')
        ax.set_ylabel(f'CV Score ({scoring})')
        ax.set_title('RFECV: Performance vs Number of Features')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'rfecv_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save CSVs
        rfecv_results_df.to_csv(output_dir / 'rfecv_scores.csv', index=False)
        ranking_df.to_csv(output_dir / 'rfecv_feature_ranking.csv', index=False)
        
        print(f"[RFECV] Saved plots and CSVs to {output_dir}")
    
    return selected_indices, selected_names, rfecv_results_df, ranking_df


# ==================== GET TOP 10 FEATURES ====================

def get_top_n_features(
    rfecv_ranking_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Get top N features from RFECV ranking.
    
    Args:
        rfecv_ranking_df: DataFrame with 'feature' and 'ranking' columns
        n: Number of top features to select
    
    Returns:
        DataFrame with top N features
    """
    top_n = rfecv_ranking_df.nsmallest(n, 'ranking').reset_index(drop=True)
    top_n['final_rank'] = range(1, n + 1)
    return top_n


# ==================== MAIN PIPELINE ====================

def run_feature_selection_pipeline(
    csv_path: Path = CSV_PATH_MULTICLASS,
    output_dir: Path = OUTPUT_DIR,
    top_k_shap: int = TOP_K_SHAP,
    target_features: int = TARGET_FEATURES,
    max_samples: Optional[int] = None,  # For dry-run mode
):
    """
    Run the complete feature selection pipeline.
    
    Args:
        csv_path: Path to the multiclass CSV dataset
        output_dir: Directory to save all outputs
        top_k_shap: Number of features to select in SHAP step
        target_features: Target number of final features
        max_samples: Max samples for dry-run (None for full dataset)
    """
    print("=" * 80)
    print("HYPERSPECTRAL CHANNEL SELECTION PIPELINE")
    print("=" * 80)
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Top-K SHAP: {top_k_shap}")
    print(f"Target features: {target_features}")
    print(f"Random state: {RANDOM_STATE}")
    print("=" * 80)
    
    ensure_dir(output_dir)
    
    # ==================== LOAD DATA ====================
    print("\n[DATA] Loading dataset...")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} rows")
    
    # Apply sample cap for dry-run
    if max_samples and len(df) > max_samples:
        print(f"[DRY-RUN] Stratified sampling {max_samples} rows...")
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(
            df, 
            test_size=max_samples / len(df),
            stratify=df['label'],
            random_state=RANDOM_STATE
        )
        df = df.reset_index(drop=True)
    
    # ==================== PREPROCESS DATA ====================
    print("\n[PREPROCESS] Preprocessing dataset...")
    
    data = preprocess_multiclass_dataset(
        df,
        wl_min=450,
        wl_max=925,
        apply_snv=True,
        remove_outliers=False,
        balanced=False,  # No balancing for feature selection
        label_col='label',
        hs_dir_col='hs_dir',
        segment_col='mask_path',
        seed=RANDOM_STATE,
    )
    
    X = data.X
    y = data.y
    feature_names = data.feature_names
    
    print(f"[PREPROCESS] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[PREPROCESS] Classes: {data.class_names}")
    print(f"[PREPROCESS] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Get class indices
    crack_class_idx = data.class_mapping.get("CRACK", 3)
    regular_class_idx = data.class_mapping.get("REGULAR", 7)
    
    print(f"[PREPROCESS] CRACK class index: {crack_class_idx}")
    print(f"[PREPROCESS] REGULAR class index: {regular_class_idx}")
    
    # ==================== CREATE CV SPLITS ====================
    print("\n[CV] Creating LOGO CV splits...")
    
    folds, split_manifest = create_domain_aware_cv_splits(
        y=y,
        original_labels=data.original_labels,
        segment_ids=data.segment_ids,
        image_ids=data.image_ids,
        grape_classes=data.grape_class_indices,
        crack_class_idx=crack_class_idx,
        regular_class_idx=regular_class_idx,
        random_state=RANDOM_STATE,
        non_grape_holdout_frac=0.20,
    )
    
    # Save split manifest
    with open(output_dir / 'cv_split_manifest.json', 'w') as f:
        json.dump(split_manifest, f, indent=2)
    
    # ==================== STEP 1: SHAP RANKING ====================
    shap_output_dir = output_dir / 'step1_shap'
    
    selected_indices_step1, selected_names_step1, shap_df = step1_shap_ranking(
        X=X,
        y=y,
        feature_names=feature_names,
        folds=folds,
        top_k=top_k_shap,
        output_dir=shap_output_dir,
    )
    
    # Subset X to top-k features
    X_subset = X[:, selected_indices_step1]
    
    # ==================== STEP 2: RFECV ====================
    rfecv_output_dir = output_dir / 'step2_rfecv'
    
    selected_indices_step2, selected_names_step2, rfecv_scores_df, ranking_df = step2_rfecv_reduction(
        X=X_subset,
        y=y,
        feature_names=selected_names_step1,
        folds=folds,
        min_features=MIN_FEATURES_TO_SELECT,
        step=RFECV_STEP,
        scoring='f1_macro',
        output_dir=rfecv_output_dir,
    )
    
    # ==================== GET TOP 10 FEATURES ====================
    print("\n" + "=" * 60)
    print("FINAL RESULTS: TOP 10 FEATURES")
    print("=" * 60)
    
    top_10_df = get_top_n_features(ranking_df, n=target_features)
    
    # Map back to original feature indices
    top_10_df['original_index'] = top_10_df['feature'].apply(lambda x: feature_names.index(x))
    
    print(f"\nTop {target_features} Optimal Hyperspectral Channels:")
    print(top_10_df.to_string(index=False))
    
    # Save final top 10
    top_10_df.to_csv(output_dir / f'top_{target_features}_channels.csv', index=False)
    
    # ==================== SAVE PIPELINE SUMMARY ====================
    summary = {
        'pipeline_version': '1.0',
        'timestamp': TIMESTAMP,
        'random_state': RANDOM_STATE,
        'input_csv': str(csv_path),
        'total_samples': X.shape[0],
        'original_features': len(feature_names),
        'step1_shap_top_k': top_k_shap,
        'step2_rfecv_optimal': len(selected_names_step2),
        'final_top_n': target_features,
        'final_channels': top_10_df['feature'].tolist(),
        'final_channel_indices': top_10_df['original_index'].tolist(),
        'cv_n_folds': len(folds),
    }
    
    with open(output_dir / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)
    
    return top_10_df, summary


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Run pipeline
    # Set max_samples=1000 for a quick dry-run, None for full dataset
    top_10_df, summary = run_feature_selection_pipeline(
        csv_path=CSV_PATH_MULTICLASS,
        output_dir=OUTPUT_DIR,
        top_k_shap=TOP_K_SHAP,
        target_features=TARGET_FEATURES,
        max_samples=None,  # Set to e.g. 5000 for dry-run
    )
    
    print("\nFinal selected channels:")
    print(top_10_df[['final_rank', 'feature', 'original_index']].to_string(index=False))
