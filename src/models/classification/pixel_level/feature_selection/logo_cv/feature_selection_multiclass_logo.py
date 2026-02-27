"""
feature_selection_multiclass.py

Minimal Multiclass Feature Selection - Optimized for CRACK PR-AUC.

Key Design Decisions:
- MULTICLASS only, XGBoost only
- Primary metric: CRACK PR-AUC (one-vs-rest) for ALL decisions
- LOGO CV on grape image groups (CRACK + REGULAR from same images)
- Fixed 80/20 holdout for non-grape classes
- SNV applied once after preprocessing
- Minimal K grid: [1, 2, 5, 10, 20, 40, 60, 80, 100]

Author: Feature Selection Pipeline (Refactored)
Date: January 2026
"""

import sys
import json
import warnings
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, NamedTuple
from tqdm import tqdm

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score, accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix as sk_confusion_matrix, precision_recall_fscore_support
)
from xgboost import XGBClassifier

from src.preprocessing.spectral_preprocessing import preprocess_multiclass_dataset

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Install with: pip install shap")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== CONFIGURATION ====================

CSV_PATH = Path(str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_multiclass_2026-01-16.csv"))
OUTPUT_BASE = Path(str(_PROJECT_ROOT / r"results/feature_selection_multiclass_logo"))

# Default K grid (minimal)
DEFAULT_K_GRID = [1, 2, 5, 10, 20, 40, 60, 80, 100]

# XGBoost params
def get_xgb_params(random_state: int = 42, use_gpu: bool = True) -> dict:
    params = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": 0,
    }
    if use_gpu:
        params["device"] = "cuda"
    return params


# ==================== SNV (Single Implementation) ====================

def apply_snv(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard Normal Variate - per sample normalization."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + eps)


# ==================== FOLD INFO ====================

class FoldInfo(NamedTuple):
    fold_idx: int
    held_out_group: str
    n_train: int
    n_test: int
    crack_train: int
    crack_test: int
    regular_train: int
    regular_test: int


# ==================== CV SPLITS (LOGO on Grape Images) ====================

def create_logo_cv_splits(
    y: np.ndarray,
    original_labels: np.ndarray,
    segment_ids: np.ndarray,
    image_ids: np.ndarray,
    crack_class_idx: int,
    regular_class_idx: int,
    random_state: int = 42,
    non_grape_holdout_frac: float = 0.20,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, FoldInfo]], Dict]:
    """
    LOGO CV on grape image groups (CRACK + REGULAR from same images).
    Fixed 80/20 holdout for non-grape classes.
    """
    n_samples = len(y)
    all_indices = np.arange(n_samples)
    
    # Identify grape samples (CRACK or REGULAR)
    is_grape = (y == crack_class_idx) | (y == regular_class_idx)
    grape_indices = all_indices[is_grape]
    non_grape_indices = all_indices[~is_grape]
    
    print(f"[CV] Total: {n_samples}, Grape: {len(grape_indices)}, Non-grape: {len(non_grape_indices)}")
    
    # Get grape image groups
    grape_image_ids = image_ids[grape_indices]
    unique_grape_images = np.unique(grape_image_ids)
    print(f"[CV] Unique grape image groups: {len(unique_grape_images)}")
    
    # Fixed non-grape holdout (by image_id or segment_id)
    if len(non_grape_indices) > 0:
        ng_image_ids = image_ids[non_grape_indices]
        ng_segment_ids = segment_ids[non_grape_indices]
        
        # Try grouping by image first
        gss = GroupShuffleSplit(n_splits=1, test_size=non_grape_holdout_frac, random_state=random_state)
        try:
            ng_train_local, ng_test_local = next(gss.split(non_grape_indices, groups=ng_image_ids))
            grouping = "image_id"
        except ValueError:
            # Fallback to segment_id
            ng_train_local, ng_test_local = next(gss.split(non_grape_indices, groups=ng_segment_ids))
            grouping = "segment_id"
        
        ng_train_global = non_grape_indices[ng_train_local]
        ng_test_global = non_grape_indices[ng_test_local]
        
        # Sanity check: no segment overlap
        train_segs = set(segment_ids[ng_train_global])
        test_segs = set(segment_ids[ng_test_global])
        assert not (train_segs & test_segs), "LEAKAGE: Non-grape segment overlap!"
        
        print(f"[CV] Non-grape split ({grouping}): {len(ng_train_global)} train, {len(ng_test_global)} test")
    else:
        ng_train_global = np.array([], dtype=int)
        ng_test_global = np.array([], dtype=int)
        grouping = "N/A"
    
    # LOGO on grape image groups
    logo = LeaveOneGroupOut()
    folds = []
    manifest = {
        "non_grape_holdout": {
            "grouping": grouping,
            "train_count": len(ng_train_global),
            "test_count": len(ng_test_global),
        },
        "folds": [],
    }
    
    for fold_idx, (grape_train_local, grape_test_local) in enumerate(
        logo.split(grape_indices, y[grape_indices], groups=grape_image_ids)
    ):
        grape_train_global = grape_indices[grape_train_local]
        grape_test_global = grape_indices[grape_test_local]
        
        # Combine: grape (LOGO) + non-grape (fixed)
        train_idx = np.concatenate([grape_train_global, ng_train_global])
        test_idx = np.concatenate([grape_test_global, ng_test_global])
        
        # Sanity check: no segment leakage
        train_segments = set(segment_ids[train_idx])
        test_segments = set(segment_ids[test_idx])
        overlap = train_segments & test_segments
        assert not overlap, f"LEAKAGE fold {fold_idx}: {len(overlap)} segments overlap!"
        
        # Sanity check: held-out grape group not in train
        held_out_group = str(np.unique(grape_image_ids[grape_test_local])[0])
        train_grape_groups = set(image_ids[grape_train_global])
        assert held_out_group not in train_grape_groups, f"LEAKAGE: held-out group {held_out_group} in train!"
        
        # Count classes
        y_train, y_test = y[train_idx], y[test_idx]
        fold_info = FoldInfo(
            fold_idx=fold_idx,
            held_out_group=held_out_group,
            n_train=len(train_idx),
            n_test=len(test_idx),
            crack_train=int((y_train == crack_class_idx).sum()),
            crack_test=int((y_test == crack_class_idx).sum()),
            regular_train=int((y_train == regular_class_idx).sum()),
            regular_test=int((y_test == regular_class_idx).sum()),
        )
        
        folds.append((train_idx, test_idx, fold_info))
        manifest["folds"].append({
            "fold_idx": fold_idx,
            "held_out_group": held_out_group,
            "n_train": fold_info.n_train,
            "n_test": fold_info.n_test,
            "crack_train": fold_info.crack_train,
            "crack_test": fold_info.crack_test,
            "regular_train": fold_info.regular_train,
            "regular_test": fold_info.regular_test,
        })
    
    print(f"[CV] Created {len(folds)} LOGO folds on grape image groups")
    return folds, manifest


# ==================== CRACK PR-AUC COMPUTATION ====================

def compute_crack_prauc(y_true: np.ndarray, y_prob: np.ndarray, crack_idx: int) -> float:
    """Compute CRACK PR-AUC (one-vs-rest)."""
    y_true_bin = (y_true == crack_idx).astype(int)
    y_score = y_prob[:, crack_idx]
    
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return np.nan
    
    return average_precision_score(y_true_bin, y_score)


# ==================== SHAP RANKING (Once, Train-Only) ====================

def compute_shap_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    xgb_params: dict,
    max_shap_samples: int = 5000,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute SHAP ranking from training data only (fold 0)."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP required. Install with: pip install shap")
    
    print("\n[SHAP] Computing feature ranking from training data...")
    
    # Train XGBoost
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    # Subsample for SHAP
    if X_train.shape[0] > max_shap_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train.shape[0], max_shap_samples, replace=False)
        X_shap = X_train[idx]
    else:
        X_shap = X_train
    
    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Handle multi-class
    if isinstance(shap_values, list):
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        importance = np.abs(shap_values).mean(axis=0)
    
    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'feature_idx': range(len(feature_names)),
        'mean_abs_shap': importance,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)
    
    print(f"[SHAP] Top 5 features: {ranking_df.head(5)['feature'].tolist()}")
    
    # Save artifacts
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ranking_df.to_csv(output_dir / 'shap_ranking.csv', index=False)
        
        # Save top100 manifest (stage tracking)
        top100_df = ranking_df.head(100)
        top100_manifest = {
            'description': 'SHAP Top-100 features from fold 0 training data',
            'n_features': len(top100_df),
            'top100_indices': top100_df['feature_idx'].tolist(),
            'top100_names': top100_df['feature'].tolist(),
            'top100_shap_scores': top100_df['mean_abs_shap'].tolist(),
            'top100_ranks': top100_df['rank'].tolist(),
        }
        with open(output_dir / 'top100_manifest.json', 'w') as f:
            json.dump(top100_manifest, f, indent=2)
        print(f"[SHAP] Saved: {output_dir / 'top100_manifest.json'}")
        
        # Plot top 30
        top_n = min(30, len(ranking_df))
        fig, ax = plt.subplots(figsize=(10, 8))
        top_df = ranking_df.head(top_n)
        ax.barh(range(top_n), top_df['mean_abs_shap'].values[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_df['feature'].values[::-1], fontsize=8)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'Top {top_n} Features by SHAP Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_top30.png', dpi=150)
        plt.close()
        print(f"[SHAP] Saved: {output_dir / 'shap_ranking.csv'}")
    
    return ranking_df


# ==================== EVALUATE SINGLE FOLD ====================

def evaluate_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_info: FoldInfo,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    crack_idx: int,
    regular_idx: int,
    n_classes: int,
    xgb_params: dict,
) -> Dict:
    """Evaluate XGBoost on single fold with selected features."""
    X_train = X[train_idx][:, feature_indices]
    X_test = X[test_idx][:, feature_indices]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train (no early stopping - fixed n_estimators)
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # PRIMARY METRIC: CRACK PR-AUC
    crack_prauc = compute_crack_prauc(y_test, y_prob, crack_idx)
    
    # Secondary metrics - global
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Per-class metrics for CRACK and REGULAR
    prfs = precision_recall_fscore_support(
        y_test, y_pred, labels=[crack_idx, regular_idx], zero_division=0
    )
    crack_prec, regular_prec = prfs[0][0], prfs[0][1]
    crack_rec, regular_rec = prfs[1][0], prfs[1][1]
    crack_f1_val, regular_f1_val = prfs[2][0], prfs[2][1]
    
    # Confusion matrix
    cm = sk_confusion_matrix(y_test, y_pred)
    
    return {
        'fold_idx': fold_info.fold_idx,
        'held_out_group': fold_info.held_out_group,
        # Primary
        'crack_prauc': crack_prauc,
        # Global metrics
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'balanced_acc': balanced_acc,
        # CRACK metrics
        'crack_precision': crack_prec,
        'crack_recall': crack_rec,
        'crack_f1': crack_f1_val,
        # REGULAR metrics
        'regular_precision': regular_prec,
        'regular_recall': regular_rec,
        'regular_f1': regular_f1_val,
        # Confusion matrix
        'confusion_matrix': json.dumps(cm.tolist()),
        # Fold info
        'crack_train': fold_info.crack_train,
        'crack_test': fold_info.crack_test,
        'regular_train': fold_info.regular_train,
        'regular_test': fold_info.regular_test,
    }


# ==================== K EVALUATION ====================

def evaluate_k_grid(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray, FoldInfo]],
    shap_ranking: pd.DataFrame,
    k_grid: List[int],
    crack_idx: int,
    regular_idx: int,
    n_classes: int,
    feature_names: List[str],
    xgb_params: dict,
    output_dir: Path,
    top100_manifest_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate K grid across all folds with rich logging."""
    print(f"\n[K-EVAL] Evaluating K grid: {k_grid}")
    
    # Get sorted feature indices (full ranking)
    sorted_feature_idx = shap_ranking['feature_idx'].values
    sorted_feature_names = shap_ranking['feature'].values
    
    # Get SHAP top-100 for stage tracking
    shap_top100_idx = sorted_feature_idx[:100].tolist()
    shap_top100_names = sorted_feature_names[:100].tolist()
    
    all_fold_results = []
    global_results = []
    
    for k in tqdm(k_grid, desc="K-sweep"):
        topk_idx = sorted_feature_idx[:k]
        topk_names = [feature_names[i] for i in topk_idx]
        
        # Full feature lists for this K (NOT truncated)
        topk_idx_list = topk_idx.tolist()
        topk_names_list = topk_names  # already a list
        
        fold_metrics = []
        for train_idx, test_idx, fold_info in folds:
            result = evaluate_fold(
                train_idx, test_idx, fold_info,
                X, y, topk_idx, crack_idx, regular_idx, n_classes, xgb_params
            )
            result['k'] = k
            # FULL selected features (not truncated)
            result['selected_feature_indices'] = json.dumps(topk_idx_list)
            result['selected_feature_names'] = json.dumps(topk_names_list)
            # Stage tracking
            result['stage_shap_top100_manifest_path'] = top100_manifest_path
            result['stage_current_topk_indices'] = json.dumps(topk_idx_list)
            result['stage_current_topk_names'] = json.dumps(topk_names_list)
            fold_metrics.append(result)
            all_fold_results.append(result)
        
        # Aggregate all metrics (PRIMARY: mean CRACK PR-AUC)
        def safe_mean(vals): return np.nanmean(vals)
        def safe_std(vals): return np.nanstd(vals)
        
        global_results.append({
            'k': k,
            # Primary metric
            'mean_crack_prauc': safe_mean([r['crack_prauc'] for r in fold_metrics]),
            'std_crack_prauc': safe_std([r['crack_prauc'] for r in fold_metrics]),
            # Global metrics
            'mean_accuracy': safe_mean([r['accuracy'] for r in fold_metrics]),
            'std_accuracy': safe_std([r['accuracy'] for r in fold_metrics]),
            'mean_macro_f1': safe_mean([r['macro_f1'] for r in fold_metrics]),
            'std_macro_f1': safe_std([r['macro_f1'] for r in fold_metrics]),
            'mean_weighted_f1': safe_mean([r['weighted_f1'] for r in fold_metrics]),
            'std_weighted_f1': safe_std([r['weighted_f1'] for r in fold_metrics]),
            'mean_balanced_acc': safe_mean([r['balanced_acc'] for r in fold_metrics]),
            'std_balanced_acc': safe_std([r['balanced_acc'] for r in fold_metrics]),
            # CRACK metrics
            'mean_crack_precision': safe_mean([r['crack_precision'] for r in fold_metrics]),
            'std_crack_precision': safe_std([r['crack_precision'] for r in fold_metrics]),
            'mean_crack_recall': safe_mean([r['crack_recall'] for r in fold_metrics]),
            'std_crack_recall': safe_std([r['crack_recall'] for r in fold_metrics]),
            'mean_crack_f1': safe_mean([r['crack_f1'] for r in fold_metrics]),
            'std_crack_f1': safe_std([r['crack_f1'] for r in fold_metrics]),
            # REGULAR metrics
            'mean_regular_precision': safe_mean([r['regular_precision'] for r in fold_metrics]),
            'std_regular_precision': safe_std([r['regular_precision'] for r in fold_metrics]),
            'mean_regular_recall': safe_mean([r['regular_recall'] for r in fold_metrics]),
            'std_regular_recall': safe_std([r['regular_recall'] for r in fold_metrics]),
            'mean_regular_f1': safe_mean([r['regular_f1'] for r in fold_metrics]),
            'std_regular_f1': safe_std([r['regular_f1'] for r in fold_metrics]),
            # Feature info
            'n_folds': len(fold_metrics),
            'selected_feature_indices': json.dumps(topk_idx_list),
            'selected_feature_names': json.dumps(topk_names_list),
        })
    
    global_df = pd.DataFrame(global_results)
    fold_df = pd.DataFrame(all_fold_results)
    
    # Create folds subdirectory
    folds_dir = output_dir / 'folds'
    folds_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    global_df.to_csv(output_dir / 'performance_vs_k.csv', index=False)
    fold_df.to_csv(folds_dir / 'fold_metrics_per_k.csv', index=False)
    
    # Save parquet (optional but preferred for speed/size)
    try:
        fold_df.to_parquet(folds_dir / 'fold_metrics_per_k.parquet', index=False)
        print(f"[K-EVAL] Saved parquet: {folds_dir / 'fold_metrics_per_k.parquet'}")
    except Exception as e:
        print(f"[K-EVAL] Parquet save skipped (install pyarrow): {e}")
    
    print(f"[K-EVAL] Saved: {output_dir / 'performance_vs_k.csv'}")
    print(f"[K-EVAL] Saved: {folds_dir / 'fold_metrics_per_k.csv'}")
    
    # Sanity checks
    print(f"\n[SANITY] fold_metrics_per_k.csv columns: {len(fold_df.columns)}")
    print(f"[SANITY] Columns: {list(fold_df.columns)}")
    for check_k in [k_grid[0], k_grid[-1]]:
        row = fold_df[fold_df['k'] == check_k].iloc[0]
        feat_names = json.loads(row['selected_feature_names'])
        print(f"[SANITY] K={check_k}: len(selected_feature_names)={len(feat_names)}, expected={check_k}, match={len(feat_names)==check_k}")
    
    return global_df, fold_df


# ==================== FIND BEST K ====================

def find_best_k(global_df: pd.DataFrame, max_k: int) -> Tuple[int, float]:
    """Find best K <= max_k by mean CRACK PR-AUC (tie-break: smaller K)."""
    subset = global_df[global_df['k'] <= max_k].copy()
    if len(subset) == 0:
        return -1, np.nan
    
    # Best by PR-AUC, tie-break by smaller K
    subset = subset.sort_values(['mean_crack_prauc', 'k'], ascending=[False, True])
    best = subset.iloc[0]
    return int(best['k']), float(best['mean_crack_prauc'])


# ==================== PLOTS ====================

def create_plots(global_df: pd.DataFrame, output_dir: Path, n_features_total: int):
    """Create comprehensive plots for feature selection results."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    k_vals = global_df['k'].values
    
    # ===== 1. PR-AUC vs K (existing, enhanced) =====
    means = global_df['mean_crack_prauc'].values
    stds = global_df['std_crack_prauc'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_vals, means, yerr=stds, fmt='o-', capsize=4, label='Mean +/- Std')
    best_idx = np.nanargmax(means)
    ax.scatter([k_vals[best_idx]], [means[best_idx]], color='red', s=100, zorder=5, label=f'Best K={k_vals[best_idx]}')
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel('CRACK PR-AUC')
    ax.set_title('Feature Selection: CRACK PR-AUC vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'prauc_vs_k.png', dpi=150)
    plt.close()
    
    # ===== 2. Accuracy vs K =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_vals, global_df['mean_accuracy'].values, 
                yerr=global_df['std_accuracy'].values, fmt='o-', capsize=4)
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Feature Selection: Accuracy vs K')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_vs_k.png', dpi=150)
    plt.close()
    
    # ===== 3. Macro F1 vs K =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_vals, global_df['mean_macro_f1'].values,
                yerr=global_df['std_macro_f1'].values, fmt='o-', capsize=4)
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel('Macro F1')
    ax.set_title('Feature Selection: Macro F1 vs K')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'macro_f1_vs_k.png', dpi=150)
    plt.close()
    
    # ===== 4. Balanced Accuracy vs K =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_vals, global_df['mean_balanced_acc'].values,
                yerr=global_df['std_balanced_acc'].values, fmt='o-', capsize=4)
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Feature Selection: Balanced Accuracy vs K')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'balanced_acc_vs_k.png', dpi=150)
    plt.close()
    
    # ===== 5. CRACK Precision/Recall/F1 vs K (combined) =====
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(k_vals, global_df['mean_crack_precision'].values,
                yerr=global_df['std_crack_precision'].values, fmt='o-', capsize=3, label='Precision')
    ax.errorbar(k_vals, global_df['mean_crack_recall'].values,
                yerr=global_df['std_crack_recall'].values, fmt='s-', capsize=3, label='Recall')
    ax.errorbar(k_vals, global_df['mean_crack_f1'].values,
                yerr=global_df['std_crack_f1'].values, fmt='^-', capsize=3, label='F1')
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel('Score')
    ax.set_title('CRACK Class: Precision / Recall / F1 vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'crack_precision_recall_f1_vs_k.png', dpi=150)
    plt.close()
    
    # ===== 6. REGULAR Precision/Recall/F1 vs K (combined) =====
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(k_vals, global_df['mean_regular_precision'].values,
                yerr=global_df['std_regular_precision'].values, fmt='o-', capsize=3, label='Precision')
    ax.errorbar(k_vals, global_df['mean_regular_recall'].values,
                yerr=global_df['std_regular_recall'].values, fmt='s-', capsize=3, label='Recall')
    ax.errorbar(k_vals, global_df['mean_regular_f1'].values,
                yerr=global_df['std_regular_f1'].values, fmt='^-', capsize=3, label='F1')
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel('Score')
    ax.set_title('REGULAR Class: Precision / Recall / F1 vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'regular_precision_recall_f1_vs_k.png', dpi=150)
    plt.close()
    
    # ===== 7. Summary comparison with error bars =====
    constraints = ['Full', 100, 40, 20, 10]
    best_ks = []
    best_praucs = []
    best_stds = []
    labels = []
    
    # Full features baseline (use max K in grid or total features)
    max_k_in_grid = max(k_vals)
    full_row = global_df[global_df['k'] == max_k_in_grid].iloc[0]
    best_ks.append(int(max_k_in_grid))
    best_praucs.append(full_row['mean_crack_prauc'])
    best_stds.append(full_row['std_crack_prauc'])
    labels.append(f'Full (K={max_k_in_grid})')
    
    for max_k in [100, 40, 20, 10]:
        k, prauc = find_best_k(global_df, max_k)
        if k > 0:
            row = global_df[global_df['k'] == k].iloc[0]
            best_ks.append(k)
            best_praucs.append(prauc)
            best_stds.append(row['std_crack_prauc'])
            labels.append(f'Best K<={max_k}\n(K={k})')
    
    if best_ks:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(labels))
        ax.bar(x, best_praucs, yerr=best_stds, alpha=0.7, capsize=5, ecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('CRACK PR-AUC (mean +/- std)')
        ax.set_title('Best K per Constraint (Optimized for CRACK PR-AUC)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (prauc, std) in enumerate(zip(best_praucs, best_stds)):
            ax.text(i, prauc + std + 0.01, f'{prauc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'summary_best_k.png', dpi=150)
        plt.close()
    
    # ===== 8. Multi-metric comparison plot =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PR-AUC
    ax = axes[0, 0]
    ax.errorbar(k_vals, global_df['mean_crack_prauc'].values,
                yerr=global_df['std_crack_prauc'].values, fmt='o-', capsize=3)
    ax.set_xlabel('K')
    ax.set_ylabel('CRACK PR-AUC')
    ax.set_title('CRACK PR-AUC vs K')
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.errorbar(k_vals, global_df['mean_accuracy'].values,
                yerr=global_df['std_accuracy'].values, fmt='o-', capsize=3, color='orange')
    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs K')
    ax.grid(True, alpha=0.3)
    
    # Macro F1
    ax = axes[1, 0]
    ax.errorbar(k_vals, global_df['mean_macro_f1'].values,
                yerr=global_df['std_macro_f1'].values, fmt='o-', capsize=3, color='green')
    ax.set_xlabel('K')
    ax.set_ylabel('Macro F1')
    ax.set_title('Macro F1 vs K')
    ax.grid(True, alpha=0.3)
    
    # Balanced Accuracy
    ax = axes[1, 1]
    ax.errorbar(k_vals, global_df['mean_balanced_acc'].values,
                yerr=global_df['std_balanced_acc'].values, fmt='o-', capsize=3, color='purple')
    ax.set_xlabel('K')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Balanced Accuracy vs K')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Selection: Multi-Metric Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'multi_metric_comparison.png', dpi=150)
    plt.close()
    
    print(f"[PLOTS] Saved {len(list(plots_dir.glob('*.png')))} plots to {plots_dir}")


# ==================== MAIN PIPELINE ====================

def run_feature_selection(
    csv_path: Path = CSV_PATH,
    output_dir: Optional[Path] = None,
    k_grid: List[int] = None,
    max_samples: Optional[int] = None,
    random_state: int = 42,
    use_gpu: bool = True,
) -> Dict:
    """Run minimal feature selection pipeline optimized for CRACK PR-AUC."""
    start_time = time.time()
    
    if k_grid is None:
        k_grid = DEFAULT_K_GRID
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_dir is None:
        output_dir = OUTPUT_BASE / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FEATURE SELECTION - CRACK PR-AUC OPTIMIZED")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"K grid: {k_grid}")
    print(f"Max samples: {max_samples or 'FULL'}")
    print("=" * 70)
    
    # ===== LOAD DATA =====
    print("\n[DATA] Loading...")
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} rows")
    
    if max_samples and len(df) > max_samples:
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(df, test_size=max_samples/len(df), stratify=df['label'], random_state=random_state)
        df = df.reset_index(drop=True)
        print(f"[DATA] Sampled to {len(df)} rows")
    
    # ===== PREPROCESS =====
    print("\n[PREPROCESS] Running...")
    data = preprocess_multiclass_dataset(
        df, wl_min=450, wl_max=925, apply_snv=False, remove_outliers=False,
        balanced=False, label_col='label', hs_dir_col='hs_dir', segment_col='mask_path', seed=random_state,
    )
    
    X = data.X
    y = data.y
    feature_names = data.feature_names
    class_names = data.class_names
    class_mapping = data.class_mapping
    
    # ===== APPLY SNV (ONCE) =====
    print("\n[SNV] Applying Standard Normal Variate...")
    X = apply_snv(X)
    
    # ===== GET CLASS INDICES =====
    crack_idx = class_mapping.get("CRACK")
    regular_idx = class_mapping.get("REGULAR")
    n_classes = len(class_names)
    
    assert crack_idx is not None, "CRACK class not found in mapping!"
    assert regular_idx is not None, "REGULAR class not found in mapping!"
    print(f"[INFO] CRACK idx: {crack_idx}, REGULAR idx: {regular_idx}, Classes: {n_classes}")
    print(f"[INFO] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # ===== CREATE CV SPLITS =====
    print("\n[CV] Creating LOGO splits on grape image groups...")
    folds, manifest = create_logo_cv_splits(
        y=y, original_labels=data.original_labels, segment_ids=data.segment_ids,
        image_ids=data.image_ids, crack_class_idx=crack_idx, regular_class_idx=regular_idx,
        random_state=random_state, non_grape_holdout_frac=0.20,
    )
    
    manifest["crack_class_idx"] = crack_idx
    manifest["regular_class_idx"] = regular_idx
    manifest["class_mapping"] = class_mapping
    with open(output_dir / 'cv_split_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # ===== XGB PARAMS =====
    xgb_params = get_xgb_params(random_state, use_gpu)
    
    # ===== SHAP RANKING (Fold 0 Train Only) =====
    train_idx_0, _, _ = folds[0]
    X_train_0, y_train_0 = X[train_idx_0], y[train_idx_0]
    
    shap_dir = output_dir / 'shap'
    shap_ranking = compute_shap_ranking(
        X_train_0, y_train_0, feature_names, xgb_params,
        max_shap_samples=5000, output_dir=shap_dir
    )
    
    # Top-100 manifest path for stage tracking
    top100_manifest_path = str(shap_dir / 'top100_manifest.json')
    
    # ===== K EVALUATION =====
    # Filter K grid to available features
    max_features = X.shape[1]
    k_grid = [k for k in k_grid if k <= max_features]
    
    global_df, fold_df = evaluate_k_grid(
        X, y, folds, shap_ranking, k_grid, crack_idx, regular_idx, n_classes, 
        feature_names, xgb_params, output_dir, top100_manifest_path
    )
    
    # ===== FIND BEST K =====
    print("\n[RESULTS] Best K per constraint (by CRACK PR-AUC):")
    summary = []
    for max_k in [100, 40, 20, 10]:
        k, prauc = find_best_k(global_df, max_k)
        if k > 0:
            print(f"  Best K <= {max_k}: K={k}, PR-AUC={prauc:.4f}")
            summary.append({'constraint': f'K<={max_k}', 'best_k': k, 'crack_prauc': prauc})
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary_best_k.csv', index=False)
    
    # ===== PLOTS =====
    create_plots(global_df, output_dir, n_features_total=X.shape[1])
    
    # ===== CONFIG =====
    config = {
        'timestamp': timestamp,
        'csv_path': str(csv_path),
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_folds': len(folds),
        'k_grid': k_grid,
        'crack_class_idx': crack_idx,
        'regular_class_idx': regular_idx,
        'class_mapping': class_mapping,
        'snv_applied': True,
        'optimization_metric': 'CRACK_PR_AUC',
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Completed in {elapsed/60:.1f} min. Output: {output_dir}")
    
    return {'global_df': global_df, 'summary_df': summary_df, 'config': config}


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Feature Selection - CRACK PR-AUC Optimized')
    parser.add_argument('--csv_path', type=str, default=str(CSV_PATH))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--k_grid', type=str, default=None, help='Comma-separated K values')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--no_gpu', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    k_grid = None
    if args.k_grid:
        k_grid = [int(k.strip()) for k in args.k_grid.split(',')]
    
    run_feature_selection(
        csv_path=Path(args.csv_path),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        k_grid=k_grid,
        max_samples=args.max_samples,
        random_state=args.random_state,
        use_gpu=not args.no_gpu,
    )
