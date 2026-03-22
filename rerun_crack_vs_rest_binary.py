"""
Re-run crack_vs_rest as TRUE binary classification (CRACK=1, REST=0).

The original pipeline had a bug where prepare_data_for_label_setup() was never called,
so crack_vs_rest ran the same 10-class model as crack_regular_rest.

This script:
1. Loads the multiclass dataset (same preprocessing as original)
2. Creates CV splits using the ORIGINAL multiclass labels (identical folds)
3. Remaps labels to binary (CRACK=1, REST=0) for model training
4. Trains all 5 models with identical hyperparameters
5. Reports CRACK-specific metrics for the thesis table
"""

import sys
from pathlib import Path

# Add project root (same as original pipeline)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
import warnings
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, NamedTuple
from collections import Counter

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.preprocessing.spectral_preprocessing import (
    preprocess_multiclass_dataset,
    PreprocessedData,
    GRAPE_CLASSES,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
NON_GRAPE_HOLDOUT_FRAC = 0.20


# ==================== FOLD INFO (same as original) ====================

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


# ==================== PLS-DA (same as original) ====================

class PLSDABinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=10, threshold=0.5):
        self.n_components = n_components
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"PLSDABinaryClassifier requires exactly 2 classes, got {len(self.classes_)}")
        self.class_0_ = self.classes_[0]
        self.class_1_ = self.classes_[1]
        y_binary = (y == self.class_1_).astype(float)
        self.pls_ = PLSRegression(n_components=min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_.fit(X, y_binary)
        return self

    def predict(self, X):
        scores = self.pls_.predict(X).ravel()
        predictions = (scores >= self.threshold).astype(int)
        return np.where(predictions == 1, self.class_1_, self.class_0_)

    def predict_proba(self, X):
        scores = self.pls_.predict(X).ravel()
        p_class1 = np.clip(scores, 0, 1)
        return np.column_stack([1 - p_class1, p_class1])


# ==================== CV SPLITS (exact copy from original pipeline) ====================

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
    EXACT copy from unified_experiment_pipeline_acc.py.
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

    # Separate grape samples into CRACK and REGULAR using provided class indices
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

    print(f"[SPLIT] Created {len(folds)} LOGO folds on CRACK samples only")
    return folds


# ==================== MODEL DEFINITIONS (same as original) ====================

def get_models():
    """All 5 models with identical hyperparameters to the original pipeline."""
    return [
        ("PLS-DA", Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSDABinaryClassifier(n_components=10)),
        ])),
        ("Logistic Regression (L1)", Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                penalty="l1", solver="saga", max_iter=500, tol=1e-3,
                class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ])),
        ("Random Forest", Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced',
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ])),
        ("XGBoost", XGBClassifier(
            n_estimators=100, max_depth=5, use_label_encoder=False,
            eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
        ("MLP (Small)", Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                solver='adam', alpha=0.0001, batch_size='auto',
                learning_rate='adaptive', learning_rate_init=0.001,
                max_iter=500, early_stopping=False, tol=1e-4,
                random_state=RANDOM_STATE, verbose=False,
            )),
        ])),
    ]


# ==================== SINGLE FOLD EVALUATION (same logic as original) ====================

def evaluate_single_fold(fold_data, model, X, y, segment_ids):
    """
    Evaluate a single fold. Same logic as original pipeline's evaluate_single_fold.
    """
    train_idx, test_idx, fold_info = fold_data
    fold_start_time = time.time()
    crack_class_idx = 1  # Binary: CRACK=1

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Compute sample weights (same as original)
    sample_weights = compute_sample_weight('balanced', y_train)

    # Clone and fit model (same logic as original)
    fold_model = clone(model)

    train_start = time.time()
    try:
        if isinstance(fold_model, XGBClassifier):
            fold_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            fold_model.fit(X_train, y_train)
    except Exception as e:
        return {
            'fold_idx': fold_info.fold_idx,
            'error': str(e),
            'success': False,
        }
    train_time = time.time() - train_start

    # Predict
    infer_start = time.time()
    y_pred = fold_model.predict(X_test)

    y_prob_all = None
    if hasattr(fold_model, 'predict_proba'):
        try:
            y_prob_all = fold_model.predict_proba(X_test)
        except Exception:
            pass
    infer_time = time.time() - infer_start

    # Calculate metrics (same as original)
    n_classes = 2
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    labels_for_metrics = list(range(n_classes))
    prec_per_class = precision_score(y_test, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels_for_metrics, zero_division=0)

    # CRACK-specific metrics
    crack_prec = prec_per_class[crack_class_idx] if crack_class_idx < len(prec_per_class) else np.nan
    crack_rec = rec_per_class[crack_class_idx] if crack_class_idx < len(rec_per_class) else np.nan
    crack_f1 = f1_per_class[crack_class_idx] if crack_class_idx < len(f1_per_class) else np.nan

    # CRACK AUC metrics
    crack_roc_auc = np.nan
    crack_pr_auc = np.nan
    y_prob_crack = None

    if y_prob_all is not None and crack_class_idx < y_prob_all.shape[1]:
        y_prob_crack = y_prob_all[:, crack_class_idx]
        y_test_binary = (y_test == crack_class_idx).astype(int)

        if y_test_binary.sum() > 0 and y_test_binary.sum() < len(y_test_binary):
            try:
                crack_roc_auc = roc_auc_score(y_test_binary, y_prob_crack)
                crack_pr_auc = average_precision_score(y_test_binary, y_prob_crack)
            except Exception:
                pass

    fold_duration = time.time() - fold_start_time

    crack_count_train = int((y_train == crack_class_idx).sum())
    crack_count_test = int((y_test == crack_class_idx).sum())

    return {
        'fold_idx': fold_info.fold_idx,
        'success': True,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'crack_prec': crack_prec,
        'crack_rec': crack_rec,
        'crack_f1': crack_f1,
        'crack_roc_auc': crack_roc_auc,
        'crack_pr_auc': crack_pr_auc,
        'prec_per_class': prec_per_class,
        'rec_per_class': rec_per_class,
        'f1_per_class': f1_per_class,
        'train_time': train_time,
        'infer_time': infer_time,
        'fold_duration': fold_duration,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob_crack': y_prob_crack,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'crack_count_train': crack_count_train,
        'crack_count_test': crack_count_test,
    }


# ==================== MODEL CV EVALUATION (same logic as original) ====================

def format_mean_std(values, decimals=4):
    """Format list of values as 'mean +/- std'."""
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def evaluate_model_cv(model, model_name, X, y, segment_ids, folds, class_names, crack_class_idx):
    """
    Evaluate model across all CV folds. Same logic as original pipeline.
    """
    n_classes = len(class_names)
    n_folds = len(folds)

    print(f"    [CV] Running {n_folds} folds...")
    eval_start = time.time()

    fold_results = []
    cumulative_time = 0.0
    for fold_idx, fold_data in enumerate(folds):
        result = evaluate_single_fold(fold_data, model, X, y, segment_ids)
        fold_results.append(result)
        if result.get('success', False):
            crack_test = result.get('crack_count_test', 0)
            crack_train = result.get('crack_count_train', 0)
            fold_time = result.get('fold_duration', 0.0)
            cumulative_time += fold_time
            print(f"      Fold {fold_idx+1}/{n_folds}: acc={result['acc']:.4f}, CRACK F1={result['crack_f1']:.4f} | "
                  f"time={fold_time:.1f}s (total={cumulative_time:.1f}s) | CRACK: train={crack_train}, test={crack_test}")
        else:
            print(f"      Fold {fold_idx+1}/{n_folds}: FAILED - {result.get('error', 'Unknown error')}")

    eval_end = time.time()
    total_eval_time = eval_end - eval_start

    # Filter successful folds
    successful_results = [r for r in fold_results if r.get('success', False)]

    if not successful_results:
        return {"Model Name": model_name, "Status": "ALL FOLDS FAILED"}

    # Aggregate metrics
    metrics = {
        'accs': [r['acc'] for r in successful_results],
        'balanced_accs': [r['balanced_acc'] for r in successful_results],
        'macro_f1s': [r['macro_f1'] for r in successful_results],
        'weighted_f1s': [r['weighted_f1'] for r in successful_results],
        'crack_precs': [r['crack_prec'] for r in successful_results],
        'crack_recs': [r['crack_rec'] for r in successful_results],
        'crack_f1s': [r['crack_f1'] for r in successful_results],
        'crack_roc_aucs': [r['crack_roc_auc'] for r in successful_results],
        'crack_pr_aucs': [r['crack_pr_auc'] for r in successful_results],
    }

    # Print fold summary
    print(f"    [SUMMARY] {len(successful_results)}/{n_folds} folds completed in {total_eval_time:.1f}s")
    print(f"    Accuracy: {np.mean(metrics['accs']):.4f} +/- {np.std(metrics['accs']):.4f}")
    print(f"    Balanced Accuracy: {np.mean(metrics['balanced_accs']):.4f} +/- {np.std(metrics['balanced_accs']):.4f}")
    print(f"    Macro-F1: {np.mean(metrics['macro_f1s']):.4f} +/- {np.std(metrics['macro_f1s']):.4f}")
    print(f"    CRACK F1: {np.mean(metrics['crack_f1s']):.4f} +/- {np.std(metrics['crack_f1s']):.4f}")

    # Build summary
    summary_dict = {
        "Model Name": model_name,
        "Accuracy": format_mean_std(metrics['accs']),
        "Balanced Accuracy": format_mean_std(metrics['balanced_accs']),
        "Macro-F1": format_mean_std(metrics['macro_f1s']),
        "Weighted-F1": format_mean_std(metrics['weighted_f1s']),
        "CRACK Precision": format_mean_std(metrics['crack_precs']),
        "CRACK Recall": format_mean_std(metrics['crack_recs']),
        "CRACK F1": format_mean_std(metrics['crack_f1s']),
        "CRACK ROC-AUC": format_mean_std(metrics['crack_roc_aucs']),
        "CRACK PR-AUC": format_mean_std(metrics['crack_pr_aucs']),
        "Total Time (s)": f"{total_eval_time:.1f}",
        "Folds Completed": f"{len(successful_results)}/{n_folds}",
    }

    # Store raw metrics
    summary_dict["_raw_metrics"] = metrics

    return summary_dict


# ==================== MAIN ====================

def main():
    # Load data (same as original pipeline)
    csv_path = _PROJECT_ROOT / "src" / "preprocessing" / "dataset_builder_grapes" / "detection" / "raw_exported_data" / "all_origin_signatures_results_multiclass_2026-01-16.csv"
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Preprocess multiclass dataset (UNBALANCED, same as original)
    data = preprocess_multiclass_dataset(
        df.copy(),
        wl_min=450, wl_max=925,
        apply_snv=True, remove_outliers=False,
        balanced=False, label_col="label",
        hs_dir_col="hs_dir", segment_col="mask_path",
        cap_class=None, max_samples_per_class=None,
        seed=RANDOM_STATE,
    )

    print(f"\nOriginal multiclass data: {data.X.shape[0]} samples, {data.X.shape[1]} features")
    print(f"Class mapping: {data.class_mapping}")
    print(f"Class distribution: {dict(Counter(data.y))}")

    # --- Create CV splits using ORIGINAL multiclass labels ---
    # This ensures IDENTICAL folds as the original pipeline
    crack_class_idx_original = data.class_mapping.get("CRACK", 3)
    regular_class_idx_original = data.class_mapping.get("REGULAR", 7)

    print(f"\n[INFO] Creating CV splits using original multiclass labels")
    print(f"[INFO] CRACK class idx (original): {crack_class_idx_original}")
    print(f"[INFO] REGULAR class idx (original): {regular_class_idx_original}")
    print(f"[INFO] Grape class indices: {data.grape_class_indices}")

    folds = create_domain_aware_cv_splits(
        y=data.y,  # Use ORIGINAL multiclass labels for split creation
        original_labels=data.original_labels,
        segment_ids=data.segment_ids,
        image_ids=data.image_ids,
        grape_classes=data.grape_class_indices,
        crack_class_idx=crack_class_idx_original,
        regular_class_idx=regular_class_idx_original,
        random_state=RANDOM_STATE,
        non_grape_holdout_frac=NON_GRAPE_HOLDOUT_FRAC,
    )

    # --- Remap labels to binary: CRACK=1, REST=0 ---
    y_binary = (data.y == crack_class_idx_original).astype(int)
    print(f"\nBinary remapping: CRACK (original idx {crack_class_idx_original}) -> 1, REST -> 0")
    print(f"Binary distribution: REST={int((y_binary == 0).sum())}, CRACK={int((y_binary == 1).sum())}")

    # Binary class info
    binary_class_names = ["REST", "CRACK"]
    binary_crack_class_idx = 1  # In binary encoding

    # Run all models
    models = get_models()
    all_results = []

    for model_name, model in models:
        print(f"\n{'='*60}")
        print(f"[MODEL] {model_name}")
        print(f"{'='*60}")

        summary = evaluate_model_cv(
            model=model,
            model_name=model_name,
            X=data.X,
            y=y_binary,  # Use BINARY labels for training/evaluation
            segment_ids=data.segment_ids,
            folds=folds,
            class_names=binary_class_names,
            crack_class_idx=binary_crack_class_idx,
        )

        all_results.append(summary)

        # Save intermediate CSV after each model
        out_csv = _PROJECT_ROOT / "data" / "results" / "crack_vs_rest_binary_rerun.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Save simplified version (without raw metrics)
        save_data = []
        for s in all_results:
            row = {k: v for k, v in s.items() if not k.startswith('_')}
            save_data.append(row)
        pd.DataFrame(save_data).to_csv(out_csv, index=False)
        print(f"  Intermediate results saved to {out_csv}")

    # Print final summary table
    print("\n" + "="*80)
    print("FINAL RESULTS - crack_vs_rest (TRUE BINARY)")
    print("="*80)
    for s in all_results:
        print(f"\n{s['Model Name']}:")
        for k, v in s.items():
            if not k.startswith('_'):
                print(f"  {k}: {v}")

    # Print LaTeX table rows
    print("\n" + "="*80)
    print("LATEX TABLE ROWS (for crack_vs_rest in results_new.tex):")
    print("="*80)
    for s in all_results:
        raw = s.get('_raw_metrics', {})
        if not raw:
            continue
        acc = np.mean(raw['accs'])
        bal = np.mean(raw['balanced_accs'])
        prec = np.mean(raw['crack_precs'])
        rec = np.mean(raw['crack_recs'])
        f1c = np.mean(raw['crack_f1s'])
        roc = np.mean([v for v in raw['crack_roc_aucs'] if not np.isnan(v)])
        pr = np.mean([v for v in raw['crack_pr_aucs'] if not np.isnan(v)])
        print(f"        & {s['Model Name']:<25s} & {acc:.3f} & {bal:.3f} "
              f"& {prec:.3f} & {rec:.3f} & {f1c:.3f} "
              f"& {roc:.3f} & {pr:.3f} \\\\")


if __name__ == "__main__":
    main()
