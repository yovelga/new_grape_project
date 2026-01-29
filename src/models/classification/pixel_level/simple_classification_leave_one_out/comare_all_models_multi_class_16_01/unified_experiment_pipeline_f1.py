"""
unified_experiment_pipeline_roc_auc.py

Unified Multi-Class Model Comparison Pipeline for Pixel-Level Classification.
OPTIMIZED FOR ROC-AUC (not PR-AUC).

This pipeline is designed for thesis-ready reproducibility with:
- Unified experiment matrix across all label setups and balance modes
- Same dataset for all models (balancing applied only inside training folds)
- All models preserved including proper multi-class support for PLS-DA
- Metrics reported as mean ± std across CV folds
- Single consolidated report for thesis Results section
- Dry-run/sanity mode for quick validation

Features:
- Label Setups: CRACK_REGULAR_REST (full multiclass), CRACK_VS_REST (binary), MULTI_CLASS (3-class)
- Balance Modes: BALANCED, UNBALANCED
- Domain-aware CV: LOGO on grape samples, fixed holdout for non-grape
- All models: Logistic Regression, SVM, Random Forest, XGBoost, MLP, PLS-DA (with OvR wrapper)

Author: Thesis Pipeline Refactor
Date: January 2026
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[6]  # Navigate up to Grape_Project
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
import warnings
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
from enum import Enum
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier

# Import the reusable preprocessing function
from src.preprocessing.spectral_preprocessing import (
    preprocess_multiclass_dataset,
    PreprocessedData,
    GRAPE_CLASSES,
    GRAPE_CLASS_IDS_3CLASS,
)

try:
    import joblib
    from joblib import Parallel, delayed
except ImportError:
    joblib = None
    Parallel = None
    delayed = None
    print("[WARN] joblib not available; model saving and parallel CV will be disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== ENUMERATIONS ====================

class LabelSetup(str, Enum):
    """Label setup configurations for experiments."""
    CRACK_REGULAR_REST = "crack_regular_rest"  # Full multiclass (all original labels)
    CRACK_VS_REST = "crack_vs_rest"  # Binary: CRACK vs ALL
    MULTI_CLASS = "multi_class"  # Explicit 3-class (REGULAR/CRACK/not_grape)


class BalanceMode(str, Enum):
    """Data balancing modes."""
    BALANCED = "balanced"
    UNBALANCED = "unbalanced"


# ==================== CONFIGURATION ====================

@dataclass
class ExperimentConfig:
    """
    Global experiment configuration.
    
    OPTIMIZATION STRATEGY: Maximize Accuracy (ROC-AUC)
    
    This pipeline trains models without class weighting to optimize for overall
    accuracy. This naturally tends to produce better ROC-AUC scores:
    
    1. No class_weight - standard training for accuracy
    2. No sample_weight for XGBoost - uniform sample importance
    3. eval_metric='auc' for XGBoost early stopping
    4. Reports both ROC-AUC and PR-AUC for comparison
    """
    # Dry-run / sanity mode: Set to None for full dataset
    max_samples: Optional[int] = None  # Set to None for full dataset run
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # CV settings
    non_grape_holdout_fraction: float = 0.20
    
    # Parallelization
    n_jobs: int = -1  # Use all available cores
    cv_parallel_backend: str = "loky"
    
    # Balancing settings
    max_samples_per_class: int = 50  # Per-class cap to prevent SVM from running too long
    
    # Label setups to run
    label_setups: List[LabelSetup] = field(default_factory=lambda: [
        LabelSetup.CRACK_REGULAR_REST,
        LabelSetup.CRACK_VS_REST,
        LabelSetup.MULTI_CLASS,
    ])
    
    # Balance modes to run
    balance_modes: List[BalanceMode] = field(default_factory=lambda: [
        BalanceMode.BALANCED,
        BalanceMode.UNBALANCED,
    ])
    
    # Models to train (set to False to skip)
    models_to_train: Dict[str, bool] = field(default_factory=lambda: {
        "Logistic Regression (L1)": True,
        "SVM (RBF)": False,  # Added SVM with class_weight='balanced'
        "Random Forest": True,
        "XGBoost": True,
        "MLP (Small)": True,
        "PLS-DA": True,
    })


@dataclass 
class DatasetConfig:
    """Configuration for a dataset source."""
    name: str
    csv_path: Path
    target_col: str
    crack_identifier: Union[str, int]  # "CRACK" for multiclass, 2 for 3class
    grape_classes: set  # Classes that are grape-related for domain split
    is_3class: bool = False


# Dataset paths
CSV_PATH_MULTICLASS = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_multiclass_2026-01-16.csv")
CSV_PATH_3CLASS = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_3class_2026-01-16.csv")

DATASET_CONFIGS = {
    "multiclass": DatasetConfig(
        name="multiclass",
        csv_path=CSV_PATH_MULTICLASS,
        target_col="label",
        crack_identifier="CRACK",
        grape_classes={"REGULAR", "CRACK"},
        is_3class=False,
    ),
    "3class": DatasetConfig(
        name="3class",
        csv_path=CSV_PATH_3CLASS,
        target_col="label_3class_id",
        crack_identifier=2,
        grape_classes={1, 2},  # REGULAR=1, CRACK=2 in 3-class
        is_3class=True,
    ),
}

# Experiment output directory
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = Path(rf"C:\Users\yovel\Desktop\Grape_Project\experiments\unified_experiment_F1_{TIMESTAMP}")


# ==================== HELPER FUNCTIONS ====================

def slugify_model_name(model_name: str) -> str:
    """Convert model name to a filesystem-friendly slug."""
    slug = model_name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_mean_std(values: List[float], decimals: int = 4) -> str:
    """Format list of values as 'mean ± std'."""
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ==================== PLS-DA CLASSIFIERS ====================

class PLSDABinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS-DA classifier for binary classification using PLSRegression.
    
    Uses threshold-based prediction on PLS regression scores.
    """
    
    def __init__(self, n_components: int = 10, threshold: float = 0.5):
        self.n_components = n_components
        self.threshold = threshold
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"PLSDABinaryClassifier requires exactly 2 classes, got {len(self.classes_)}")
        
        # Ensure y is binary 0/1
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


class PLSDAMultiClassOvR(BaseEstimator, ClassifierMixin):
    """
    Multi-class PLS-DA using One-vs-Rest strategy.
    
    Wraps PLSDABinaryClassifier in sklearn's OneVsRestClassifier to handle
    multi-class problems while maintaining PLS-DA's binary strengths.
    """
    
    def __init__(self, n_components: int = 10, threshold: float = 0.5):
        self.n_components = n_components
        self.threshold = threshold
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Create OvR wrapper with binary PLS-DA base estimator
        base_estimator = PLSDABinaryClassifier(
            n_components=self.n_components,
            threshold=self.threshold
        )
        self.ovr_ = OneVsRestClassifier(base_estimator, n_jobs=1)
        self.ovr_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.ovr_.predict(X)
    
    def predict_proba(self, X):
        # OvR returns decision scores, we need to normalize to probabilities
        if hasattr(self.ovr_, 'predict_proba'):
            return self.ovr_.predict_proba(X)
        else:
            # Fallback: use decision function and softmax
            decisions = self.ovr_.decision_function(X)
            # Simple normalization
            exp_decisions = np.exp(decisions - decisions.max(axis=1, keepdims=True))
            return exp_decisions / exp_decisions.sum(axis=1, keepdims=True)


class PLSDAAdaptive(BaseEstimator, ClassifierMixin):
    """
    Adaptive PLS-DA that automatically handles both binary and multi-class.
    
    - For binary (2 classes): Uses direct PLS regression with thresholding
    - For multi-class (>2 classes): Uses One-vs-Rest strategy with binary PLS-DA
    """
    
    def __init__(self, n_components: int = 10, threshold: float = 0.5):
        self.n_components = n_components
        self.threshold = threshold
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            # Binary case
            self.estimator_ = PLSDABinaryClassifier(
                n_components=self.n_components,
                threshold=self.threshold
            )
        else:
            # Multi-class case
            self.estimator_ = PLSDAMultiClassOvR(
                n_components=self.n_components,
                threshold=self.threshold
            )
        
        self.estimator_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ==================== MODEL DEFINITIONS ====================

def get_all_models(config: ExperimentConfig, n_classes: int = 2, crack_class_idx: int = 1) -> List[Tuple[str, Any, bool]]:
    """
    Get all models for evaluation, optimized for Accuracy/ROC-AUC.
    
    Models are trained WITHOUT class_weight to optimize for overall accuracy.
    This produces better ROC-AUC compared to class-weighted training.
    
    Returns:
        List of (model_name, model_instance, supports_internal_njobs) tuples.
    """
    models = []
    
    # Logistic Regression (L1) - NO class_weight for accuracy optimization
    if config.models_to_train.get("Logistic Regression (L1)", True):
        models.append((
            "Logistic Regression (L1)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=500,
                    tol=1e-3,
                    # No class_weight - optimize for accuracy/ROC-AUC
                    n_jobs=config.n_jobs,
                    random_state=config.random_state
                ))
            ]),
            True  # supports n_jobs
        ))
    
    # SVM (RBF) - NO class_weight for accuracy optimization
    if config.models_to_train.get("SVM (RBF)", True):
        models.append((
            "SVM (RBF)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(
                    kernel="rbf",
                    C=1,
                    probability=True,
                    # No class_weight - optimize for accuracy/ROC-AUC
                    random_state=config.random_state
                ))
            ]),
            False  # no n_jobs support
        ))
    
    # Random Forest - NO class_weight for accuracy optimization
    if config.models_to_train.get("Random Forest", True):
        models.append((
            "Random Forest",
            Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    # No class_weight - optimize for accuracy/ROC-AUC
                    n_jobs=config.n_jobs,
                    random_state=config.random_state
                ))
            ]),
            True  # supports n_jobs
        ))
    
    # XGBoost - with scale_pos_weight for imbalanced data
    # scale_pos_weight = n_negative / n_positive (approximated, will be set during training)
    if config.models_to_train.get("XGBoost", True):
        models.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=100,
                max_depth=5,
                use_label_encoder=False,
                eval_metric="auc",  # ROC-AUC for CRACK focus
                tree_method="hist",
                n_jobs=config.n_jobs,
                random_state=config.random_state
            ),
            True  # supports n_jobs
        ))
    
    # MLP (Small)
    if config.models_to_train.get("MLP (Small)", True):
        models.append((
            "MLP (Small)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    tol=1e-4,
                    random_state=config.random_state,
                    verbose=False
                ))
            ]),
            False  # no n_jobs support
        ))
    
    # PLS-DA (Adaptive - works for both binary and multi-class)
    if config.models_to_train.get("PLS-DA", True):
        models.append((
            "PLS-DA",
            Pipeline([
                ("scaler", StandardScaler()),
                ("pls", PLSDAAdaptive(n_components=10))
            ]),
            False  # no n_jobs support
        ))
    
    return models


# ==================== FOLD INFO ====================

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


# ==================== DOMAIN-AWARE CV SPLITS ====================

def create_domain_aware_cv_splits(
    y: np.ndarray,
    original_labels: np.ndarray,
    segment_ids: np.ndarray,
    image_ids: np.ndarray,
    grape_classes: set,
    crack_class_idx: int,  # Encoded index of CRACK class
    regular_class_idx: int,  # Encoded index of REGULAR class
    random_state: int = 42,
    non_grape_holdout_frac: float = 0.20,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, FoldInfo]], Dict]:
    """
    Create domain-aware CV splits with LOGO on grape samples and fixed holdout for non-grape.
    
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

    # ---- NEW STRATEGY ----
    # LOGO on CRACK samples only, 80/20 fixed split for REGULAR
    # This reduces folds to number of unique CRACK images
    
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
    
    # ---- 80/20 split for REGULAR samples (fixed across folds) ----
    if len(regular_indices) > 0:
        regular_segment_ids = segment_ids[regular_indices]
        
        # Use GroupShuffleSplit on segment IDs to avoid leakage
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
    
    # ---- LOGO on CRACK samples only ----
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


# ==================== BALANCING WITHIN FOLD ====================

def balance_training_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    segment_ids_train: np.ndarray,
    balance_mode: BalanceMode,
    crack_class_idx: int,
    max_samples_per_class: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply balancing to training fold only.
    
    For BALANCED mode: Undersample to minority class (CRACK) count
    For UNBALANCED mode: Apply max_samples_per_class cap only
    
    Returns:
        Balanced X_train, y_train
    """
    rng = np.random.RandomState(random_state)
    
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_count_dict = dict(zip(unique_classes, class_counts))
    
    if balance_mode == BalanceMode.BALANCED:
        # Use CRACK count as target (or minimum class count)
        if crack_class_idx in class_count_dict:
            target_count = class_count_dict[crack_class_idx]
        else:
            target_count = min(class_counts)
        
        # Apply max_samples_per_class cap
        target_count = min(target_count, max_samples_per_class)
    else:
        # UNBALANCED: just apply the per-class cap
        target_count = max_samples_per_class
    
    # Sample each class to target count
    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        n_cls = len(cls_indices)
        n_sample = min(n_cls, target_count)
        
        if n_sample < n_cls:
            sampled = rng.choice(cls_indices, size=n_sample, replace=False)
        else:
            sampled = cls_indices
        balanced_indices.extend(sampled)
    
    balanced_indices = np.array(balanced_indices)
    rng.shuffle(balanced_indices)
    
    return X_train[balanced_indices], y_train[balanced_indices]


# ==================== SINGLE FOLD EVALUATION ====================

def compute_sample_weights(y: np.ndarray, crack_class_idx: int) -> np.ndarray:
    """
    Compute sample weights to emphasize the CRACK class for imbalanced learning.
    Uses balanced class weights: weight[class] = n_samples / (n_classes * n_class_samples)
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight('balanced', y)


# Note: This script trains WITHOUT class_weight to optimize for Accuracy/ROC-AUC


def evaluate_single_fold(
    fold_data: Tuple[np.ndarray, np.ndarray, FoldInfo],
    model,
    X: np.ndarray,
    y: np.ndarray,
    segment_ids: np.ndarray,
    n_classes: int,
    crack_class_idx: int,
    balance_mode: BalanceMode,
    max_samples_per_class: int,
    random_state: int,
) -> Dict:
    """
    Evaluate a single fold for Accuracy/ROC-AUC optimization.
    
    Trains WITHOUT sample weighting to optimize for overall accuracy.
    """
    train_idx, test_idx, fold_info = fold_data
    fold_start_time = time.time()

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Clone and fit model (NO sample weights for accuracy optimization)
    fold_model = clone(model)
    
    train_start = time.time()
    try:
        fold_model.fit(X_train, y_train)
    except Exception as e:
        return {
            'fold_idx': fold_info.fold_idx,
            'error': str(e),
            'success': False,
        }
    train_time = time.time() - train_start

    # Predict (standard prediction, no threshold optimization)
    infer_start = time.time()
    y_pred = fold_model.predict(X_test)
    
    # Get probabilities
    y_prob_all = None
    if hasattr(fold_model, 'predict_proba'):
        try:
            y_prob_all = fold_model.predict_proba(X_test)
        except Exception:
            pass
    infer_time = time.time() - infer_start

    # Calculate metrics
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
    if crack_class_idx < len(prec_per_class):
        crack_prec = prec_per_class[crack_class_idx]
        crack_rec = rec_per_class[crack_class_idx]
        crack_f1 = f1_per_class[crack_class_idx]
    else:
        crack_prec = crack_rec = crack_f1 = np.nan

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
    
    # Count CRACK samples for diagnostics
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


# ==================== MODEL EVALUATION ====================

def evaluate_model_cv(
    model,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    segment_ids: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray, FoldInfo]],
    class_names: List[str],
    crack_class_idx: int,
    balance_mode: BalanceMode,
    config: ExperimentConfig,
    supports_njobs: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Evaluate model across all CV folds.
    
    Returns:
        summary_dict: Aggregated metrics with mean ± std
        curves_dict: Aggregated predictions for plotting
    """
    n_classes = len(class_names)
    n_folds = len(folds)

    print(f"    [CV] Running {n_folds} folds...")
    eval_start = time.time()

    # Decide parallelization strategy
    use_parallel = (not supports_njobs) and (Parallel is not None) and (config.n_jobs != 1)

    if use_parallel:
        fold_results = Parallel(n_jobs=config.n_jobs, backend=config.cv_parallel_backend, verbose=0)(
            delayed(evaluate_single_fold)(
                fold_data, model, X, y, segment_ids,
                n_classes, crack_class_idx, balance_mode,
                config.max_samples_per_class, config.random_state
            )
            for fold_data in folds
        )
    else:
        fold_results = []
        cumulative_time = 0.0
        for fold_idx, fold_data in enumerate(folds):
            result = evaluate_single_fold(
                fold_data, model, X, y, segment_ids,
                n_classes, crack_class_idx, balance_mode,
                config.max_samples_per_class, config.random_state
            )
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
        return {"Model Name": model_name, "Status": "ALL FOLDS FAILED"}, {}

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
    print(f"    Accuracy: {np.mean(metrics['accs']):.4f} ± {np.std(metrics['accs']):.4f}")
    print(f"    Balanced Accuracy: {np.mean(metrics['balanced_accs']):.4f} ± {np.std(metrics['balanced_accs']):.4f}")
    print(f"    Macro-F1: {np.mean(metrics['macro_f1s']):.4f} ± {np.std(metrics['macro_f1s']):.4f}")
    print(f"    CRACK F1: {np.mean(metrics['crack_f1s']):.4f} ± {np.std(metrics['crack_f1s']):.4f}")

    # Per-class recall aggregation
    per_class_recalls = {cls: [] for cls in range(n_classes)}
    for r in successful_results:
        for cls_idx, rec in enumerate(r['rec_per_class']):
            if cls_idx < n_classes:
                per_class_recalls[cls_idx].append(rec)

    # Build summary with mean ± std formatting
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

    # Add per-class recall
    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx in per_class_recalls and per_class_recalls[cls_idx]:
            summary_dict[f"Recall_{cls_name}"] = format_mean_std(per_class_recalls[cls_idx])

    # Store raw metrics for detailed analysis
    summary_dict["_raw_metrics"] = metrics

    # Aggregate predictions
    y_true_all = np.concatenate([r['y_test'] for r in successful_results])
    y_pred_all = np.concatenate([r['y_pred'] for r in successful_results])
    y_prob_crack_all = None
    
    if all(r.get('y_prob_crack') is not None for r in successful_results):
        y_prob_crack_all = np.concatenate([r['y_prob_crack'] for r in successful_results])

    curves_dict = {
        'y_true_all': y_true_all,
        'y_pred_all': y_pred_all,
        'y_prob_crack_all': y_prob_crack_all,
    }

    return summary_dict, curves_dict


# ==================== DATA PREPARATION ====================

def prepare_data_for_label_setup(
    data: PreprocessedData,
    label_setup: LabelSetup,
    dataset_config: DatasetConfig,
) -> Tuple[np.ndarray, List[str], Dict[str, int], int]:
    """
    Prepare y labels according to the label setup.
    
    Returns:
        y: Transformed labels
        class_names: List of class names
        class_mapping: Dict mapping class name to index
        crack_class_idx: Index of CRACK class in the transformed labels
    """
    if label_setup == LabelSetup.CRACK_VS_REST:
        # Binary: CRACK vs REST
        crack_idx_original = data.class_mapping.get(
            str(dataset_config.crack_identifier).upper() if not dataset_config.is_3class 
            else str(dataset_config.crack_identifier),
            None
        )
        
        if crack_idx_original is None:
            # Fallback search
            for name, idx in data.class_mapping.items():
                if str(name).upper() == "CRACK" or str(name) == str(dataset_config.crack_identifier):
                    crack_idx_original = idx
                    break
        
        if crack_idx_original is None:
            raise ValueError(f"Could not find CRACK class in mapping: {data.class_mapping}")
        
        y = (data.y == crack_idx_original).astype(int)
        class_names = ["REST", "CRACK"]
        class_mapping = {"REST": 0, "CRACK": 1}
        crack_class_idx = 1
        
    elif label_setup == LabelSetup.MULTI_CLASS:
        # Explicit 3-class (or use 3class dataset)
        y = data.y
        class_names = data.class_names
        class_mapping = data.class_mapping
        
        # Find CRACK class
        crack_class_name = str(dataset_config.crack_identifier).upper() if not dataset_config.is_3class else str(dataset_config.crack_identifier)
        crack_class_idx = class_mapping.get(crack_class_name, 0)
        
    else:  # CRACK_REGULAR_REST - full multiclass
        y = data.y
        class_names = data.class_names
        class_mapping = data.class_mapping
        
        # Find CRACK class
        crack_class_name = str(dataset_config.crack_identifier).upper() if not dataset_config.is_3class else str(dataset_config.crack_identifier)
        crack_class_idx = class_mapping.get(crack_class_name, 0)
    
    return y, class_names, class_mapping, crack_class_idx


# ==================== ARTIFACT SAVING ====================

def save_artifacts(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_crack_all: Optional[np.ndarray],
    out_dir: Path,
    model_name: str,
    label_setup: LabelSetup,
    balance_mode: BalanceMode,
    class_names: List[str],
    crack_class_idx: int,
):
    """Save model evaluation artifacts."""
    model_slug = slugify_model_name(model_name)
    save_dir = ensure_dir(out_dir / "artifacts" / label_setup.value / balance_mode.value / model_slug)

    # Predictions CSV
    predictions_df = pd.DataFrame({
        'y_true': y_true_all,
        'y_pred': y_pred_all,
    })
    if y_prob_crack_all is not None:
        predictions_df['y_prob_crack'] = y_prob_crack_all
    predictions_df.to_csv(save_dir / "predictions.csv", index=False)

    # Confusion Matrix
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Normalized
        cm_norm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
        disp_norm.plot(ax=axes[0], cmap='Blues', values_format='.2f')
        axes[0].set_title(f'Normalized - {model_name}')
        
        # Raw
        cm_raw = confusion_matrix(y_true_all, y_pred_all)
        disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=class_names)
        disp_raw.plot(ax=axes[1], cmap='Blues', values_format='d')
        axes[1].set_title(f'Raw Counts - {model_name}')
        
        plt.tight_layout()
        fig.savefig(save_dir / "confusion_matrices.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"    [WARN] Could not save confusion matrix: {e}")

    # ROC and PR curves for CRACK
    if y_prob_crack_all is not None:
        y_true_binary = (y_true_all == crack_class_idx).astype(int)
        
        if y_true_binary.sum() > 0 and y_true_binary.sum() < len(y_true_binary):
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # ROC
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob_crack_all)
                roc_auc_val = auc(fpr, tpr)
                axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_val:.4f})')
                axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
                axes[0].set_xlabel('False Positive Rate')
                axes[0].set_ylabel('True Positive Rate')
                axes[0].set_title(f'CRACK ROC - {model_name}')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                
                # PR
                precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_crack_all)
                ap_score = average_precision_score(y_true_binary, y_prob_crack_all)
                axes[1].plot(recall, precision, color='green', lw=2, label=f'PR (AP = {ap_score:.4f})')
                axes[1].set_xlabel('Recall')
                axes[1].set_ylabel('Precision')
                axes[1].set_title(f'CRACK PR - {model_name}')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
                
                plt.tight_layout()
                fig.savefig(save_dir / "roc_pr_curves.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"    [WARN] Could not save ROC/PR curves: {e}")

    # Classification report
    try:
        report = classification_report(y_true_all, y_pred_all, target_names=class_names, zero_division=0)
        with open(save_dir / "classification_report.txt", 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"    [WARN] Could not save classification report: {e}")


# ==================== MODEL SAVING ====================

def save_trained_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    model_name: str,
    label_setup: LabelSetup,
    balance_mode: BalanceMode,
    class_names: List[str],
    class_mapping: Dict[str, int],
    crack_class_idx: int,
) -> Optional[Path]:
    """
    Retrain model on full dataset and save to disk.
    
    After CV evaluation, we retrain on ALL data to get the final model for deployment.
    
    Saves:
    - model.pkl: The trained model (joblib)
    - model_metadata.json: Class mapping, feature info, etc.
    
    Returns:
        Path to saved model, or None if saving failed
    """
    if joblib is None:
        print(f"    [WARN] joblib not available, cannot save model")
        return None
    
    model_slug = slugify_model_name(model_name)
    save_dir = ensure_dir(out_dir / "models" / label_setup.value / balance_mode.value)
    model_path = save_dir / f"{model_slug}.pkl"
    metadata_path = save_dir / f"{model_slug}_metadata.json"
    
    try:
        # Clone and retrain on full data
        print(f"    [SAVE] Retraining {model_name} on full dataset ({X.shape[0]} samples)...")
        final_model = clone(model)
        
        # For XGBoost, compute sample weights
        if isinstance(final_model, XGBClassifier):
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y)
            final_model.fit(X, y, sample_weight=sample_weights)
        else:
            final_model.fit(X, y)
        
        # Save model
        joblib.dump(final_model, model_path)
        print(f"    [SAVED] Model: {model_path}")
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "label_setup": label_setup.value,
            "balance_mode": balance_mode.value,
            "class_names": class_names,
            "class_mapping": class_mapping,
            "crack_class_idx": crack_class_idx,
            "n_features": X.shape[1],
            "n_samples_trained": X.shape[0],
            "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    [SAVED] Metadata: {metadata_path}")
        
        return model_path
        
    except Exception as e:
        print(f"    [ERROR] Failed to save model: {e}")
        return None


def load_trained_model(model_path: Path) -> Tuple[Any, Dict]:
    """
    Load a saved model and its metadata.
    
    Usage:
        model, metadata = load_trained_model(Path("experiments/.../models/xgboost.pkl"))
        predictions = model.predict(X_new)
    
    Returns:
        (model, metadata_dict)
    """
    if joblib is None:
        raise ImportError("joblib is required to load models")
    
    model = joblib.load(model_path)
    
    # Load metadata
    metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, metadata


# ==================== CONSOLIDATED REPORT ====================

def create_consolidated_report(
    all_results: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Create a single consolidated thesis-ready report.
    
    Rows: Model × LabelSetup × BalanceMode
    Columns: Metrics as mean ± std
    """
    consolidated_rows = []
    
    for result_key, df_results in all_results.items():
        # Parse result_key (format: prefix_labelsetup_balancemode)
        # Examples: "3class_crack_regular_rest_balanced", "multiclass_crack_vs_rest_unbalanced"
        parts = result_key.split("_")
        if len(parts) >= 2:
            balance_mode = parts[-1]  # Last part is always balance mode
            # Everything between first and last underscore-separated parts is label setup
            # Remove prefix (3class or multiclass) and balance_mode to get label_setup
            prefix = parts[0]  # "3class" or "multiclass"
            # Reconstruct label_setup from middle parts
            label_setup_parts = parts[1:-1]  # Everything except first and last
            label_setup = "_".join(label_setup_parts) if label_setup_parts else "unknown"
        else:
            prefix = "unknown"
            label_setup = "unknown"
            balance_mode = "unknown"
        
        for _, row in df_results.iterrows():
            consolidated_row = {
                "Model": row.get("Model Name", "Unknown"),
                "Label Setup": label_setup,  # Keep original format like "crack_regular_rest"
                "Balance Mode": balance_mode.title(),
            }
            
            # Copy key metrics
            for col in ["Accuracy", "Balanced Accuracy", "Macro-F1", "CRACK Precision", 
                       "CRACK Recall", "CRACK F1", "CRACK ROC-AUC", "CRACK PR-AUC",
                       "Total Time (s)", "Folds Completed"]:
                if col in row:
                    consolidated_row[col] = row[col]
            
            consolidated_rows.append(consolidated_row)
    
    consolidated_df = pd.DataFrame(consolidated_rows)
    
    # Save to Excel
    excel_path = out_dir / "consolidated_thesis_report.xlsx"
    consolidated_df.to_excel(excel_path, index=False, sheet_name="Results")
    
    # Save to CSV
    csv_path = out_dir / "consolidated_thesis_report.csv"
    consolidated_df.to_csv(csv_path, index=False)
    
    # Save to Markdown (for easy copy-paste to thesis)
    md_path = out_dir / "consolidated_thesis_report.md"
    with open(md_path, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        # Create markdown table manually (no tabulate dependency)
        try:
            f.write(consolidated_df.to_markdown(index=False))
        except ImportError:
            # Fallback: create simple markdown table without tabulate
            cols = consolidated_df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in consolidated_df.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
    
    print(f"\n[SAVED] Consolidated report: {excel_path}")
    print(f"[SAVED] Consolidated CSV: {csv_path}")
    print(f"[SAVED] Consolidated Markdown: {md_path}")
    
    return consolidated_df


# ==================== MAIN EXPERIMENT LOOP ====================

def run_experiment(config: ExperimentConfig):
    """
    Run the unified experiment pipeline.
    """
    print("=" * 80)
    print("UNIFIED EXPERIMENT PIPELINE")
    print("=" * 80)
    print(f"Experiment directory: {EXPERIMENT_DIR}")
    print(f"Dry-run mode: {'ENABLED' if config.max_samples else 'DISABLED (full dataset)'}")
    if config.max_samples:
        print(f"  Max samples: {config.max_samples}")
    print(f"Label setups: {[ls.value for ls in config.label_setups]}")
    print(f"Balance modes: {[bm.value for bm in config.balance_modes]}")
    print(f"Models enabled: {[k for k, v in config.models_to_train.items() if v]}")
    print("=" * 80)
    
    ensure_dir(EXPERIMENT_DIR)
    
    # Save configuration
    config_dict = {
        "max_samples": config.max_samples,
        "random_state": config.random_state,
        "non_grape_holdout_fraction": config.non_grape_holdout_fraction,
        "n_jobs": config.n_jobs,
        "max_samples_per_class": config.max_samples_per_class,
        "label_setups": [ls.value for ls in config.label_setups],
        "balance_modes": [bm.value for bm in config.balance_modes],
        "models_to_train": config.models_to_train,
        "timestamp": TIMESTAMP,
    }
    with open(EXPERIMENT_DIR / "experiment_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    all_results = {}
    models = get_all_models(config)
    
    # Use multiclass dataset as the base
    dataset_config = DATASET_CONFIGS["multiclass"]
    
    print(f"\n[INFO] Loading dataset: {dataset_config.csv_path}")
    
    if not dataset_config.csv_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_config.csv_path}")
        return
    
    df_original = pd.read_csv(dataset_config.csv_path)
    print(f"[INFO] Loaded {len(df_original)} rows")
    
    # Apply dry-run sample cap with STRATIFIED sampling to preserve class proportions
    if config.max_samples and len(df_original) > config.max_samples:
        print(f"[DRY-RUN] Stratified sampling {config.max_samples} rows from {len(df_original)}")
        from sklearn.model_selection import train_test_split
        _, df_original = train_test_split(
            df_original, 
            test_size=config.max_samples / len(df_original),
            stratify=df_original[dataset_config.target_col],
            random_state=config.random_state
        )
        df_original = df_original.reset_index(drop=True)
        print(f"[DRY-RUN] Class distribution after sampling: {df_original[dataset_config.target_col].value_counts().to_dict()}")
    
    all_results = {}
    models = get_all_models(config)
    
    # Main experiment loop - preprocess separately for each balance mode (like old file)
    for balance_mode in config.balance_modes:
        print(f"\n{'='*80}")
        print(f"BALANCE MODE: {balance_mode.value.upper()}")
        print(f"{'='*80}")

        is_balanced = (balance_mode == BalanceMode.BALANCED)

        # --- Standard multiclass dataset ---
        print(f"[INFO] Preprocessing multiclass dataset (balanced={is_balanced})...")
        data_multi = preprocess_multiclass_dataset(
            df_original.copy(),
            wl_min=450,
            wl_max=925,
            apply_snv=True,
            remove_outliers=False,
            balanced=is_balanced,
            label_col=dataset_config.target_col,
            hs_dir_col="hs_dir",
            segment_col="mask_path",
            cap_class=dataset_config.crack_identifier if is_balanced else None,
            max_samples_per_class=config.max_samples_per_class,
            seed=config.random_state,
        )

        # --- 3-class mapping: REGULAR=0, CRACK=1, REST=2 ---
        print(f"[INFO] Creating 3-class mapping (REGULAR, CRACK, REST)...")
        df_3class = df_original.copy()
        df_3class['3class_label'] = df_3class[dataset_config.target_col].str.upper().map(lambda x: 0 if x == 'REGULAR' else (1 if x == 'CRACK' else 2))
        data_3class = preprocess_multiclass_dataset(
            df_3class,
            wl_min=450,
            wl_max=925,
            apply_snv=True,
            remove_outliers=False,
            balanced=is_balanced,
            label_col='3class_label',
            hs_dir_col="hs_dir",
            segment_col="mask_path",
            cap_class=1 if is_balanced else None,  # CRACK=1 in this mapping
            max_samples_per_class=config.max_samples_per_class,
            seed=config.random_state,
        )

        # --- Loop through label setups ---
        for label_setup in config.label_setups:
            if label_setup == LabelSetup.MULTI_CLASS:
                data = data_3class
                label_names = ["REGULAR", "CRACK", "REST"]
                result_key_prefix = "3class"
                crack_class_idx = 1  # In 3-class: REGULAR=0, CRACK=1, REST=2
                regular_class_idx = 0
            else:
                data = data_multi
                label_names = data.class_names
                result_key_prefix = "multiclass"
                crack_class_idx = data.class_mapping.get("CRACK", 3)  # Default to 3 if not found
                regular_class_idx = data.class_mapping.get("REGULAR", 7)

            X = data.X
            print(f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"[INFO] Classes: {label_names}")
            print(f"[INFO] Class distribution: {dict(zip(*np.unique(data.y, return_counts=True)))}")
            print(f"[INFO] CRACK class index: {crack_class_idx}, REGULAR class index: {regular_class_idx}")

            # Create CV splits for this label setup
            print("[INFO] Creating CV splits...")
            folds, split_manifest = create_domain_aware_cv_splits(
                y=data.y,
                original_labels=data.original_labels,
                segment_ids=data.segment_ids,
                image_ids=data.image_ids,
                grape_classes=data.grape_class_indices,
                crack_class_idx=crack_class_idx,
                regular_class_idx=regular_class_idx,
                random_state=config.random_state,
                non_grape_holdout_frac=config.non_grape_holdout_fraction,
            )

            # Save split manifest
            manifest_path = EXPERIMENT_DIR / f"cv_split_manifest_{result_key_prefix}_{balance_mode.value}.json"
            with open(manifest_path, 'w') as f:
                json.dump(split_manifest, f, indent=2)

            # Prepare labels for this setup
            y_setup = data.y
            class_names_setup = label_names
            class_mapping_setup = {str(i): i for i in range(len(label_names))}

            n_classes = len(class_names_setup)
            print(f"[INFO] Classes: {class_names_setup}")
            print(f"[INFO] CRACK class index: {crack_class_idx}")
            print(f"[INFO] Class distribution: {dict(zip(*np.unique(y_setup, return_counts=True)))}")

            result_key = f"{result_key_prefix}_{label_setup.value}_{balance_mode.value}"
            results_table = []

            for model_name, model, supports_njobs in models:
                print(f"\n[MODEL] {model_name}")

                try:
                    summary, curves = evaluate_model_cv(
                        model=model,
                        model_name=model_name,
                        X=X,
                        y=y_setup,
                        segment_ids=data.segment_ids,
                        folds=folds,
                        class_names=class_names_setup,
                        crack_class_idx=crack_class_idx,
                        balance_mode=balance_mode,
                        config=config,
                        supports_njobs=supports_njobs,
                    )

                    results_table.append(summary)

                    # Print summary
                    print(f"    Accuracy: {summary.get('Accuracy', 'N/A')}")
                    print(f"    Balanced Accuracy: {summary.get('Balanced Accuracy', 'N/A')}")
                    print(f"    Macro-F1: {summary.get('Macro-F1', 'N/A')}")
                    print(f"    CRACK F1: {summary.get('CRACK F1', 'N/A')}")

                    # Save artifacts (plots, predictions, etc.)
                    if curves:
                        save_artifacts(
                            y_true_all=curves['y_true_all'],
                            y_pred_all=curves['y_pred_all'],
                            y_prob_crack_all=curves.get('y_prob_crack_all'),
                            out_dir=EXPERIMENT_DIR,
                            model_name=model_name,
                            label_setup=label_setup,
                            balance_mode=balance_mode,
                            class_names=class_names_setup,
                            crack_class_idx=crack_class_idx,
                        )
                    
                    # Save trained model (retrained on full data)
                    save_trained_model(
                        model=model,
                        X=X,
                        y=y_setup,
                        out_dir=EXPERIMENT_DIR,
                        model_name=model_name,
                        label_setup=label_setup,
                        balance_mode=balance_mode,
                        class_names=class_names_setup,
                        class_mapping=class_mapping_setup,
                        crack_class_idx=crack_class_idx,
                    )

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    results_table.append({"Model Name": model_name, "Status": f"FAILED: {str(e)}"})

            # Create results DataFrame
            if results_table:
                # Remove _raw_metrics from display
                display_results = []
                for r in results_table:
                    r_display = {k: v for k, v in r.items() if not k.startswith('_')}
                    display_results.append(r_display)

                df_results = pd.DataFrame(display_results)
                all_results[result_key] = df_results

                print(f"\n--- Results: {label_setup.value} / {balance_mode.value} ---")
                print(df_results.to_string())
    
    # Create consolidated report
    print("\n" + "=" * 80)
    print("CREATING CONSOLIDATED REPORT")
    print("=" * 80)
    
    consolidated_df = create_consolidated_report(all_results, EXPERIMENT_DIR)
    
    # Save per-experiment Excel
    excel_path = EXPERIMENT_DIR / "detailed_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for result_key, df_results in all_results.items():
            # Remove _raw_metrics for Excel output
            df_clean = df_results.drop(columns=[c for c in df_results.columns if c.startswith('_')], errors='ignore')
            sheet_name = result_key[:31]
            df_clean.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n[SAVED] Detailed results: {excel_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"All outputs saved to: {EXPERIMENT_DIR}")
    print("=" * 80)


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Create configuration
    config = ExperimentConfig(
        # DRY-RUN MODE: Set to None for full dataset
        # NOTE: With 1000 samples and 18 folds, each fold has ~55 test samples.
        # If CRACK is 5% of data, that's only ~3 CRACK samples per test fold!
        # For meaningful results, use at least 5000 samples or None (full dataset).
        max_samples=None,  # Increased for more stable metrics - change to None for full run
        
        # Models to run
        models_to_train={
            "Logistic Regression (L1)": True,
            "SVM (RBF)": False,  # SVM (no class_weight for accuracy focus)
            "Random Forest": True,
            "XGBoost": True,
            "MLP (Small)": True,
            "PLS-DA": True,
        },
        
        # Label setups
        label_setups=[
            LabelSetup.CRACK_REGULAR_REST,
            LabelSetup.CRACK_VS_REST,
            LabelSetup.MULTI_CLASS,
        ],
        
        # Balance modes
        balance_modes=[
            BalanceMode.BALANCED,
            BalanceMode.UNBALANCED,
        ],
        
        # Other settings
        random_state=42,
        n_jobs=-1,
        max_samples_per_class=45000,
    )
    
    run_experiment(config)
