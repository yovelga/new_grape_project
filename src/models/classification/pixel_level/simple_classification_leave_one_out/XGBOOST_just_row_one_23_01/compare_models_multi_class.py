"""
compare_models_multi_class.py

Multi-class model comparison benchmark for pixel-level classification.

Features:
- Supports both multi-class (all labels) and 3-class (REGULAR/CRACK/not_grape) datasets
- Domain-aware CV split: LOGO on grape samples, fixed holdout for non-grape distractors
- Cap-based segment-proportional undersampling (CRACK count as cap)
- Multi-class evaluation with CRACK-focused metrics
- Strict leakage prevention (no segment_id in both train/test)

Outputs:
- Per-model predictions, confusion matrices, ROC/PR curves (CRACK one-vs-rest)
- Excel summary with CRACK precision/recall/F1/AUC prominently displayed
- Split manifest for reproducibility verification
"""

import time
import warnings
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Import the reusable preprocessing function
from src.preprocessing.spectral_preprocessing import (
    preprocess_multiclass_dataset,
    PreprocessedData,
    save_class_mapping,
    GRAPE_CLASSES,
    GRAPE_CLASS_IDS_3CLASS,
)
from datetime import datetime

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


# ==================== CONFIGURATION ====================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    csv_path: Path
    target_col: str
    crack_identifier: Union[str, int]  # "CRACK" for multiclass, 2 for 3class
    grape_classes: set  # Classes that are grape-related for domain split
    is_3class: bool = False


# Dataset configurations
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
    DatasetConfig(
        name="3class",
        csv_path=CSV_PATH_3CLASS,
        target_col="label_3class_id",
        crack_identifier=2,
        grape_classes={1, 2},  # REGULAR=1, CRACK=2 in 3-class
        is_3class=True,
    ),
]

# Create timestamped experiment directory
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = Path(rf"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class_{TIMESTAMP}")
# EXPERIMENT_DIR = Path(r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_compare_multiclass_vs_3class")
NON_GRAPE_HOLDOUT_FRACTION = 0.20
RANDOM_STATE = 42

# ==================== BALANCING SETTINGS ====================
# Maximum samples per class for BOTH balanced and unbalanced modes. Options:
#   - None: Use only cap_class count (e.g., CRACK count) as the maximum (balanced mode only)
#   - Integer (e.g., 10000): Use this specific number as the maximum per class
# If both cap_class and MAX_SAMPLES_PER_CLASS are set, the minimum is used.
# Applied in BOTH balanced and unbalanced modes to prevent SVM from running too long.
MAX_SAMPLES_PER_CLASS = 50000 # Reduced to 10k to prevent SVM from getting stuck (was 50000)

# ==================== PARALLELIZATION SETTINGS ====================
N_JOBS = -1  # Use all available CPU cores (-1 = all cores)
CV_PARALLEL_BACKEND = "loky"  # 'loky' is more robust than 'multiprocessing'

# ==================== MODEL SELECTION ====================
# Set to True to train the model, False to skip it
# This allows you to easily choose which models to train
MODELS_TO_TRAIN = {
    "SVM (RBF)": False,                    # SVM is first - set to True to train
    "PLS-DA": True,                       # PLS-DA (will be run as binary in CRACK_vs_REST mode)
    "Logistic Regression (L1)": True,
    "Random Forest": True,
    "XGBoost": True,
    "MLP (Small)": True,                  # Small MLP with 3-4 layers
}

# ==================== EVALUATION MODES ====================
# For each dataset (multiclass, 3class), we can evaluate in different modes:
# 1. "full" - Use all classes as-is (6 classes for multiclass, 3 classes for 3class)
# 2. "crack_vs_rest" - Binary evaluation: CRACK vs all other classes combined
# Set to True to enable, False to skip
EVALUATION_MODES = {
    "full": True,           # Full multi-class or 3-class evaluation
    "crack_vs_rest": True,  # Binary CRACK vs REST evaluation for all models
}


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


# ==================== PLS-DA WRAPPER (CRACK vs REST BINARY) ====================

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS-DA classifier using PLSRegression with thresholding.

    For multi-class problems, this automatically converts to binary:
    - Class 1 (positive): CRACK class (specified by crack_class_idx)
    - Class 0 (negative): All other classes (REST)

    This allows PLS-DA to be used with multi-class datasets by treating it
    as a CRACK detection problem (CRACK vs REST).

    IMPORTANT: Predictions are returned in original multi-class space:
    - CRACK predictions return crack_class_idx
    - REST predictions return 0 (or the most common non-CRACK class)
    """
    BINARY_ONLY = False  # No longer binary-only - handles multi-class via binarization
    IS_BINARY_WRAPPER = True  # Flag to indicate this is a binary wrapper for multi-class

    def __init__(self, n_components=10, threshold=0.5, crack_class_idx=None):
        self.n_components = n_components
        self.threshold = threshold
        self.crack_class_idx = crack_class_idx  # Will be set dynamically if None

    def fit(self, X, y):
        self.original_classes_ = np.unique(y)

        # Determine crack_class_idx if not set
        if self.crack_class_idx is None:
            # Default: assume highest class index is CRACK (common convention)
            self.crack_class_idx_ = int(max(self.original_classes_))
        else:
            self.crack_class_idx_ = int(self.crack_class_idx)

        # Find the most common non-CRACK class for REST predictions
        non_crack_classes = [c for c in self.original_classes_ if c != self.crack_class_idx_]
        if non_crack_classes:
            # Use the most frequent non-CRACK class as the default REST prediction
            from collections import Counter
            non_crack_labels = y[y != self.crack_class_idx_]
            if len(non_crack_labels) > 0:
                self.rest_class_idx_ = Counter(non_crack_labels).most_common(1)[0][0]
            else:
                self.rest_class_idx_ = non_crack_classes[0]
        else:
            self.rest_class_idx_ = 0

        # Convert to binary: CRACK=1, REST=0
        y_binary = (y == self.crack_class_idx_).astype(int)

        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, y_binary)

        # For sklearn compatibility, classes_ should contain the actual class indices used
        # We expose [rest_class, crack_class] so metrics can find CRACK
        self.classes_ = np.array([self.rest_class_idx_, self.crack_class_idx_])

        return self

    def predict(self, X):
        """
        Predict in original multi-class space.
        Returns crack_class_idx for CRACK predictions, rest_class_idx for REST.
        """
        binary_pred = (self.pls_.predict(X).ravel() >= self.threshold).astype(int)
        # Map back: 1 -> crack_class_idx, 0 -> rest_class_idx
        return np.where(binary_pred == 1, self.crack_class_idx_, self.rest_class_idx_)

    def predict_proba(self, X):
        """
        Return probabilities for [REST, CRACK] (2 columns).
        Column order matches self.classes_ = [rest_class_idx, crack_class_idx]
        """
        scores = self.pls_.predict(X).ravel()
        p_crack = np.clip(scores, 0, 1)
        return np.column_stack([1 - p_crack, p_crack])


# ==================== MODEL DEFINITIONS ====================

def get_models(n_classes: int) -> List[Tuple[str, Any, bool, bool]]:
    """
    Get list of models to evaluate.

    Returns:
        List of (model_name, model, binary_only, supports_njobs) tuples.
    """
    models = [
        # SVM (RBF): does NOT support n_jobs -> CV-level parallel (FIRST for priority testing)
        ("SVM (RBF)", Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1, probability=True, random_state=RANDOM_STATE))
        ]), False, False),

        # PLS-DA: Binary-only model, no n_jobs support -> CV-level parallel
        # Note: Only used in crack_vs_rest mode (converts multi-class to binary internally)
        ("PLS-DA", Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSDAClassifier(n_components=10))
        ]), True, False),  # binary_only=True - only runs in crack_vs_rest mode

        # Logistic Regression: supports n_jobs with saga solver
        ("Logistic Regression (L1)", Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=500,
                tol=1e-3,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE
            ))
        ]), False, True),


        # Random Forest: supports n_jobs
        ("Random Forest", Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE
            ))
        ]), False, True),

        # XGBoost: supports n_jobs
        ("XGBoost", XGBClassifier(
            n_estimators=100,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ), False, True),

        # Small MLP: 3-4 hidden layers with decreasing sizes
        # Architecture: input -> 128 -> 64 -> 32 -> output
        # Uses ReLU activation, Adam optimizer, and early stopping
        ("MLP (Small)", Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4,
                random_state=RANDOM_STATE,
                verbose=False
            ))
        ]), False, False),  # Does not support n_jobs for CV parallelization
    ]
    return models


def is_model_compatible(model_tuple: Tuple[str, Any, bool, bool], n_classes: int) -> bool:
    """Check if model is compatible with the number of classes and if it's enabled in MODELS_TO_TRAIN."""
    model_name, model, binary_only, supports_njobs = model_tuple

    # Check if model is enabled in MODELS_TO_TRAIN
    if model_name in MODELS_TO_TRAIN and not MODELS_TO_TRAIN[model_name]:
        print(f"[SKIP] {model_name} is disabled in MODELS_TO_TRAIN")
        return False

    if binary_only and n_classes > 2:
        print(f"[SKIP] {model_name} is binary-only, skipping for {n_classes}-class problem")
        return False
    return True


# ==================== DOMAIN-AWARE CV SPLITS ====================

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
    random_state: int = 42,
    non_grape_holdout_frac: float = 0.20,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, FoldInfo]], Dict]:
    """
    Create domain-aware CV splits with LOGO on grape samples and fixed holdout for non-grape.

    Logic:
    1. Split samples into grape-related (REGULAR/CRACK) and non-grape (distractors)
    2. Create fixed non-grape holdout (20%) by hs_dir (image-level), fallback to mask_path
    3. Apply LOGO on grape samples using hs_dir as group
    4. Each fold: Test = grape from held-out group + non_grape_test
                  Train = remaining grape + non_grape_train
    5. Verify no segment_id leakage between train and test

    Args:
        y: Encoded labels (integers)
        original_labels: Original string labels (for grape detection)
        segment_ids: Segment IDs (mask_path) for leakage prevention
        image_ids: Image IDs (hs_dir) for grouping
        grape_classes: Set of grape-related class names/IDs
        random_state: Random seed for reproducibility
        non_grape_holdout_frac: Fraction of non-grape images for holdout

    Returns:
        List of (train_idx, test_idx, fold_info) tuples
        Split manifest dict for reproducibility
    """
    n_samples = len(y)
    all_indices = np.arange(n_samples)

    # Determine if labels are strings or integers
    if isinstance(list(grape_classes)[0], str):
        # String labels - compare with original_labels (uppercase)
        is_grape = np.array([str(lbl).upper() in grape_classes for lbl in original_labels])
    else:
        # Integer labels - compare with y directly
        is_grape = np.isin(y, list(grape_classes))

    grape_indices = all_indices[is_grape]
    non_grape_indices = all_indices[~is_grape]

    print(f"[SPLIT] Total samples: {n_samples}")
    print(f"[SPLIT] Grape samples: {len(grape_indices)}")
    print(f"[SPLIT] Non-grape samples: {len(non_grape_indices)}")

    # --- Fixed non-grape holdout ---
    if len(non_grape_indices) > 0:
        non_grape_image_ids = image_ids[non_grape_indices]
        non_grape_segment_ids = segment_ids[non_grape_indices]
        unique_non_grape_images = np.unique(non_grape_image_ids)

        # Try image-level (hs_dir) grouping first
        gss = GroupShuffleSplit(n_splits=1, test_size=non_grape_holdout_frac, random_state=random_state)

        # Create dummy y for GroupShuffleSplit (we're just splitting indices)
        dummy_y = np.zeros(len(non_grape_indices))

        try:
            ng_train_idx_local, ng_test_idx_local = next(gss.split(
                non_grape_indices, dummy_y, groups=non_grape_image_ids
            ))

            # Check if any class is entirely missing from train after split
            ng_train_labels = y[non_grape_indices[ng_train_idx_local]]
            ng_test_labels = y[non_grape_indices[ng_test_idx_local]]

            train_classes = set(np.unique(ng_train_labels))
            test_classes = set(np.unique(ng_test_labels))
            all_ng_classes = set(np.unique(y[non_grape_indices]))

            # Fallback if any class is missing from train
            if train_classes != all_ng_classes:
                print(f"[SPLIT] WARNING: Image-level split would miss classes in train. Falling back to segment-level.")
                gss_segment = GroupShuffleSplit(n_splits=1, test_size=non_grape_holdout_frac, random_state=random_state)
                ng_train_idx_local, ng_test_idx_local = next(gss_segment.split(
                    non_grape_indices, dummy_y, groups=non_grape_segment_ids
                ))
                grouping_used = "segment (mask_path)"
            else:
                grouping_used = "image (hs_dir)"

        except ValueError as e:
            print(f"[SPLIT] WARNING: Image-level split failed ({e}). Falling back to segment-level.")
            gss_segment = GroupShuffleSplit(n_splits=1, test_size=non_grape_holdout_frac, random_state=random_state)
            ng_train_idx_local, ng_test_idx_local = next(gss_segment.split(
                non_grape_indices, dummy_y, groups=non_grape_segment_ids
            ))
            grouping_used = "segment (mask_path)"

        non_grape_train_global = non_grape_indices[ng_train_idx_local]
        non_grape_test_global = non_grape_indices[ng_test_idx_local]

        # Verification: no overlap in image_ids and segment_ids
        ng_train_images = set(image_ids[non_grape_train_global])
        ng_test_images = set(image_ids[non_grape_test_global])
        ng_train_segments = set(segment_ids[non_grape_train_global])
        ng_test_segments = set(segment_ids[non_grape_test_global])

        image_intersection = ng_train_images & ng_test_images
        segment_intersection = ng_train_segments & ng_test_segments

        print(f"[SPLIT] Non-grape holdout grouping: {grouping_used}")
        print(f"[SPLIT] Non-grape train images: {len(ng_train_images)}, test images: {len(ng_test_images)}")
        print(f"[SPLIT] Non-grape image intersection: {len(image_intersection)} (should be 0)")
        print(f"[SPLIT] Non-grape segment intersection: {len(segment_intersection)} (should be 0)")

        if segment_intersection:
            raise ValueError(f"LEAKAGE DETECTED: {len(segment_intersection)} segments in both non-grape train and test!")
    else:
        non_grape_train_global = np.array([], dtype=int)
        non_grape_test_global = np.array([], dtype=int)
        ng_train_images = set()
        ng_test_images = set()
        grouping_used = "N/A (no non-grape samples)"

    # --- LOGO on grape samples ---
    grape_image_ids = image_ids[grape_indices]
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
        "folds": [],
    }

    for fold_idx, (grape_train_local, grape_test_local) in enumerate(
        logo.split(grape_indices, y[grape_indices], groups=grape_image_ids)
    ):
        grape_train_global = grape_indices[grape_train_local]
        grape_test_global = grape_indices[grape_test_local]

        # Combine: train = grape_train + non_grape_train
        #          test = grape_test + non_grape_test
        train_idx = np.concatenate([grape_train_global, non_grape_train_global])
        test_idx = np.concatenate([grape_test_global, non_grape_test_global])

        # Strict leakage guard: no segment_id in both train and test
        train_segments = set(segment_ids[train_idx])
        test_segments = set(segment_ids[test_idx])
        segment_leak = train_segments & test_segments

        if segment_leak:
            raise ValueError(
                f"LEAKAGE DETECTED in fold {fold_idx}: {len(segment_leak)} segments in both train and test!\n"
                f"Leaked segments: {list(segment_leak)[:5]}..."
            )

        # Create fold info
        held_out_group = str(np.unique(grape_image_ids[grape_test_local])[0])
        train_grape_groups = list(np.unique(grape_image_ids[grape_train_local]))

        fold_info = FoldInfo(
            fold_idx=fold_idx,
            grape_group_held_out=held_out_group,
            n_train=len(train_idx),
            n_test=len(test_idx),
            train_grape_groups=train_grape_groups,
            test_grape_group=held_out_group,
            non_grape_train_images=sorted(list(ng_train_images)),
            non_grape_test_images=sorted(list(ng_test_images)),
        )

        folds.append((train_idx, test_idx, fold_info))

        # Add to manifest
        fold_manifest["folds"].append({
            "fold_idx": fold_idx,
            "grape_group_held_out": held_out_group,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_segments": len(train_segments),
            "test_segments": len(test_segments),
        })

    print(f"[SPLIT] Created {len(folds)} LOGO folds on grape samples")

    return folds, fold_manifest


# ==================== SINGLE FOLD EVALUATION ====================

def _evaluate_single_fold(
    fold_data: Tuple[np.ndarray, np.ndarray, FoldInfo],
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    crack_class_idx: int,
    fold_total: int,
) -> Dict:
    """
    Evaluate a single fold. This function is designed to be called in parallel.

    Returns a dict with all metrics and predictions for this fold.
    """
    import time as time_module
    from datetime import datetime

    train_idx, test_idx, fold_info = fold_data
    fold_start_time = time_module.time()
    fold_start_str = datetime.now().strftime("%H:%M:%S")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Debug: CRACK counts
    crack_count_train = int((y_train == crack_class_idx).sum())
    crack_count_test = int((y_test == crack_class_idx).sum())

    # Clone the model for each fold
    fold_model = clone(model)

    # Set crack_class_idx for PLS-DA models (needed for CRACK vs REST binarization)
    if hasattr(fold_model, 'named_steps'):
        # Pipeline: find PLSDAClassifier step
        for step_name, step in fold_model.named_steps.items():
            if isinstance(step, PLSDAClassifier):
                step.crack_class_idx = crack_class_idx
    elif isinstance(fold_model, PLSDAClassifier):
        fold_model.crack_class_idx = crack_class_idx

    # Training
    train_start = time_module.time()
    fold_model.fit(X_train, y_train)
    train_time = time_module.time() - train_start

    # Inference
    infer_start = time_module.time()
    y_pred = fold_model.predict(X_test)

    # Get probabilities if available - SAFELY handle column ordering
    has_proba = hasattr(fold_model, 'predict_proba')
    y_prob_crack = None
    crack_in_model_classes = False

    if has_proba:
        try:
            y_prob_all_classes = fold_model.predict_proba(X_test)

            # Get the model's classes (may be different order or missing classes)
            if hasattr(fold_model, 'classes_'):
                model_classes = fold_model.classes_
            elif hasattr(fold_model, 'named_steps'):
                # Pipeline: find the classifier step
                model_classes = None
                for step_name, step in fold_model.named_steps.items():
                    if hasattr(step, 'classes_'):
                        model_classes = step.classes_
                        break
            else:
                model_classes = None

            if model_classes is not None:
                # Find CRACK class column by matching to crack_class_idx
                model_classes_list = list(model_classes)
                if crack_class_idx in model_classes_list:
                    crack_col_idx = model_classes_list.index(crack_class_idx)
                    y_prob_crack = y_prob_all_classes[:, crack_col_idx]
                    crack_in_model_classes = True
                else:
                    # For binary wrappers, CRACK is always the second column (index 1)
                    # Check if this is a binary wrapper with 2 classes
                    if y_prob_all_classes.shape[1] == 2:
                        # Assume column 1 is CRACK probability for binary wrappers
                        y_prob_crack = y_prob_all_classes[:, 1]
                        crack_in_model_classes = True
                    else:
                        y_prob_crack = np.zeros(len(y_test))
                        crack_in_model_classes = False
            else:
                # Fallback: assume standard ordering 0..K-1
                if y_prob_all_classes.shape[1] > crack_class_idx:
                    y_prob_crack = y_prob_all_classes[:, crack_class_idx]
                    crack_in_model_classes = True
                elif y_prob_all_classes.shape[1] == 2:
                    # Binary model fallback - column 1 is positive class
                    y_prob_crack = y_prob_all_classes[:, 1]
                    crack_in_model_classes = True
                else:
                    y_prob_crack = np.zeros(len(y_test))
                    crack_in_model_classes = False

        except Exception as e:
            has_proba = False
            y_prob_crack = np.zeros(len(y_test))
            crack_in_model_classes = False
    else:
        y_prob_crack = np.zeros(len(y_test))
        crack_in_model_classes = False

    infer_time = time_module.time() - infer_start

    # Standard metrics calculation (works for both multi-class and binary)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # CRACK-specific metrics
    # Get unique labels for per-class metrics
    labels_for_metrics = list(range(n_classes))
    prec_per_class = precision_score(y_test, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels_for_metrics, zero_division=0)

    if crack_class_idx < len(prec_per_class):
        crack_prec = prec_per_class[crack_class_idx]
        crack_rec = rec_per_class[crack_class_idx]
        crack_f1 = f1_per_class[crack_class_idx]
    else:
        crack_prec = crack_rec = crack_f1 = 0.0

    # CRACK ROC-AUC and PR-AUC (one-vs-rest)
    crack_roc_auc = np.nan
    crack_pr_auc = np.nan
    warn_msg = None

    if crack_count_test == 0:
        warn_msg = f"No CRACK samples in test (held out: {fold_info.grape_group_held_out})"
    elif not crack_in_model_classes:
        warn_msg = "CRACK class was not in model's training classes - AUC undefined"
    elif has_proba and y_prob_crack is not None and y_prob_crack.sum() > 0:
        try:
            y_test_binary = (y_test == crack_class_idx).astype(int)
            crack_roc_auc = roc_auc_score(y_test_binary, y_prob_crack)
            crack_pr_auc = average_precision_score(y_test_binary, y_prob_crack)
        except Exception as e:
            warn_msg = f"AUC calculation failed: {e}"

    fold_end_time = time_module.time()
    fold_duration = fold_end_time - fold_start_time

    return {
        'fold_idx': fold_info.fold_idx,
        'fold_info': fold_info,
        'acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'crack_prec': crack_prec,
        'crack_rec': crack_rec,
        'crack_f1': crack_f1,
        'crack_roc_auc': crack_roc_auc,
        'crack_pr_auc': crack_pr_auc,
        'train_time': train_time,
        'infer_time': infer_time,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob_crack': y_prob_crack,
        'crack_count_train': crack_count_train,
        'crack_count_test': crack_count_test,
        'warn_msg': warn_msg,
        'fold_duration': fold_duration,
        'fold_start_str': fold_start_str,
    }


# ==================== MULTI-CLASS EVALUATION ====================

def evaluate_multiclass_logo(
    model,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray, FoldInfo]],
    class_names: List[str],
    crack_class_idx: int,
    crack_class_name: str,
    supports_njobs: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Performs domain-aware Leave-One-Group-Out cross-validation with multi-class metrics.

    For models that don't support n_jobs internally, folds are evaluated in parallel
    using joblib.Parallel (if available and supports_njobs=False).

    Returns:
        summary_dict: Summary metrics for the model
        curves_dict: Contains y_true_all, y_pred_all, y_prob_all (for CRACK class)
    """
    n_classes = len(class_names)
    n_folds = len(folds)

    print(f"    [TIMING] Starting {n_folds} folds, supports_njobs={supports_njobs}")
    eval_start = time.time()

    # Decide whether to parallelize folds
    # Only parallelize if model doesn't have internal n_jobs AND joblib is available
    use_parallel_folds = (not supports_njobs) and (Parallel is not None) and (N_JOBS != 1)

    if use_parallel_folds:
        print(f"    [PARALLEL] Running folds in parallel (n_jobs={N_JOBS})")
        # Run folds in parallel
        fold_results = Parallel(n_jobs=N_JOBS, backend=CV_PARALLEL_BACKEND, verbose=0)(
            delayed(_evaluate_single_fold)(
                fold_data, model, X, y, n_classes, crack_class_idx, n_folds
            )
            for fold_data in folds
        )
    else:
        # Run folds sequentially (model uses internal parallelism)
        fold_results = []
        for fold_idx, fold_data in enumerate(folds):
            print(f"    [FOLD {fold_idx+1}/{n_folds}] Starting...", end=" ", flush=True)
            result = _evaluate_single_fold(
                fold_data, model, X, y, n_classes, crack_class_idx, n_folds
            )
            print(f"done in {result['fold_duration']:.1f}s (train={result['train_time']:.1f}s, CRACK_test={result['crack_count_test']})")
            if result['warn_msg']:
                print(f"        [WARN] {result['warn_msg']}")
            fold_results.append(result)

    eval_end = time.time()
    print(f"    [TIMING] All folds completed in {eval_end - eval_start:.1f}s")

    # Aggregate results
    metrics = {
        'accs': [r['acc'] for r in fold_results],
        'macro_f1s': [r['macro_f1'] for r in fold_results],
        'weighted_f1s': [r['weighted_f1'] for r in fold_results],
        'crack_precs': [r['crack_prec'] for r in fold_results],
        'crack_recs': [r['crack_rec'] for r in fold_results],
        'crack_f1s': [r['crack_f1'] for r in fold_results],
        'crack_roc_aucs': [r['crack_roc_auc'] for r in fold_results],
        'crack_pr_aucs': [r['crack_pr_auc'] for r in fold_results],
        'train_times': [r['train_time'] for r in fold_results],
        'infer_times': [r['infer_time'] for r in fold_results],
    }

    # Print any warnings from parallel execution
    if use_parallel_folds:
        for r in fold_results:
            if r['warn_msg']:
                print(f"    [WARN] Fold {r['fold_idx']}: {r['warn_msg']}")

    # Aggregate predictions (sort by fold_idx to ensure consistent ordering)
    fold_results_sorted = sorted(fold_results, key=lambda r: r['fold_idx'])
    y_true_list = [r['y_test'] for r in fold_results_sorted]
    y_pred_list = [r['y_pred'] for r in fold_results_sorted]
    y_prob_crack_list = [r['y_prob_crack'] for r in fold_results_sorted]

    summary_dict = {
        "Model Name": model_name,
        "Mean Accuracy": np.nanmean(metrics['accs']),
        "Std Accuracy": np.nanstd(metrics['accs']),
        "Mean Macro-F1": np.nanmean(metrics['macro_f1s']),
        "Mean Weighted-F1": np.nanmean(metrics['weighted_f1s']),
        # CRACK-focused metrics (prominent)
        "CRACK_Precision": np.nanmean(metrics['crack_precs']),
        "CRACK_Recall": np.nanmean(metrics['crack_recs']),
        "CRACK_F1": np.nanmean(metrics['crack_f1s']),
        "CRACK_ROC_AUC": np.nanmean(metrics['crack_roc_aucs']),
        "CRACK_PR_AUC": np.nanmean(metrics['crack_pr_aucs']),
        "Mean Train Time (s)": np.mean(metrics['train_times']),
        "Mean Infer Time (s)": np.mean(metrics['infer_times']),
        "Total Eval Time (s)": eval_end - eval_start,
    }

    curves_dict = {
        'y_true_all': np.concatenate(y_true_list),
        'y_pred_all': np.concatenate(y_pred_list),
        'y_prob_crack_all': np.concatenate(y_prob_crack_list),
    }

    return summary_dict, curves_dict


# ==================== SAVE ARTIFACTS ====================

def save_multiclass_artifacts(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_crack_all: np.ndarray,
    out_dir: Path,
    model_name: str,
    dataset_name: str,
    balance_mode: str,
    class_names: List[str],
    crack_class_idx: int,
    class_mapping: Dict[str, int],
    eval_mode: str = "full",
):
    """
    Save ROC curve, PR curve (CRACK one-vs-rest), confusion matrices, and predictions CSV.
    """
    model_slug = slugify_model_name(model_name)
    # Map eval_mode to folder names: full -> dataset_name (3class/multiclass), crack_vs_rest -> crackVsRest
    eval_folder = "crackVsRest" if eval_mode == "crack_vs_rest" else dataset_name
    save_dir = ensure_dir(out_dir / "results" / "Pixel_Level_Classification" / eval_folder / balance_mode / model_slug)

    # --- Save class mapping ---
    with open(save_dir / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"    [SAVED] class_mapping.json")

    # --- Save predictions CSV ---
    predictions_df = pd.DataFrame({
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_prob_crack': y_prob_crack_all,
    })
    predictions_df.to_csv(save_dir / "predictions.csv", index=False)
    print(f"    [SAVED] predictions.csv")

    # --- CRACK one-vs-rest ROC Curve ---
    y_true_binary = (y_true_all == crack_class_idx).astype(int)

    if y_true_binary.sum() > 0 and y_prob_crack_all.sum() > 0:  # Has CRACK samples and probabilities
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_crack_all)
            roc_auc_val = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'CRACK One-vs-Rest ROC - {model_name}\n({dataset_name}, {balance_mode})')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(save_dir / "roc_curve_crack.png", dpi=150)
            plt.close(fig)
            print(f"    [SAVED] roc_curve_crack.png")

            # --- CRACK one-vs-rest Precision-Recall Curve ---
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_crack_all)
            ap_score = average_precision_score(y_true_binary, y_prob_crack_all)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'CRACK One-vs-Rest PR Curve - {model_name}\n({dataset_name}, {balance_mode})')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(save_dir / "pr_curve_crack.png", dpi=150)
            plt.close(fig)
            print(f"    [SAVED] pr_curve_crack.png")
        except Exception as e:
            print(f"    [WARN] Could not save ROC/PR curves: {e}")
    else:
        print(f"    [SKIP] No CRACK samples or no probabilities - skipping ROC/PR curves")

    # --- Confusion Matrix (Normalized) ---
    try:
        cm_norm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True)
        ax.set_title(f'Confusion Matrix (Normalized) - {model_name}\n({dataset_name}, {balance_mode})')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(f"    [SAVED] confusion_matrix.png")

        # --- Confusion Matrix (Raw) ---
        cm_raw = confusion_matrix(y_true_all, y_pred_all, normalize=None)
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
        disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=class_names)
        disp_raw.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
        ax.set_title(f'Confusion Matrix (Raw) - {model_name}\n({dataset_name}, {balance_mode})')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(save_dir / "confusion_matrix_raw.png", dpi=150)
        plt.close(fig)
        print(f"    [SAVED] confusion_matrix_raw.png")
    except Exception as e:
        print(f"    [WARN] Could not save confusion matrix: {e}")

    # --- Classification Report ---
    try:
        report = classification_report(y_true_all, y_pred_all, target_names=class_names, zero_division=0)
        with open(save_dir / "classification_report.txt", 'w') as f:
            f.write(report)
        print(f"    [SAVED] classification_report.txt")
    except Exception as e:
        print(f"    [WARN] Could not save classification report: {e}")


def save_model(model, out_dir: Path, model_name: str, dataset_name: str, balance_mode: str, eval_mode: str = "full"):
    """Save model to disk."""
    if joblib is None:
        return None
    # Map eval_mode to folder names: full -> dataset_name (3class/multiclass), crack_vs_rest -> crackVsRest
    eval_folder = "crackVsRest" if eval_mode == "crack_vs_rest" else dataset_name
    models_dir = ensure_dir(out_dir / "models" / eval_folder)
    path = models_dir / f"{model_name.replace(' ', '_')}_{balance_mode}.pkl"
    try:
        joblib.dump(model, path)
        print(f"[MODEL SAVED] {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Failed to save {model_name}: {e}")
        return None


def save_split_manifest(manifest: Dict, out_dir: Path, dataset_name: str, balance_mode: str, eval_mode: str = "full"):
    """Save split manifest for reproducibility verification."""
    # Map eval_mode to folder names: full -> dataset_name (3class/multiclass), crack_vs_rest -> crackVsRest
    eval_folder = "crackVsRest" if eval_mode == "crack_vs_rest" else dataset_name
    manifest_dir = ensure_dir(out_dir / "manifests" / eval_folder)
    manifest_path = manifest_dir / f"split_manifest_{balance_mode}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] Split manifest: {manifest_path}")


# ==================== MAIN ====================

def main():
    """Main benchmark function."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Store all results for final Excel summary
    all_results = {}

    for config in DATASET_CONFIGS:
        print(f"\n{'='*80}")
        print(f"DATASET: {config.name.upper()}")
        print(f"{'='*80}")

        # Check if CSV exists
        if not config.csv_path.exists():
            print(f"[ERROR] CSV not found: {config.csv_path}")
            continue

        # Load data
        print(f"[INFO] Loading data from {config.csv_path}...")
        df = pd.read_csv(config.csv_path)
        print(f"[INFO] Loaded {len(df)} rows")

        # for is_balanced in [True, False]:  # Balanced first, then Unbalanced
        for is_balanced in [False,True]:  # Balanced first, then Unbalanced
            balance_mode = "Balanced" if is_balanced else "Unbalanced"

            print(f"\n{'-'*60}")
            print(f"--- Running Benchmark: {config.name} / {balance_mode} ---")
            print(f"{'-'*60}")

            # Preprocess data using multi-class function
            try:
                data = preprocess_multiclass_dataset(
                    df,
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
                continue

            X, y = data.X, data.y
            class_names = data.class_names
            class_mapping = data.class_mapping
            n_classes = len(class_names)

            print(f"[INFO] Data preprocessed: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
            print(f"[INFO] Classes: {class_names}")
            print(f"[INFO] Class mapping: {class_mapping}")

            # Determine CRACK class index from the ACTUAL class_mapping (single source of truth)
            # This is the encoded label value used in y after preprocessing
            if config.is_3class:
                # For 3-class: crack_identifier is the original label (e.g., 2)
                # But after LabelEncoder, it may be remapped to 0..K-1
                # Look up in class_mapping which maps original -> encoded
                crack_original_label = str(config.crack_identifier)
                if crack_original_label in class_mapping:
                    crack_class_idx = class_mapping[crack_original_label]
                    crack_class_name = crack_original_label
                else:
                    # Fallback: search for a class containing "2" or check if it's already an int key
                    found = False
                    for name, idx in class_mapping.items():
                        if name == str(config.crack_identifier) or name == config.crack_identifier:
                            crack_class_idx = idx
                            crack_class_name = name
                            found = True
                            break
                    if not found:
                        print(f"[WARN] CRACK class '{config.crack_identifier}' not found in class_mapping: {class_mapping}")
                        # Use the crack_identifier directly if it's valid as encoded index
                        if config.crack_identifier in np.unique(y):
                            crack_class_idx = config.crack_identifier
                            crack_class_name = str(config.crack_identifier)
                        else:
                            print(f"[ERROR] Cannot determine CRACK class index!")
                            crack_class_idx = 0
                            crack_class_name = class_names[0]
            else:
                # For multiclass: crack_identifier is string ("CRACK")
                crack_class_name = str(config.crack_identifier).upper()
                if crack_class_name in class_mapping:
                    crack_class_idx = class_mapping[crack_class_name]
                else:
                    print(f"[WARN] CRACK class '{crack_class_name}' not found in class_mapping: {class_mapping}")
                    crack_class_idx = 0
                    crack_class_name = class_names[0]

            # Verify CRACK class exists in y
            crack_count_total = (y == crack_class_idx).sum()
            print(f"[INFO] CRACK class: '{crack_class_name}' (encoded index {crack_class_idx}, {crack_count_total} samples)")

            # Create domain-aware CV splits
            # Use data.grape_class_indices (encoded indices) instead of config.grape_classes (original labels)
            print(f"[INFO] Using grape_class_indices for split: {data.grape_class_indices}")
            try:
                folds, split_manifest = create_domain_aware_cv_splits(
                    y=y,
                    original_labels=data.original_labels,
                    segment_ids=data.segment_ids,
                    image_ids=data.image_ids,
                    grape_classes=data.grape_class_indices,  # Use encoded indices from preprocessing
                    random_state=RANDOM_STATE,
                    non_grape_holdout_frac=NON_GRAPE_HOLDOUT_FRACTION,
                )
            except Exception as e:
                print(f"[ERROR] CV split creation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Get models
            models = get_models(n_classes)

            # ==================== EVALUATION MODES LOOP ====================
            for eval_mode in ["full", "crack_vs_rest"]:
                if not EVALUATION_MODES.get(eval_mode, False):
                    print(f"\n[SKIP] Evaluation mode '{eval_mode}' is disabled")
                    continue

                print(f"\n{'='*60}")
                print(f"EVALUATION MODE: {eval_mode.upper()}")
                print(f"{'='*60}")

                # Save split manifest for this evaluation mode
                save_split_manifest(split_manifest, EXPERIMENT_DIR, config.name, balance_mode, eval_mode)

                # Prepare data for this evaluation mode
                if eval_mode == "crack_vs_rest":
                    # Convert to binary: CRACK=1, REST=0
                    y_eval = (y == crack_class_idx).astype(int)
                    class_names_eval = ["REST", "CRACK"]
                    class_mapping_eval = {"REST": 0, "CRACK": 1}
                    n_classes_eval = 2
                    crack_class_idx_eval = 1  # CRACK is class 1 in binary
                    crack_class_name_eval = "CRACK"
                else:
                    # Full multi-class evaluation
                    y_eval = y
                    class_names_eval = class_names
                    class_mapping_eval = class_mapping
                    n_classes_eval = n_classes
                    crack_class_idx_eval = crack_class_idx
                    crack_class_name_eval = crack_class_name

                results_table = []
                result_key = f"{config.name}_{balance_mode.lower()}_{eval_mode}"

                for model_name, model, binary_only, supports_njobs in models:
                    # Check model compatibility
                    # In crack_vs_rest mode, all models can run (it's binary)
                    # In full mode, binary_only models are skipped for multi-class
                    if eval_mode == "full" and binary_only and n_classes_eval > 2:
                        print(f"[SKIP] {model_name} is binary-only, skipping for {n_classes_eval}-class problem in '{eval_mode}' mode")
                        continue

                    # Check if model is enabled in MODELS_TO_TRAIN
                    if model_name in MODELS_TO_TRAIN and not MODELS_TO_TRAIN[model_name]:
                        print(f"[SKIP] {model_name} is disabled in MODELS_TO_TRAIN")
                        continue

                    print(f"\n[INFO] Evaluating: {model_name} ({eval_mode} mode, n_jobs support: {supports_njobs})")
                    model_start_time = time.time()

                    try:
                        # For crack_vs_rest mode, we need to update folds to use binary y
                        if eval_mode == "crack_vs_rest":
                            # Create new folds with binary y (same indices, different y values)
                            folds_eval = folds  # Same fold indices work for binary y
                        else:
                            folds_eval = folds

                        metrics, curves_dict = evaluate_multiclass_logo(
                            model=model,
                            model_name=model_name,
                            X=X,
                            y=y_eval,
                            folds=folds_eval,
                            class_names=class_names_eval,
                            crack_class_idx=crack_class_idx_eval,
                            crack_class_name=crack_class_name_eval,
                            supports_njobs=supports_njobs,
                        )

                        results_table.append(metrics)

                        model_total_time = time.time() - model_start_time

                        print(f"[SUCCESS] {model_name} ({eval_mode}) (total time: {model_total_time:.1f}s):")
                        print(f"    Accuracy: {metrics['Mean Accuracy']:.4f}")
                        print(f"    Macro-F1: {metrics['Mean Macro-F1']:.4f}")
                        print(f"    CRACK F1: {metrics['CRACK_F1']:.4f}")
                        print(f"    CRACK Precision: {metrics['CRACK_Precision']:.4f}")
                        print(f"    CRACK Recall: {metrics['CRACK_Recall']:.4f}")
                        print(f"    CRACK ROC-AUC: {metrics['CRACK_ROC_AUC']:.4f}")

                        # Save evaluation artifacts
                        if curves_dict is not None:
                            save_multiclass_artifacts(
                                y_true_all=curves_dict['y_true_all'],
                                y_pred_all=curves_dict['y_pred_all'],
                                y_prob_crack_all=curves_dict['y_prob_crack_all'],
                                out_dir=EXPERIMENT_DIR,
                                model_name=model_name,
                                dataset_name=config.name,
                                balance_mode=balance_mode,
                                class_names=class_names_eval,
                                crack_class_idx=crack_class_idx_eval,
                                class_mapping=class_mapping_eval,
                                eval_mode=eval_mode,
                            )

                        # Clone and train on full data for saving
                        final_model = clone(model)
                        # Set crack_class_idx for PLS-DA models
                        if hasattr(final_model, 'named_steps'):
                            for step_name, step in final_model.named_steps.items():
                                if isinstance(step, PLSDAClassifier):
                                    step.crack_class_idx = crack_class_idx_eval
                        elif isinstance(final_model, PLSDAClassifier):
                            final_model.crack_class_idx = crack_class_idx_eval
                        final_model.fit(X, y_eval)
                        save_model(final_model, EXPERIMENT_DIR, model_name, config.name, balance_mode, eval_mode)

                    except Exception as e:
                        print(f"[ERROR] {model_name} ({eval_mode}) failed: {e}")
                        import traceback
                        traceback.print_exc()
                        results_table.append({"Model Name": model_name, "Mean Accuracy": "FAILED"})

                # Create results DataFrame for this evaluation mode
                if results_table:
                    df_results = pd.DataFrame(results_table)
                    for col in df_results.select_dtypes(include=np.number).columns:
                        df_results[col] = df_results[col].round(4)
                    all_results[result_key] = df_results

                    # Print results table
                    print(f"\n--- {config.name} / {balance_mode} / {eval_mode} Results ---")
                    print(df_results.to_string())

    # Save unified Excel summary
    excel_path = EXPERIMENT_DIR / "model_comparison_results.xlsx"
    print(f"\n[INFO] Saving results to: {excel_path}")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for result_key, df_results in all_results.items():
            sheet_name = result_key[:31]  # Excel sheet name limit
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n[INFO] Benchmark complete!")
    print(f"[INFO] Results saved to: {excel_path}")
    print(f"[INFO] Artifacts saved under: {EXPERIMENT_DIR / 'results'}")


if __name__ == "__main__":
    main()
