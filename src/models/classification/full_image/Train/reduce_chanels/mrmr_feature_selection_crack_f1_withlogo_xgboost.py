"""
mRMR (Minimum Redundancy Maximum Relevance) Feature Selection for Crack Detection
==================================================================================

This script implements mRMR feature selection with XGBoost to optimize feature (wavelength)
selection for multi-class classification of grape hyperspectral images, with the primary
objective of maximizing F1-score for the CRACK class.

Author: ML Research Team
Date: 2025-12-11
Purpose: Thesis research - Optimal wavelength selection for crack detection using mRMR

Key Features:
- Multi-class classification with focus on CRACK class F1 optimization
- mRMR (Minimum Redundancy Maximum Relevance) feature ranking
- Handles class imbalance via sample weights in XGBoost (no oversampling)
- NO TEST LEAKAGE: mRMR uses only train_sub, test used once at end
- Comprehensive logging and visualization for research thesis
- High-quality plots (300 DPI) for publication
- Systematic evaluation across different feature subset sizes

mRMR Algorithm:
- Relevance: Mutual information with target labels (maximize)
- Redundancy: Average correlation with already selected features (minimize)
- Greedy forward selection balancing both criteria

Dependencies:
- numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn, joblib, tqdm
"""
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv")
RESULTS_FOLDER = str(_PROJECT_ROOT / r"results/mrmr_feature_selection_crack_f1")

# Wavelengths mapping (from wavelengths.py)
WAVELENGTHS = {
    1: 397.32, 2: 400.20, 3: 403.09, 4: 405.97, 5: 408.85,
    6: 411.74, 7: 414.63, 8: 417.52, 9: 420.40, 10: 423.29,
    11: 426.19, 12: 429.08, 13: 431.97, 14: 434.87, 15: 437.76,
    16: 440.66, 17: 443.56, 18: 446.45, 19: 449.35, 20: 452.25,
    21: 455.16, 22: 458.06, 23: 460.96, 24: 463.87, 25: 466.77,
    26: 469.68, 27: 472.59, 28: 475.50, 29: 478.41, 30: 481.32,
    31: 484.23, 32: 487.14, 33: 490.06, 34: 492.97, 35: 495.89,
    36: 498.80, 37: 501.72, 38: 504.64, 39: 507.56, 40: 510.48,
    41: 513.40, 42: 516.33, 43: 519.25, 44: 522.18, 45: 525.10,
    46: 528.03, 47: 530.96, 48: 533.89, 49: 536.82, 50: 539.75,
    51: 542.68, 52: 545.62, 53: 548.55, 54: 551.49, 55: 554.43,
    56: 557.36, 57: 560.30, 58: 563.24, 59: 566.18, 60: 569.12,
    61: 572.07, 62: 575.01, 63: 577.96, 64: 580.90, 65: 583.85,
    66: 586.80, 67: 589.75, 68: 592.70, 69: 595.65, 70: 598.60,
    71: 601.55, 72: 604.51, 73: 607.46, 74: 610.42, 75: 613.38,
    76: 616.34, 77: 619.30, 78: 622.26, 79: 625.22, 80: 628.18,
    81: 631.15, 82: 634.11, 83: 637.08, 84: 640.04, 85: 643.01,
    86: 645.98, 87: 648.95, 88: 651.92, 89: 654.89, 90: 657.87,
    91: 660.84, 92: 663.81, 93: 666.79, 94: 669.77, 95: 672.75,
    96: 675.73, 97: 678.71, 98: 681.69, 99: 684.67, 100: 687.65,
    101: 690.64, 102: 693.62, 103: 696.61, 104: 699.60, 105: 702.58,
    106: 705.57, 107: 708.57, 108: 711.56, 109: 714.55, 110: 717.54,
    111: 720.54, 112: 723.53, 113: 726.53, 114: 729.53, 115: 732.53,
    116: 735.53, 117: 738.53, 118: 741.53, 119: 744.53, 120: 747.54,
    121: 750.54, 122: 753.55, 123: 756.56, 124: 759.56, 125: 762.57,
    126: 765.58, 127: 768.60, 128: 771.61, 129: 774.62, 130: 777.64,
    131: 780.65, 132: 783.67, 133: 786.68, 134: 789.70, 135: 792.72,
    136: 795.74, 137: 798.77, 138: 801.79, 139: 804.81, 140: 807.84,
    141: 810.86, 142: 813.89, 143: 816.92, 144: 819.95, 145: 822.98,
    146: 826.01, 147: 829.04, 148: 832.07, 149: 835.11, 150: 838.14,
    151: 841.18, 152: 844.22, 153: 847.25, 154: 850.29, 155: 853.33,
    156: 856.37, 157: 859.42, 158: 862.46, 159: 865.50, 160: 868.55,
    161: 871.60, 162: 874.64, 163: 877.69, 164: 880.74, 165: 883.79,
    166: 886.84, 167: 889.90, 168: 892.95, 169: 896.01, 170: 899.06,
    171: 902.12, 172: 905.18, 173: 908.24, 174: 911.30, 175: 914.36,
    176: 917.42, 177: 920.48, 178: 923.55, 179: 926.61, 180: 929.68,
    181: 932.74, 182: 935.81, 183: 938.88, 184: 941.95, 185: 945.02,
    186: 948.10, 187: 951.17, 188: 954.24, 189: 957.32, 190: 960.40,
    191: 963.47, 192: 966.55, 193: 969.63, 194: 972.71, 195: 975.79,
    196: 978.88, 197: 981.96, 198: 985.05, 199: 988.13, 200: 991.22,
    201: 994.31, 202: 997.40, 203: 1000.49, 204: 1003.58,
}

# Columns to exclude from features
EXCLUDE_COLUMNS = ['label', 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']

# mRMR Configuration
MRMR_CONFIG = {
    'min_features': 5,
    'max_features': 50,
    'k_values_step': 1,  # Step size for k values to test
    'redundancy_weight': 1.0,  # Weight for redundancy penalty (0-1)
    'random_state': 42
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.20,
    'random_state': 42,
    'target_class': 'CRACK',
    'classifier': 'XGBoost'  # XGBoost Classifier with sample weights
}

# XGBoost Configuration
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1,  # Use all CPU cores
    'tree_method': 'hist'  # Fast histogram-based algorithm
}

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_results_folder(results_folder: str) -> None:
    """Create results folder structure."""
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_folder, 'plots')).mkdir(exist_ok=True)
    Path(os.path.join(results_folder, 'logs')).mkdir(exist_ok=True)
    Path(os.path.join(results_folder, 'models')).mkdir(exist_ok=True)
    print(f"✓ Results folder created: {results_folder}")


def load_and_prepare_data(data_path: str, exclude_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], LabelEncoder]:
    """
    Load dataset, remove outliers, and separate features from labels.

    Returns:
        df: Full DataFrame (after outlier removal) with all columns including hs_dir
        X: Feature DataFrame
        y: Label Series (encoded as integers)
        feature_names: List of feature column names
        label_encoder: LabelEncoder for converting between string and numeric labels
    """
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    df = pd.read_csv(data_path)
    print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Remove outliers if 'is_outlier' column exists
    samples_before = len(df)
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == 0].copy()
        samples_after = len(df)
        removed = samples_before - samples_after
        print(f"✓ Outlier removal: {removed} samples removed ({removed/samples_before*100:.2f}%)")
        print(f"✓ Remaining samples: {samples_after}")
    else:
        print(f"⚠️  Warning: 'is_outlier' column not found, skipping outlier removal")

    # Separate labels
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")

    # Store original string labels
    y_original = df['label']
    print(f"\n✓ Original label distribution:")
    label_counts = y_original.value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(y_original)*100:.2f}%)")

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_original)

    print(f"\n✓ Label encoding mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"    {label} -> {i}")

    # Update dataframe with encoded labels
    df['label'] = y_encoded
    y = pd.Series(y_encoded, index=y_original.index, name='label')

    # Extract features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]

    print(f"\n✓ Features extracted: {len(feature_cols)} wavelength channels")
    print(f"✓ Feature range: {X.min().min():.2f} to {X.max().max():.2f}")

    return df, X, y, feature_cols, label_encoder


def create_wavelength_mapping(feature_names: List[str]) -> Dict[str, float]:
    """
    Create mapping from feature names to wavelength values.
    Tries to match with WAVELENGTHS dict, fallback to column names.
    """
    wavelength_map = {}

    for fname in feature_names:
        # Try to extract band number from column name (e.g., "band_1" -> 1)
        try:
            if 'band_' in fname.lower():
                band_num = int(fname.lower().split('band_')[-1])
                wavelength_map[fname] = WAVELENGTHS.get(band_num, None)
            elif fname.replace('.', '').replace('nm', '').replace('_', '').isdigit():
                # Column name is already a wavelength
                wavelength_map[fname] = float(fname.replace('nm', '').replace('_', ''))
            else:
                wavelength_map[fname] = None
        except:
            wavelength_map[fname] = None

    # Fill missing with generic names
    for i, (fname, wl) in enumerate(wavelength_map.items()):
        if wl is None:
            wavelength_map[fname] = f"Feature_{i+1}"

    return wavelength_map


def prepare_logo_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int,
    label_encoder: LabelEncoder,
    group_col: str = "hs_dir"
) -> Tuple:
    """
    Prepare data for hybrid LOGO (Leave-One-Group-Out) strategy:
    - CRACK/REGULAR: Apply LOGO with filtered validation groups
      * Only groups containing BOTH CRACK and REGULAR can be validation folds
      * Groups with only one class are always added to training (never left out)
    - BACKGROUND/PLASTIC/BRANCH: Static 70/30 split (train_other/val_other) + test_other

    NO OVERSAMPLING - preserves natural class distribution.
    Class imbalance handled via sample weights during XGBoost training.

    This avoids data leakage while preserving biological structure.

    Args:
        df: Full DataFrame with features, labels (encoded as integers), and group column
        feature_cols: List of feature column names
        random_state: Random seed
        label_encoder: LabelEncoder for label conversion
        group_col: Column name for grouping (default: "hs_dir")

    Returns:
        logo_validation_groups: List of hs_dir groups that can serve as validation (contain both classes)
        always_train_groups: List of hs_dir groups that always stay in training (single class only)
        df_cr: DataFrame with CRACK/REGULAR samples only
        train_other: DataFrame with training samples from other classes
        val_other: DataFrame with validation samples from other classes
        test_other: DataFrame with test samples from other classes
        feature_cols: Feature column names
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "="*80)
    print("HYBRID LOGO DATA SPLITTING (FILTERED VALIDATION GROUPS)")
    print("="*80)
    print(f"✓ Grouping by: {group_col}")
    print(f"✓ Class imbalance handled via sample weights in XGBoost training")

    # Verify group column exists
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe")

    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Total unique {group_col}: {df[group_col].nunique()}")

    # Split into CRACK/REGULAR vs OTHER classes
    # Get encoded values for class names
    CR_CLASSES = ['CRACK', 'REGULAR']
    OTHER_CLASSES = ['BACKGROUND', 'PLASTIC', 'BRANCH']

    # Get encoded values
    crack_encoded = label_encoder.transform(['CRACK'])[0] if 'CRACK' in label_encoder.classes_ else None
    regular_encoded = label_encoder.transform(['REGULAR'])[0] if 'REGULAR' in label_encoder.classes_ else None
    cr_encoded = [crack_encoded, regular_encoded]
    cr_encoded = [x for x in cr_encoded if x is not None]
    other_encoded = [label_encoder.transform([c])[0] for c in OTHER_CLASSES if c in label_encoder.classes_]

    df_cr = df[df['label'].isin(cr_encoded)].copy()
    df_other = df[df['label'].isin(other_encoded)].copy()

    print(f"\n✓ Class split:")
    print(f"  CRACK/REGULAR samples: {len(df_cr)} ({len(df_cr)/len(df)*100:.2f}%)")
    print(f"  OTHER classes samples: {len(df_other)} ({len(df_other)/len(df)*100:.2f}%)")

    # Filter groups: only those with BOTH CRACK and REGULAR can be validation folds
    all_cr_groups = sorted(df_cr[group_col].unique())
    logo_validation_groups = []
    always_train_groups = []

    print(f"\n✓ Filtering CRACK/REGULAR groups for LOGO validation:")
    print(f"  Total unique {group_col}: {len(all_cr_groups)}")

    for group in all_cr_groups:
        group_data = df_cr[df_cr[group_col] == group]
        group_classes = set(group_data['label'].unique())

        # Check if group has both CRACK and REGULAR
        has_both = (crack_encoded in group_classes and regular_encoded in group_classes)

        if has_both:
            logo_validation_groups.append(group)
            crack_count = len(group_data[group_data['label'] == crack_encoded])
            regular_count = len(group_data[group_data['label'] == regular_encoded])
            print(f"    ✓ {group}: BOTH classes (CRACK={crack_count}, REGULAR={regular_count}) → Validation fold")
        else:
            always_train_groups.append(group)
            class_list = ', '.join([f"{label_encoder.inverse_transform([cls])[0]}={len(group_data[group_data['label'] == cls])}"
                                   for cls in group_classes])
            print(f"    ✗ {group}: Single class only ({class_list}) → Always in training")

    print(f"\n✓ LOGO strategy:")
    print(f"  Validation groups (can be left out): {len(logo_validation_groups)}")
    print(f"  Always-train groups (never left out): {len(always_train_groups)}")

    print(f"\n✓ Class distribution in CRACK/REGULAR:")
    for label_encoded, count in df_cr['label'].value_counts().items():
        label_name = label_encoder.inverse_transform([label_encoded])[0]
        print(f"    {label_name}: {count} ({count/len(df_cr)*100:.2f}%)")

    # Split OTHER classes: 70% train, 15% val, 15% test
    # Use stratified split to preserve class proportions
    if len(df_other) > 0:
        print(f"\n✓ Splitting OTHER classes (BACKGROUND/PLASTIC/BRANCH):")
        print(f"  Total OTHER samples: {len(df_other)}")

        # First split: 70% train, 30% temp (val + test)
        train_other, temp_other = train_test_split(
            df_other,
            test_size=0.30,
            random_state=random_state,
            stratify=df_other['label']
        )

        # Second split: 50/50 of temp -> val and test (15% each of total)
        val_other, test_other = train_test_split(
            temp_other,
            test_size=0.50,
            random_state=random_state + 1,
            stratify=temp_other['label']
        )

        print(f"  Train_other: {len(train_other)} ({len(train_other)/len(df_other)*100:.2f}%)")
        print(f"  Val_other: {len(val_other)} ({len(val_other)/len(df_other)*100:.2f}%)")
        print(f"  Test_other: {len(test_other)} ({len(test_other)/len(df_other)*100:.2f}%)")

        print(f"\n  Train_other class distribution:")
        for label_encoded, count in train_other['label'].value_counts().items():
            label_name = label_encoder.inverse_transform([label_encoded])[0]
            print(f"    {label_name}: {count} ({count/len(train_other)*100:.2f}%)")

        print(f"\n  Val_other class distribution:")
        for label_encoded, count in val_other['label'].value_counts().items():
            label_name = label_encoder.inverse_transform([label_encoded])[0]
            print(f"    {label_name}: {count} ({count/len(val_other)*100:.2f}%)")

        print(f"\n  Test_other class distribution:")
        for label_encoded, count in test_other['label'].value_counts().items():
            label_name = label_encoder.inverse_transform([label_encoded])[0]
            print(f"    {label_name}: {count} ({count/len(test_other)*100:.2f}%)")
    else:
        train_other = pd.DataFrame(columns=df.columns)
        val_other = pd.DataFrame(columns=df.columns)
        test_other = pd.DataFrame(columns=df.columns)
        print(f"\n⚠️  No OTHER classes found in dataset")

    print(f"\n✓ LOGO setup complete:")
    print(f"  - CRACK/REGULAR will use LOGO with {len(logo_validation_groups)} validation folds")
    print(f"  - {len(always_train_groups)} CRACK/REGULAR groups always stay in training")
    print(f"  - OTHER classes have static train/val/test split")
    print(f"  - Test set (test_other) will be used ONLY ONCE at the end")

    return logo_validation_groups, always_train_groups, df_cr, train_other, val_other, test_other, feature_cols


def create_logo_fold(
    fold_idx: int,
    val_group: str,
    logo_validation_groups: List[str],
    always_train_groups: List[str],
    df_cr: pd.DataFrame,
    train_other: pd.DataFrame,
    val_other: pd.DataFrame,
    feature_cols: List[str],
    random_state: int,
    group_col: str = "hs_dir"
) -> Tuple:
    """
    Create a single LOGO fold with filtered validation groups:
    - Training: ALL always_train_groups + all logo_validation_groups except current + train_other
    - Validation: current validation group + val_other

    This ensures groups with only one class (CRACK or REGULAR) are never wasted
    and always contribute to training.

    NO OVERSAMPLING - preserves natural class distribution.
    Class imbalance handled via sample weights during XGBoost training.

    Args:
        fold_idx: Fold number (for logging)
        val_group: Current hs_dir group for validation (from logo_validation_groups)
        logo_validation_groups: List of groups that can serve as validation
        always_train_groups: List of groups that always stay in training
        df_cr: Full CRACK/REGULAR DataFrame
        train_other: Training samples from other classes
        val_other: Validation samples from other classes
        feature_cols: Feature column names
        random_state: Random seed
        group_col: Group column name

    Returns:
        X_train, y_train, X_val, y_val (natural distribution, no oversampling)
    """
    # Split CRACK/REGULAR into train and val based on current validation group
    df_cr_val = df_cr[df_cr[group_col] == val_group].copy()

    # Training includes:
    # 1. All always_train_groups (never left out)
    # 2. All logo_validation_groups EXCEPT the current validation group
    train_groups = always_train_groups + [g for g in logo_validation_groups if g != val_group]
    df_cr_train = df_cr[df_cr[group_col].isin(train_groups)].copy()

    # Combine with OTHER classes
    df_train_fold = pd.concat([df_cr_train, train_other], axis=0).reset_index(drop=True)
    df_val_fold = pd.concat([df_cr_val, val_other], axis=0).reset_index(drop=True)

    # Extract features and labels (no oversampling)
    X_train = df_train_fold[feature_cols]
    y_train = df_train_fold['label']
    X_val = df_val_fold[feature_cols]
    y_val = df_val_fold['label']

    return X_train, y_train, X_val, y_val




# ============================================================================
# mRMR FEATURE SELECTION FUNCTIONS
# ============================================================================

def compute_mrmr_ranking(X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                        feature_names: List[str],
                        redundancy_weight: float = 1.0) -> Tuple[List[int], np.ndarray]:
    """
    Compute mRMR (Minimum Redundancy Maximum Relevance) feature ranking.

    Algorithm:
    1. Compute relevance: Mutual Information between each feature and target
       - Uses up to 30K samples per class for MI to speed up computation
    2. Greedy forward selection:
       - Start with feature having highest MI
       - For remaining features, compute score = MI - redundancy_penalty
       - Redundancy = average absolute correlation with already selected features
       - Select feature with highest score
       - Repeat until all features ranked

    Args:
        X_train_sub: Training subset features
        y_train_sub: Training subset labels
        feature_names: List of feature names
        redundancy_weight: Weight for redundancy penalty (0-1)

    Returns:
        ranked_indices: Indices of features in ranked order (best to worst)
        mrmr_scores: mRMR scores for each feature in ranked order
    """
    print("\n" + "="*80)
    print("COMPUTING mRMR FEATURE RANKING")
    print("="*80)
    print(f"⚠️  mRMR COMPUTED ON TRAIN_SUB ONLY – VALIDATION & TEST WITHHELD")
    print("="*80)

    n_features = X_train_sub.shape[1]
    rng = np.random.RandomState(MRMR_CONFIG['random_state'])

    # =========================
    # Step 1: Sample for MI (30K per class to speed up)
    # =========================
    print("\n✓ Computing relevance (Mutual Information with labels)...")

    max_per_class = 150000

    # Combine X and y for per-class sampling
    df_mi = X_train_sub.copy()
    df_mi["__label__"] = y_train_sub.values

    sampled_parts = []
    for cls, group in df_mi.groupby("__label__"):
        n_cls = len(group)
        n_sample = min(n_cls, max_per_class)
        if n_sample < n_cls:
            print(f"  - Class {cls}: sampling {n_sample} / {n_cls} pixels for MI")
            sampled = group.sample(
                n=n_sample,
                random_state=MRMR_CONFIG["random_state"],
                replace=False,
            )
        else:
            print(f"  - Class {cls}: using all {n_cls} pixels for MI")
            sampled = group
        sampled_parts.append(sampled)

    df_mi_sampled = pd.concat(sampled_parts, axis=0).reset_index(drop=True)
    X_mi = df_mi_sampled[feature_names]
    y_mi = df_mi_sampled["__label__"]

    print(f"  → Total samples used for MI: {len(X_mi)}")

    # Compute MI on the sampled balanced subset
    mi_scores = mutual_info_classif(
        X_mi,
        y_mi,
        random_state=MRMR_CONFIG["random_state"],
    )

    # Create relevance dictionary
    relevance = dict(zip(feature_names, mi_scores))

    print(f"  MI range: {mi_scores.min():.4f} to {mi_scores.max():.4f}")
    print(f"  Mean MI: {mi_scores.mean():.4f}")

    # Step 2: Compute feature correlation matrix (for redundancy) - use full data
    print("\n✓ Computing feature correlation matrix (for redundancy)...")
    corr_matrix = np.abs(X_train_sub.corr().values)
    np.fill_diagonal(corr_matrix, 0)  # Remove self-correlation

    print(f"  Max correlation: {corr_matrix.max():.4f}")
    print(f"  Mean correlation: {corr_matrix.mean():.4f}")

    # Step 3: Greedy mRMR feature selection
    print("\n✓ Running greedy mRMR selection...")
    selected_indices = []
    remaining_indices = list(range(n_features))
    mrmr_scores = []

    # Select first feature (highest MI)
    first_idx = np.argmax(mi_scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    mrmr_scores.append(mi_scores[first_idx])

    # Greedy selection for remaining features
    for _ in tqdm(range(n_features - 1), desc="mRMR Ranking"):
        best_score = -np.inf
        best_idx = None

        for idx in remaining_indices:
            # Relevance (MI with target)
            relevance_score = mi_scores[idx]

            # Redundancy (average correlation with already selected features)
            if len(selected_indices) > 0:
                redundancy = np.mean([corr_matrix[idx, sel_idx] for sel_idx in selected_indices])
            else:
                redundancy = 0.0

            # mRMR score = Relevance - redundancy_weight * Redundancy
            score = relevance_score - redundancy_weight * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        mrmr_scores.append(best_score)

    mrmr_scores = np.array(mrmr_scores)

    print("\n" + "="*80)
    print("mRMR RANKING COMPLETE")
    print("="*80)
    print(f"✓ Ranked {n_features} features")
    print(f"✓ Top 5 feature indices: {selected_indices[:5]}")
    print(f"✓ Top 5 mRMR scores: {mrmr_scores[:5]}")

    return selected_indices, mrmr_scores


def evaluate_feature_subset(k: int, ranked_indices: List[int],
                            X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            target_class: str, label_encoder: LabelEncoder) -> Dict:
    """
    Evaluate a feature subset of size k using the top-k features from mRMR ranking.

    Uses sample weights to handle class imbalance (no oversampling).

    Args:
        k: Number of top features to select
        ranked_indices: mRMR-ranked feature indices
        X_train_sub: Training subset features
        y_train_sub: Training subset labels (encoded as integers)
        X_val: Validation features
        y_val: Validation labels (encoded as integers)
        target_class: Target class name (e.g., 'CRACK')
        label_encoder: LabelEncoder for label conversion

    Returns:
        Dictionary with evaluation metrics
    """
    # Select top-k features
    selected_indices = ranked_indices[:k]

    # Subset data
    X_train_sub_selected = X_train_sub.iloc[:, selected_indices]
    X_val_selected = X_val.iloc[:, selected_indices]

    # Compute sample weights to handle class imbalance
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_sub)

    # Train XGBoost with sample weights
    xgb_model = XGBClassifier(**XGBOOST_CONFIG)
    xgb_model.fit(X_train_sub_selected, y_train_sub, sample_weight=sample_weight, verbose=False)

    # Predict on validation
    y_pred_val = xgb_model.predict(X_val_selected)

    # Compute metrics
    accuracy_val = accuracy_score(y_val, y_pred_val)
    f1_weighted_val = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

    # Target class metrics - convert target_class to encoded value
    target_encoded = label_encoder.transform([target_class])[0] if target_class in label_encoder.classes_ else None
    labels = sorted(y_val.unique())

    if target_encoded is not None and target_encoded in labels:
        target_idx = labels.index(target_encoded)
        f1_crack_val = f1_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
        precision_crack_val = precision_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
        recall_crack_val = recall_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
        f2_crack_val = fbeta_score(y_val, y_pred_val, beta=2, labels=labels, average=None, zero_division=0)[target_idx]
    else:
        f1_crack_val = precision_crack_val = recall_crack_val = f2_crack_val = 0.0

    return {
        'k': k,
        'num_features': k,
        'f1_crack_val': f1_crack_val,
        'precision_crack_val': precision_crack_val,
        'recall_crack_val': recall_crack_val,
        'f2_crack_val': f2_crack_val,
        'accuracy_val': accuracy_val,
        'f1_weighted_val': f1_weighted_val
    }


def run_mrmr_feature_selection_logo(
    logo_validation_groups: List[str],
    always_train_groups: List[str],
    df_cr: pd.DataFrame,
    train_other: pd.DataFrame,
    val_other: pd.DataFrame,
    feature_names: List[str],
    target_class: str,
    label_encoder: LabelEncoder,
    mrmr_config: Dict,
    random_state: int,
    results_folder: str,
    group_col: str = "hs_dir"
) -> Tuple[pd.DataFrame, int, List[int], np.ndarray, List[Dict]]:
    """
    Run mRMR feature selection using LOGO (Leave-One-Group-Out) cross-validation
    with filtered validation groups.

    For each logo_validation_group (groups containing both CRACK and REGULAR):
    1. Create train/val split (LOGO)
       - Train: all other logo_validation_groups + ALL always_train_groups + train_other
       - Val: current logo_validation_group + val_other
    2. Compute mRMR ranking on train_sub (70% of train for speed)
    3. Evaluate k features on validation
    4. Record F1_CRACK for this fold

    Aggregate results across all folds to find best k.

    IMPORTANT: NO TEST LEAKAGE
    - mRMR ranking computed on train portion only (per fold)
    - Validation uses the held-out group + val_other
    - Test set (test_other) never touched until final evaluation
    - always_train_groups are ALWAYS in training (never wasted)

    Returns:
        results_df: DataFrame with aggregated results for each k (mean across folds)
        best_k: Best number of features (based on mean F1_CRACK across folds)
        ranked_indices: mRMR-ranked feature indices (from fold 1, for reference)
        mrmr_scores: mRMR scores (from fold 1, for reference)
        fold_results: List of per-fold results dictionaries
    """
    print("\n" + "="*80)
    print("LOGO (LEAVE-ONE-GROUP-OUT) mRMR FEATURE SELECTION")
    print("="*80)
    print(f"✓ Total LOGO validation folds: {len(logo_validation_groups)}")
    print(f"✓ Always-train groups (never left out): {len(always_train_groups)}")
    print(f"⚠️  mRMR COMPUTED PER FOLD ON TRAIN ONLY – VALIDATION & TEST WITHHELD")
    print("="*80)

    min_features = mrmr_config['min_features']
    max_features = min(mrmr_config['max_features'], len(feature_names))
    step_size = mrmr_config['k_values_step']
    k_values = list(range(min_features, max_features + 1, step_size))

    print(f"\n✓ Testing {len(k_values)} different feature subset sizes...")
    print(f"  Range: {min_features} to {max_features} features")

    # Store results for each fold
    fold_results = []
    all_fold_k_results = []  # List of DataFrames, one per fold

    # Store first fold's mRMR ranking for reference/plotting
    reference_ranked_indices = None
    reference_mrmr_scores = None

    # Iterate through LOGO folds (only validation groups that contain both classes)
    for fold_idx, val_group in enumerate(tqdm(logo_validation_groups, desc="LOGO Folds")):
        print(f"\n" + "-"*80)
        print(f"FOLD {fold_idx + 1}/{len(logo_validation_groups)}: Validation group = {val_group}")
        print("-"*80)

        # Create fold data
        X_train, y_train, X_val, y_val = create_logo_fold(
            fold_idx, val_group, logo_validation_groups, always_train_groups,
            df_cr, train_other, val_other,
            feature_names, random_state, group_col
        )

        print(f"  Train samples: {len(X_train)} (natural distribution, no oversampling)")
        print(f"  Val samples: {len(X_val)}")

        # For speed, use 70% of train for mRMR computation
        # (mRMR is expensive, especially MI calculation)
        from sklearn.model_selection import train_test_split
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train, y_train,
            train_size=0.7,
            random_state=random_state,
            stratify=y_train
        )

        print(f"  Train_sub for mRMR: {len(X_train_sub)} samples")

        # Compute mRMR ranking for this fold
        try:
            ranked_indices, mrmr_scores = compute_mrmr_ranking(
                X_train_sub, y_train_sub, feature_names,
                redundancy_weight=mrmr_config['redundancy_weight']
            )

            # Store first fold's ranking as reference
            if fold_idx == 0:
                reference_ranked_indices = ranked_indices
                reference_mrmr_scores = mrmr_scores

        except Exception as e:
            print(f"\n⚠️  ERROR in fold {fold_idx + 1}: mRMR ranking failed: {e}")
            print(f"  Skipping this fold...")
            continue

        # Evaluate different k values on this fold
        fold_k_results = []

        for k in k_values:
            try:
                result = evaluate_feature_subset(
                    k, ranked_indices,
                    X_train_sub, y_train_sub,  # Use train_sub for consistency
                    X_val, y_val,
                    target_class, label_encoder
                )
                result['fold'] = fold_idx + 1
                result['val_group'] = val_group
                fold_k_results.append(result)

            except Exception as e:
                print(f"\n⚠️  Warning: Fold {fold_idx + 1}, k={k} failed: {e}")
                result = {
                    'fold': fold_idx + 1,
                    'val_group': val_group,
                    'k': k,
                    'num_features': k,
                    'f1_crack_val': 0.0,
                    'precision_crack_val': 0.0,
                    'recall_crack_val': 0.0,
                    'f2_crack_val': 0.0,
                    'accuracy_val': 0.0,
                    'f1_weighted_val': 0.0
                }
                fold_k_results.append(result)

        # Store fold results
        fold_df = pd.DataFrame(fold_k_results)
        all_fold_k_results.append(fold_df)

        # Get best k for this fold
        best_idx_fold = fold_df['f1_crack_val'].idxmax()
        best_k_fold = int(fold_df.loc[best_idx_fold, 'k'])
        best_f1_fold = fold_df.loc[best_idx_fold, 'f1_crack_val']

        print(f"\n  ✓ Fold {fold_idx + 1} best k: {best_k_fold} (F1_CRACK = {best_f1_fold:.4f})")

        fold_results.append({
            'fold': fold_idx + 1,
            'val_group': val_group,
            'best_k': best_k_fold,
            'best_f1_crack': best_f1_fold,
            'n_train': len(X_train),
            'n_val': len(X_val)
        })

    # Aggregate results across folds
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS LOGO FOLDS")
    print("="*80)

    # Combine all fold results
    all_folds_df = pd.concat(all_fold_k_results, axis=0, ignore_index=True)

    # Save per-fold results
    all_folds_df.to_csv(
        os.path.join(results_folder, 'logs', 'mrmr_logo_per_fold_results.csv'),
        index=False
    )
    print(f"✓ Per-fold results saved")

    # Compute mean and std across folds for each k
    results_by_k = all_folds_df.groupby('k').agg({
        'f1_crack_val': ['mean', 'std'],
        'precision_crack_val': ['mean', 'std'],
        'recall_crack_val': ['mean', 'std'],
        'f2_crack_val': ['mean', 'std'],
        'accuracy_val': ['mean', 'std'],
        'f1_weighted_val': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    results_by_k.columns = [
        'k',
        'f1_crack_val_mean', 'f1_crack_val_std',
        'precision_crack_val_mean', 'precision_crack_val_std',
        'recall_crack_val_mean', 'recall_crack_val_std',
        'f2_crack_val_mean', 'f2_crack_val_std',
        'accuracy_val_mean', 'accuracy_val_std',
        'f1_weighted_val_mean', 'f1_weighted_val_std'
    ]

    results_by_k['num_features'] = results_by_k['k']

    # Find best k based on mean F1_CRACK across folds
    best_idx = results_by_k['f1_crack_val_mean'].idxmax()
    best_k = int(results_by_k.loc[best_idx, 'k'])
    best_f1_mean = results_by_k.loc[best_idx, 'f1_crack_val_mean']
    best_f1_std = results_by_k.loc[best_idx, 'f1_crack_val_std']

    print(f"\n✓ Best k (across {len(logo_validation_groups)} validation folds): {best_k} features")
    print(f"✓ Mean F1_CRACK (validation): {best_f1_mean:.4f} ± {best_f1_std:.4f}")
    print(f"✓ Mean accuracy (validation): {results_by_k.loc[best_idx, 'accuracy_val_mean']:.4f}")

    # Save aggregated results
    results_by_k.to_csv(
        os.path.join(results_folder, 'logs', 'mrmr_logo_aggregated_results.csv'),
        index=False
    )
    print(f"✓ Aggregated results saved")

    # Save fold summary
    fold_summary_df = pd.DataFrame(fold_results)
    fold_summary_df.to_csv(
        os.path.join(results_folder, 'logs', 'mrmr_logo_fold_summary.csv'),
        index=False
    )
    print(f"✓ Fold summary saved")

    # Save reference mRMR ranking (from first fold)
    if reference_ranked_indices is not None:
        wavelength_map = create_wavelength_mapping(feature_names)
        ranking_df = pd.DataFrame({
            'rank': range(1, len(reference_ranked_indices) + 1),
            'feature_index': reference_ranked_indices,
            'feature_name': [feature_names[i] for i in reference_ranked_indices],
            'wavelength_nm': [wavelength_map.get(feature_names[i], 'N/A') for i in reference_ranked_indices],
            'mrmr_score': reference_mrmr_scores
        })
        ranking_df.to_csv(
            os.path.join(results_folder, 'logs', 'mrmr_reference_ranking.csv'),
            index=False
        )
        print(f"✓ Reference mRMR ranking saved (from fold 1)")

    return results_by_k, best_k, reference_ranked_indices, reference_mrmr_scores, fold_results


def run_mrmr_feature_selection(X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               feature_names: List[str],
                               target_class: str,
                               label_encoder: LabelEncoder,
                               mrmr_config: Dict,
                               results_folder: str) -> Tuple[pd.DataFrame, int, List[int], np.ndarray]:
    """
    Run mRMR feature selection and evaluate different k values.

    IMPORTANT: NO TEST LEAKAGE
    - mRMR ranking computed on train_sub only
    - Model evaluation on validation only
    - Test set never touched

    Returns:
        results_df: DataFrame with results for each k
        best_k: Best number of features (based on validation CRACK F1)
        ranked_indices: mRMR-ranked feature indices
        mrmr_scores: mRMR scores for each feature
    """
    # Step 1: Compute global mRMR ranking
    ranked_indices, mrmr_scores = compute_mrmr_ranking(
        X_train_sub, y_train_sub, feature_names,
        redundancy_weight=mrmr_config['redundancy_weight']
    )

    # Step 2: Evaluate different k values
    print("\n" + "="*80)
    print("EVALUATING FEATURE SUBSETS")
    print("="*80)
    print(f"⚠️  EVALUATION ON VALIDATION SET ONLY – TEST WITHHELD")
    print("="*80)

    min_features = mrmr_config['min_features']
    max_features = min(mrmr_config['max_features'], len(feature_names))
    step_size = mrmr_config['k_values_step']

    k_values = list(range(min_features, max_features + 1, step_size))

    print(f"\n✓ Testing {len(k_values)} different feature subset sizes...")
    print(f"  Range: {min_features} to {max_features} features\n")

    results = []

    for k in tqdm(k_values, desc="Evaluating k values"):
        try:
            result = evaluate_feature_subset(
                k, ranked_indices,
                X_train_sub, y_train_sub,
                X_val, y_val,
                target_class, label_encoder
            )
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Warning: Evaluation with k={k} failed: {e}")
            result = {
                'k': k,
                'num_features': k,
                'f1_crack_val': 0.0,
                'precision_crack_val': 0.0,
                'recall_crack_val': 0.0,
                'f2_crack_val': 0.0,
                'accuracy_val': 0.0,
                'f1_weighted_val': 0.0
            }
            results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best k
    best_idx = results_df['f1_crack_val'].idxmax()
    best_k = int(results_df.loc[best_idx, 'k'])
    best_f1_val = results_df.loc[best_idx, 'f1_crack_val']

    print("\n" + "="*80)
    print("mRMR FEATURE SELECTION COMPLETE")
    print("="*80)
    print(f"✓ Best k: {best_k} features")
    print(f"✓ Best F1_CRACK (validation): {best_f1_val:.4f}")
    print(f"✓ Best accuracy (validation): {results_df.loc[best_idx, 'accuracy_val']:.4f}")

    # Save results log
    results_df.to_csv(os.path.join(results_folder, 'logs', 'mrmr_results_by_k.csv'), index=False)
    print(f"✓ Results saved to: {os.path.join(results_folder, 'logs', 'mrmr_results_by_k.csv')}")

    # Save mRMR ranking
    wavelength_map = create_wavelength_mapping(feature_names)
    ranking_df = pd.DataFrame({
        'rank': range(1, len(ranked_indices) + 1),
        'feature_index': ranked_indices,
        'feature_name': [feature_names[i] for i in ranked_indices],
        'wavelength_nm': [wavelength_map.get(feature_names[i], 'N/A') for i in ranked_indices],
        'mrmr_score': mrmr_scores
    })
    ranking_df.to_csv(os.path.join(results_folder, 'logs', 'mrmr_full_ranking.csv'), index=False)
    print(f"✓ Full mRMR ranking saved to: {os.path.join(results_folder, 'logs', 'mrmr_full_ranking.csv')}")

    return results_df, best_k, ranked_indices, mrmr_scores


def evaluate_best_mrmr_model_logo(
    best_k: int,
    ranked_indices: List[int],
    df_cr: pd.DataFrame,
    train_other: pd.DataFrame,
    test_other: pd.DataFrame,
    feature_names: List[str],
    target_class: str,
    best_val_metrics: Dict,
    random_state: int,
    results_folder: str
) -> Tuple[Any, Dict]:
    """
    Train final model with best mRMR-selected features on ALL CRACK/REGULAR + train_other.
    Evaluate on test_other (USED ONLY ONCE HERE - NO LEAKAGE).

    NO OVERSAMPLING - uses sample weights to handle class imbalance.

    Final training set:
    - All CRACK/REGULAR samples (all hs_dir groups)
    - train_other (70% of BACKGROUND/PLASTIC/BRANCH)

    Final test set:
    - test_other (15% of BACKGROUND/PLASTIC/BRANCH)
    - This is completely independent of CRACK/REGULAR images
    - Zero leakage from LOGO cross-validation

    Args:
        best_k: Best number of features
        ranked_indices: mRMR-ranked feature indices
        df_cr: All CRACK/REGULAR samples
        train_other: Training portion of other classes
        test_other: Test portion of other classes
        feature_names: Feature names
        target_class: Target class (CRACK)
        best_val_metrics: Validation metrics from LOGO
        random_state: Random seed
        results_folder: Results folder path

    Returns:
        final_model: Trained LDA model
        results: Dictionary with test results
    """
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON TEST SET (USED ONLY ONCE)")
    print("="*80)

    # Get selected features
    selected_indices = ranked_indices[:best_k]
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"✓ Selected {len(selected_features)} features")

    # Combine all CRACK/REGULAR with train_other for final training
    df_train_final = pd.concat([df_cr, train_other], axis=0).reset_index(drop=True)

    print(f"\n✓ Final training set composition:")
    print(f"  CRACK/REGULAR samples: {len(df_cr)}")
    print(f"  OTHER (train) samples: {len(train_other)}")
    print(f"  Total training samples: {len(df_train_final)}")

    print(f"\n  Class distribution (natural, no oversampling):")
    for label, count in df_train_final['label'].value_counts().items():
        print(f"    {label}: {count} ({count/len(df_train_final)*100:.2f}%)")

    # Extract features and labels (no oversampling)
    X_train_final = df_train_final[feature_names]
    y_train_final = df_train_final['label']

    # Compute sample weights to handle class imbalance
    sample_weight_full = compute_sample_weight(class_weight="balanced", y=y_train_final)

    print(f"\n✓ Using class-balanced sample weights (no oversampling)")

    # Subset to selected features
    X_train_subset = X_train_final.iloc[:, selected_indices]

    # Prepare test set
    X_test = test_other[feature_names].iloc[:, selected_indices]
    y_test = test_other['label']

    print(f"\n✓ Test set (test_other only):")
    print(f"  Test samples: {len(test_other)}")
    print(f"  Class distribution:")
    for label, count in y_test.value_counts().items():
        print(f"    {label}: {count} ({count/len(y_test)*100:.2f}%)")

    print(f"\n⚠️  NOTE: Test set contains only OTHER classes (BACKGROUND/PLASTIC/BRANCH)")
    print(f"⚠️  NOTE: CRACK/REGULAR were used for LOGO cross-validation only")
    print(f"⚠️  NOTE: This ensures zero leakage between train and test")

    # Train final model with sample weights
    print(f"\n✓ Training final XGBoost model with class-balanced sample weights...")
    final_model = XGBClassifier(**XGBOOST_CONFIG)
    final_model.fit(X_train_subset, y_train_final, sample_weight=sample_weight_full, verbose=False)

    # Predictions on test set
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)

    # Metrics on test set
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=final_model.classes_)
    report = classification_report(y_test, y_pred, labels=final_model.classes_,
                                   output_dict=True, zero_division=0)

    print(f"\n✓ Final Model Performance on Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report (Test Set - OTHER classes only):")
    print(classification_report(y_test, y_pred, labels=final_model.classes_, zero_division=0))

    # CRACK-specific metrics on test set (if CRACK appears in test)
    crack_metrics = report.get(target_class, {})
    if target_class in final_model.classes_ and target_class in y_test.unique():
        print(f"\n✓ {target_class} Class Metrics (Test Set):")
        print(f"  Precision: {crack_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {crack_metrics.get('recall', 0.0):.4f}")
        print(f"  F1-score: {crack_metrics.get('f1-score', 0.0):.4f}")

        # Compute F2 score for CRACK
        labels_test = list(final_model.classes_)
        target_idx = labels_test.index(target_class)
        f2_crack_test = fbeta_score(y_test, y_pred, beta=2, labels=labels_test,
                                    average=None, zero_division=0)[target_idx]
        print(f"  F2-score: {f2_crack_test:.4f}")
    else:
        print(f"\n⚠️  {target_class} not present in test set (expected, as test_other contains only OTHER classes)")
        f2_crack_test = 0.0

    # Save model
    model_path = os.path.join(results_folder, 'models', 'mrmr_logo_selected_lda_model.pkl')
    joblib.dump({
        'model': final_model,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'best_k': best_k,
        'mrmr_validation_metrics': best_val_metrics,
        'test_metrics': {
            'accuracy': accuracy,
            'crack_precision': crack_metrics.get('precision', 0.0),
            'crack_recall': crack_metrics.get('recall', 0.0),
            'crack_f1': crack_metrics.get('f1-score', 0.0),
            'crack_f2': f2_crack_test
        },
        'classes': final_model.classes_.tolist()
    }, model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Save selected features list
    wavelength_map = create_wavelength_mapping(feature_names)
    selected_features_df = pd.DataFrame({
        'rank': range(1, best_k + 1),
        'feature_name': selected_features,
        'feature_index': selected_indices,
        'wavelength_nm': [wavelength_map.get(f, 'N/A') for f in selected_features]
    })

    # Add XGBoost feature importances if available
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        selected_features_df['xgb_importance'] = importances
        selected_features_df['importance_rank'] = selected_features_df['xgb_importance'].rank(ascending=False)

    selected_features_df.to_csv(
        os.path.join(results_folder, 'logs', 'mrmr_logo_selected_features.csv'),
        index=False
    )
    print(f"✓ Selected features saved to: {os.path.join(results_folder, 'logs', 'mrmr_logo_selected_features.csv')}")

    results = {
        'model': final_model,
        'selected_features': selected_features,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy
    }

    return final_model, results


def evaluate_best_mrmr_model(best_k: int, ranked_indices: List[int],
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             feature_names: List[str],
                             target_class: str,
                             best_val_metrics: Dict,
                             results_folder: str) -> Tuple[Any, Dict]:
    """
    Train final model with best mRMR-selected features on FULL training set.
    Evaluate on test set (USED ONLY ONCE HERE - NO LEAKAGE).

    NO OVERSAMPLING - uses sample weights to handle class imbalance.
    """
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON TEST SET (USED ONLY ONCE)")
    print("="*80)

    # Get selected features
    selected_indices = ranked_indices[:best_k]
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"✓ Selected {len(selected_features)} features")

    # Subset data using best mRMR features
    X_train_subset = X_train.iloc[:, selected_indices]
    X_test_subset = X_test.iloc[:, selected_indices]

    print(f"\n✓ Training final model on FULL training set with class-balanced sample weights:")
    print(f"  Training samples: {X_train_subset.shape[0]}")
    print(f"  Selected features: {X_train_subset.shape[1]}")

    # Compute sample weights to handle class imbalance
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # Train final model with sample weights
    final_model = XGBClassifier(**XGBOOST_CONFIG)
    final_model.fit(X_train_subset, y_train, sample_weight=sample_weight, verbose=False)

    print(f"\n✓ Evaluating on test set:")
    print(f"  Test samples: {X_test_subset.shape[0]}")

    # Predictions on test set
    y_pred = final_model.predict(X_test_subset)
    y_pred_proba = final_model.predict_proba(X_test_subset)

    # Metrics on test set
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n✓ Final Model Performance on Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # CRACK-specific metrics on test set
    crack_metrics = report.get(target_class, {})
    print(f"\n✓ {target_class} Class Metrics (Test Set):")
    print(f"  Precision: {crack_metrics.get('precision', 0.0):.4f}")
    print(f"  Recall: {crack_metrics.get('recall', 0.0):.4f}")
    print(f"  F1-score: {crack_metrics.get('f1-score', 0.0):.4f}")

    # Compute F2 score for CRACK
    labels_test = sorted(y_test.unique())
    f2_crack_test = 0.0
    if target_class in labels_test:
        target_idx = labels_test.index(target_class)
        f2_crack_test = fbeta_score(y_test, y_pred, beta=2, labels=labels_test, average=None, zero_division=0)[target_idx]
        print(f"  F2-score: {f2_crack_test:.4f}")

    # Save model
    model_path = os.path.join(results_folder, 'models', 'mrmr_selected_lda_model.pkl')
    joblib.dump({
        'model': final_model,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'best_k': best_k,
        'mrmr_validation_metrics': best_val_metrics,
        'test_metrics': {
            'accuracy': accuracy,
            'crack_precision': crack_metrics.get('precision', 0.0),
            'crack_recall': crack_metrics.get('recall', 0.0),
            'crack_f1': crack_metrics.get('f1-score', 0.0),
            'crack_f2': f2_crack_test
        },
        'classes': final_model.classes_
    }, model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Save selected features list
    wavelength_map = create_wavelength_mapping(feature_names)
    selected_features_df = pd.DataFrame({
        'rank': range(1, best_k + 1),
        'feature_name': selected_features,
        'feature_index': selected_indices,
        'wavelength_nm': [wavelength_map.get(f, 'N/A') for f in selected_features]
    })

    # Add XGBoost feature importances if available
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        selected_features_df['xgb_importance'] = importances
        selected_features_df['importance_rank'] = selected_features_df['xgb_importance'].rank(ascending=False)

    selected_features_df.to_csv(
        os.path.join(results_folder, 'logs', 'mrmr_selected_features.csv'),
        index=False
    )
    print(f"✓ Selected features saved to: {os.path.join(results_folder, 'logs', 'mrmr_selected_features.csv')}")

    results = {
        'model': final_model,
        'selected_features': selected_features,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy
    }

    return final_model, results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_mrmr_performance_vs_k(results_df: pd.DataFrame, best_k: int, results_folder: str, is_logo: bool = False) -> None:
    """Plot mRMR performance (CRACK F1 on validation) vs number of features."""
    print("\n✓ Generating mRMR performance vs k plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    if is_logo:
        # LOGO: plot mean with error bars
        k_values = results_df['k']
        f1_mean = results_df['f1_crack_val_mean']
        f1_std = results_df['f1_crack_val_std']

        ax.plot(k_values, f1_mean, marker='o', linewidth=2.5, markersize=6,
                label='F1_CRACK (mean across LOGO folds)', color='#d62728')
        ax.fill_between(k_values, f1_mean - f1_std, f1_mean + f1_std,
                        alpha=0.2, color='#d62728', label='± 1 std')

        # Mark best k
        best_f1 = results_df[results_df['k'] == best_k]['f1_crack_val_mean'].values[0]
        best_std = results_df[results_df['k'] == best_k]['f1_crack_val_std'].values[0]
        ax.scatter([best_k], [best_f1], s=500, c='gold', edgecolor='black',
                   linewidth=2.5, zorder=5,
                   label=f'Best k = {best_k} (F1={best_f1:.3f}±{best_std:.3f})', marker='*')
    else:
        # Single split: plot F1_CRACK vs k
        ax.plot(results_df['k'], results_df['f1_crack_val'],
                marker='o', linewidth=2.5, markersize=6, label='F1_CRACK (Validation)', color='#d62728')

        # Mark best k
        best_f1 = results_df[results_df['k'] == best_k]['f1_crack_val'].values[0]
        ax.scatter([best_k], [best_f1], s=500, c='gold', edgecolor='black',
                   linewidth=2.5, zorder=5, label=f'Best k = {best_k}', marker='*')

    # Add vertical line at best k
    ax.axvline(best_k, color='gold', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Number of Selected Features (k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score (CRACK Class on Validation)', fontsize=14, fontweight='bold')

    title = 'mRMR Feature Selection: Performance vs Number of Features'
    if is_logo:
        title += '\n(LOGO Cross-Validation)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'mrmr_performance_vs_k.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_mrmr_ranked_wavelengths(ranked_indices: List[int], mrmr_scores: np.ndarray,
                                 feature_names: List[str], results_folder: str,
                                 top_n: int = 30) -> None:
    """Bar plot showing top-N wavelengths from mRMR ranking."""
    print("\n✓ Generating mRMR ranked wavelengths plot...")

    # Take top N
    top_indices = ranked_indices[:top_n]
    top_scores = mrmr_scores[:top_n]

    # Get wavelengths
    wavelength_map = create_wavelength_mapping(feature_names)
    labels = []
    for idx in top_indices:
        fname = feature_names[idx]
        wl = wavelength_map.get(fname, 'N/A')
        if isinstance(wl, (int, float)):
            labels.append(f"{wl:.1f}nm")
        else:
            labels.append(str(fname)[:15])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(range(len(top_scores)), top_scores,
                   color='steelblue', edgecolor='black', linewidth=1.2)

    # Color gradient
    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_scores)):
        ax.text(val + val*0.02, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('mRMR Score (Relevance - Redundancy)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wavelength (Ranked)', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {len(top_scores)} Wavelengths by mRMR Ranking',
                 fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'mrmr_ranked_wavelengths.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_confusion_matrix_mrmr(cm: np.ndarray, classes: List[str], results_folder: str) -> None:
    """Plot confusion matrix heatmap for mRMR model on test set."""
    print("\n✓ Generating confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=1, linecolor='gray', ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix: mRMR-Selected XGBoost Model on Test Set',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'mrmr_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_mrmr_metrics_comparison(results_df: pd.DataFrame, best_k: int, results_folder: str, is_logo: bool = False) -> None:
    """Plot multiple metrics (F1, Precision, Recall) vs k."""
    print("\n✓ Generating mRMR metrics comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    if is_logo:
        # LOGO: plot means
        ax.plot(results_df['k'], results_df['f1_crack_val_mean'],
                marker='o', linewidth=2, markersize=5, label='F1-Score', color='#d62728')
        ax.plot(results_df['k'], results_df['precision_crack_val_mean'],
                marker='s', linewidth=2, markersize=5, label='Precision', color='#2ca02c', alpha=0.7)
        ax.plot(results_df['k'], results_df['recall_crack_val_mean'],
                marker='^', linewidth=2, markersize=5, label='Recall', color='#ff7f0e', alpha=0.7)
    else:
        # Single split: plot metrics
        ax.plot(results_df['k'], results_df['f1_crack_val'],
                marker='o', linewidth=2, markersize=5, label='F1-Score', color='#d62728')
        ax.plot(results_df['k'], results_df['precision_crack_val'],
                marker='s', linewidth=2, markersize=5, label='Precision', color='#2ca02c', alpha=0.7)
        ax.plot(results_df['k'], results_df['recall_crack_val'],
                marker='^', linewidth=2, markersize=5, label='Recall', color='#ff7f0e', alpha=0.7)

    # Mark best k
    ax.axvline(best_k, color='gold', linestyle='--', linewidth=2,
               label=f'Best k = {best_k}', alpha=0.7)

    ax.set_xlabel('Number of Selected Features (k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (CRACK Class on Validation)', fontsize=14, fontweight='bold')

    title = 'mRMR Feature Selection: Metrics Comparison'
    if is_logo:
        title += '\n(LOGO Cross-Validation)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'mrmr_metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_feature_importance_summary(best_k: int, ranked_indices: List[int],
                                   mrmr_scores: np.ndarray, final_model: Any,
                                   feature_names: List[str], results_folder: str) -> None:
    """
    Create a combined bar plot showing feature importance from both mRMR ranking
    and model-based feature importance for the selected top-k features.

    Args:
        best_k: Number of selected features
        ranked_indices: mRMR-ranked feature indices
        mrmr_scores: mRMR scores for all features
        final_model: Trained model (XGBoost or LDA)
        feature_names: List of all feature names
        results_folder: Path to save results
    """
    print("\n✓ Generating combined feature importance summary plot...")

    # Get wavelength mapping
    wavelength_map = create_wavelength_mapping(feature_names)

    # Get selected feature indices (top-k from mRMR ranking)
    selected_indices = ranked_indices[:best_k]

    # Extract mRMR scores for selected features
    selected_mrmr_scores = mrmr_scores[:best_k]

    # Extract model-based feature importance
    if hasattr(final_model, 'feature_importances_'):
        # XGBoost and tree-based models
        model_importance = final_model.feature_importances_
    elif hasattr(final_model, 'coef_') and final_model.coef_ is not None:
        # Linear models like LDA
        model_importance = np.abs(final_model.coef_).mean(axis=0)
    else:
        # Fallback if no importance available
        model_importance = np.ones(best_k)

    # Build DataFrame
    importance_data = []
    for i, idx in enumerate(selected_indices):
        feat_name = feature_names[idx]
        wl = wavelength_map.get(feat_name, 'N/A')

        # Format wavelength label
        if isinstance(wl, (int, float)):
            wl_label = f"{wl:.1f} nm"
        else:
            wl_label = str(feat_name)[:20]

        importance_data.append({
            'feature_name': feat_name,
            'feature_index': idx,
            'wavelength_nm': wl,
            'wavelength_label': wl_label,
            'mrmr_score': selected_mrmr_scores[i],
            'model_importance': model_importance[i]
        })

    df_importance = pd.DataFrame(importance_data)

    # Normalize scores to [0, 1]
    if df_importance['mrmr_score'].max() > 0:
        df_importance['normalized_mrmr'] = (
            (df_importance['mrmr_score'] - df_importance['mrmr_score'].min()) /
            (df_importance['mrmr_score'].max() - df_importance['mrmr_score'].min())
        )
    else:
        df_importance['normalized_mrmr'] = 0.5

    if df_importance['model_importance'].max() > 0:
        df_importance['normalized_model'] = (
            (df_importance['model_importance'] - df_importance['model_importance'].min()) /
            (df_importance['model_importance'].max() - df_importance['model_importance'].min())
        )
    else:
        df_importance['normalized_model'] = 0.5

    # Combined importance: 50% mRMR + 50% model importance
    df_importance['normalized_importance'] = (
        0.5 * df_importance['normalized_mrmr'] +
        0.5 * df_importance['normalized_model']
    )

    # Sort by combined importance (descending)
    df_importance = df_importance.sort_values('normalized_importance', ascending=True)

    # Save importance summary to CSV
    df_importance.to_csv(
        os.path.join(results_folder, 'logs', 'feature_importance_summary.csv'),
        index=False
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(8, int(best_k * 0.3))))

    # Use viridis colormap
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(df_importance['normalized_importance'])

    # Create horizontal bar plot
    bars = ax.barh(range(len(df_importance)), df_importance['normalized_importance'],
                   color=colors, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_importance['normalized_importance'])):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # Set y-tick labels to wavelengths
    ax.set_yticks(range(len(df_importance)))
    ax.set_yticklabels(df_importance['wavelength_label'], fontsize=10)

    # Labels and title
    ax.set_xlabel('Normalized Importance Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wavelength', fontsize=14, fontweight='bold')
    ax.set_title('Combined Feature Importance (mRMR + LDA)\nTop Features for CRACK Detection',
                 fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.15)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Importance Score', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'mrmr_feature_importance_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")
    print(f"  Saved importance data: {os.path.join(results_folder, 'logs', 'feature_importance_summary.csv')}")


def plot_logo_fold_performance(fold_results: List[Dict], results_folder: str) -> None:
    """
    Plot per-fold performance showing best k and F1_CRACK for each LOGO fold.

    Args:
        fold_results: List of dictionaries with per-fold results
        results_folder: Path to save results
    """
    print("\n✓ Generating LOGO fold performance plot...")

    fold_df = pd.DataFrame(fold_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Best k per fold
    fold_nums = fold_df['fold']
    best_ks = fold_df['best_k']

    ax1.bar(fold_nums, best_ks, color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
    ax1.axhline(best_ks.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {best_ks.mean():.1f}')
    ax1.set_xlabel('LOGO Fold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best k (Number of Features)', fontsize=12, fontweight='bold')
    ax1.set_title('Best k per LOGO Fold', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Best F1_CRACK per fold
    best_f1s = fold_df['best_f1_crack']

    ax2.bar(fold_nums, best_f1s, color='coral', edgecolor='black', linewidth=1.2, alpha=0.8)
    ax2.axhline(best_f1s.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {best_f1s.mean():.3f}')
    ax2.set_xlabel('LOGO Fold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Best F1_CRACK', fontsize=12, fontweight='bold')
    ax2.set_title('Best F1_CRACK per LOGO Fold', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'logo_fold_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline with hybrid LOGO strategy (no oversampling, sample weights)."""
    start_time = time.time()

    print("\n" + "="*80)
    print(" mRMR with HYBRID LOGO FOR CRACK DETECTION ")
    print(" Hyperspectral Imaging - Multi-class Classification ")
    print(" LOGO for CRACK/REGULAR + Static Split for OTHER classes ")
    print(" NO OVERSAMPLING - Class imbalance via sample weights ")
    print("="*80)

    # Setup
    setup_results_folder(RESULTS_FOLDER)

    # Load data
    df, X, y, feature_names, label_encoder = load_and_prepare_data(DATA_PATH, EXCLUDE_COLUMNS)

    # Prepare LOGO splits with filtered validation groups
    logo_validation_groups, always_train_groups, df_cr, train_other, val_other, test_other, feature_cols = prepare_logo_splits(
        df, feature_names,
        MODEL_CONFIG['random_state'],
        label_encoder
    )

    # Run mRMR feature selection with LOGO cross-validation
    results_df, best_k, ranked_indices, mrmr_scores, fold_results = run_mrmr_feature_selection_logo(
        logo_validation_groups, always_train_groups, df_cr, train_other, val_other,
        feature_names,
        MODEL_CONFIG['target_class'],
        label_encoder,
        MRMR_CONFIG,
        MODEL_CONFIG['random_state'],
        RESULTS_FOLDER
    )

    # Get validation metrics for best k
    best_val_metrics = {
        'f1_crack_val': results_df[results_df['k'] == best_k]['f1_crack_val_mean'].values[0],
        'f1_crack_val_std': results_df[results_df['k'] == best_k]['f1_crack_val_std'].values[0],
        'precision_crack_val': results_df[results_df['k'] == best_k]['precision_crack_val_mean'].values[0],
        'recall_crack_val': results_df[results_df['k'] == best_k]['recall_crack_val_mean'].values[0],
        'f2_crack_val': results_df[results_df['k'] == best_k]['f2_crack_val_mean'].values[0],
        'accuracy_val': results_df[results_df['k'] == best_k]['accuracy_val_mean'].values[0],
        'f1_weighted_val': results_df[results_df['k'] == best_k]['f1_weighted_val_mean'].values[0]
    }

    # Evaluate best model on TEST SET (used only ONCE here)
    final_model, results = evaluate_best_mrmr_model_logo(
        best_k, ranked_indices,
        df_cr, train_other, test_other,
        feature_names,
        MODEL_CONFIG['target_class'],
        best_val_metrics,
        MODEL_CONFIG['random_state'],
        RESULTS_FOLDER
    )

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_logo_fold_performance(fold_results, RESULTS_FOLDER)
    plot_mrmr_performance_vs_k(results_df, best_k, RESULTS_FOLDER, is_logo=True)
    plot_mrmr_ranked_wavelengths(ranked_indices, mrmr_scores, feature_names, RESULTS_FOLDER, top_n=30)
    plot_confusion_matrix_mrmr(results['confusion_matrix'], final_model.classes_, RESULTS_FOLDER)
    plot_mrmr_metrics_comparison(results_df, best_k, RESULTS_FOLDER, is_logo=True)
    plot_feature_importance_summary(best_k, ranked_indices, mrmr_scores, final_model, feature_names, RESULTS_FOLDER)

    # Save summary report
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_FOLDER, 'MRMR_LOGO_FEATURE_SELECTION_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("mRMR with HYBRID LOGO - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Total samples: {len(df)}\n")
        f.write(f"  Total features: {len(feature_names)}\n")
        f.write(f"  CRACK/REGULAR samples: {len(df_cr)}\n")
        f.write(f"  CRACK/REGULAR unique hs_dir: {len(logo_validation_groups) + len(always_train_groups)}\n")
        f.write(f"  - Validation groups (both classes): {len(logo_validation_groups)}\n")
        f.write(f"  - Always-train groups (single class): {len(always_train_groups)}\n")
        f.write(f"  OTHER (train) samples: {len(train_other)}\n")
        f.write(f"  OTHER (val) samples: {len(val_other)}\n")
        f.write(f"  OTHER (test) samples: {len(test_other)}\n\n")

        f.write("LOGO Strategy:\n")
        f.write(f"  - CRACK/REGULAR: LOGO cross-validation ({len(logo_validation_groups)} validation folds)\n")
        f.write(f"  - Validation groups: Only those with BOTH CRACK and REGULAR\n")
        f.write(f"  - Always-train groups: {len(always_train_groups)} groups with single class (never left out)\n")
        f.write(f"  - BACKGROUND/PLASTIC/BRANCH: Static 70/15/15 split\n")
        f.write(f"  - NO OVERSAMPLING: Class imbalance handled via sample weights\n")
        f.write(f"  - Test set: OTHER classes only (zero leakage)\n\n")

        f.write("mRMR Configuration:\n")
        for key, value in MRMR_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Best Configuration Found (LOGO Cross-Validation):\n")
        f.write(f"  Best k: {best_k} features\n")
        f.write(f"  F1-Score (CRACK, mean): {best_val_metrics['f1_crack_val']:.4f} ± {best_val_metrics['f1_crack_val_std']:.4f}\n")
        f.write(f"  Precision (CRACK, mean): {best_val_metrics['precision_crack_val']:.4f}\n")
        f.write(f"  Recall (CRACK, mean): {best_val_metrics['recall_crack_val']:.4f}\n")
        f.write(f"  F2-Score (CRACK, mean): {best_val_metrics['f2_crack_val']:.4f}\n")
        f.write(f"  Accuracy (mean): {best_val_metrics['accuracy_val']:.4f}\n")
        f.write(f"  Weighted F1 (mean): {best_val_metrics['f1_weighted_val']:.4f}\n\n")

        f.write("Per-Fold Summary:\n")
        for fold_res in fold_results:
            f.write(f"  Fold {fold_res['fold']} ({fold_res['val_group']}): ")
            f.write(f"best_k={fold_res['best_k']}, F1_CRACK={fold_res['best_f1_crack']:.4f}\n")
        f.write("\n")

        f.write("Final Model Performance on Test Set (OTHER classes only):\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"  Note: Test set contains only BACKGROUND/PLASTIC/BRANCH\n")
        f.write(f"  Note: CRACK/REGULAR used for LOGO cross-validation only\n\n")

        f.write("Classification Report (Test Set):\n")
        f.write(classification_report(test_other['label'], results['y_pred'],
                                     labels=final_model.classes_, zero_division=0))
        f.write("\n")

        f.write("Selected Wavelengths (mRMR Ranked - Reference from Fold 1):\n")
        wavelength_map = create_wavelength_mapping(feature_names)
        selected_indices = ranked_indices[:best_k]
        for rank, idx in enumerate(selected_indices, 1):
            feat = feature_names[idx]
            wl = wavelength_map[feat]
            if isinstance(wl, (int, float)):
                f.write(f"  {rank}. {feat}: {wl:.2f} nm (mRMR score: {mrmr_scores[rank-1]:.4f})\n")
            else:
                f.write(f"  {rank}. {feat} (mRMR score: {mrmr_scores[rank-1]:.4f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Total execution time: {time.time() - start_time:.2f} seconds\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary report saved to: {report_path}")

    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"✓ LOGO validation folds: {len(logo_validation_groups)}")
    print(f"✓ Always-train groups: {len(always_train_groups)}")
    print(f"✓ Best k: {best_k} features")
    print(f"✓ Best F1_CRACK (mean): {best_val_metrics['f1_crack_val']:.4f} ± {best_val_metrics['f1_crack_val_std']:.4f}")
    print(f"✓ Test accuracy (OTHER classes): {results['accuracy']:.4f}")
    print(f"✓ Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"✓ All results saved to: {RESULTS_FOLDER}")
    print("\n⚠️  IMPORTANT: Hybrid LOGO strategy ensures zero leakage!")
    print("  - CRACK/REGULAR: LOGO cross-validation with filtered groups")
    print("    * Validation: Only groups with BOTH CRACK and REGULAR")
    print("    * Training: Always includes single-class groups + other validation groups")
    print("  - OTHER classes: Static split (test_other independent)")
    print("  - NO OVERSAMPLING: Sample weights handle class imbalance")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

