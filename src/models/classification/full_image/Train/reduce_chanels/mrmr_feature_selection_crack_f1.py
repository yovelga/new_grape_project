"""
mRMR (Minimum Redundancy Maximum Relevance) Feature Selection for Crack Detection
==================================================================================

This script implements mRMR feature selection with LDA to optimize feature (wavelength)
selection for multi-class classification of grape hyperspectral images, with the primary
objective of maximizing F1-score for the CRACK class.

Author: ML Research Team
Date: 2025-12-09
Purpose: Thesis research - Optimal wavelength selection for crack detection using mRMR

Key Features:
- Multi-class classification with focus on CRACK class F1 optimization
- mRMR (Minimum Redundancy Maximum Relevance) feature ranking
- Handles class imbalance via RandomOverSampler on training set
- NO TEST LEAKAGE: mRMR uses only train_sub, test used once at end
- Comprehensive logging and visualization for research thesis
- High-quality plots (300 DPI) for publication
- Systematic evaluation across different feature subset sizes

mRMR Algorithm:
- Relevance: Mutual information with target labels (maximize)
- Redundancy: Average correlation with already selected features (minimize)
- Greedy forward selection balancing both criteria

Dependencies:
- numpy, pandas, scikit-learn, imblearn, matplotlib, seaborn, joblib, tqdm
"""

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv"
RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\results\mrmr_feature_selection_crack_f1"

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
    'classifier': 'LDA'  # LinearDiscriminantAnalysis
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


def load_and_prepare_data(data_path: str, exclude_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Load dataset, remove outliers, and separate features from labels.

    Returns:
        df: Full DataFrame (after outlier removal) with all columns including hs_dir
        X: Feature DataFrame
        y: Label Series
        feature_names: List of feature column names
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

    y = df['label']
    print(f"\n✓ Label distribution after outlier removal:")
    label_counts = y.value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(y)*100:.2f}%)")

    # Extract features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]

    print(f"\n✓ Features extracted: {len(feature_cols)} wavelength channels")
    print(f"✓ Feature range: {X.min().min():.2f} to {X.max().max():.2f}")

    return df, X, y, feature_cols


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


def split_and_balance_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float,
    random_state: int,
    validation_size: float = 0.2,
    group_col: str = "hs_dir"
) -> Tuple:
    """
    Split data into train/val/test using GroupShuffleSplit to ensure all pixels
    from the same hs_dir belong to exactly one split (no leakage between groups).
    Apply oversampling only to training set.
    Create internal train_sub/validation split for mRMR optimization.

    Args:
        df: Full DataFrame with features, labels, and group column
        feature_cols: List of feature column names
        test_size: Proportion for test set
        random_state: Random seed
        validation_size: Proportion for validation set (from remaining data)
        group_col: Column name for grouping (default: "hs_dir")

    Returns:
        X_train_sub, X_val, y_train_sub, y_val, X_train_balanced, X_test, y_train_balanced, y_test
    """
    from sklearn.model_selection import GroupShuffleSplit

    print("\n" + "="*80)
    print("GROUPED SPLITTING AND BALANCING DATA (NO TEST LEAKAGE)")
    print("="*80)
    print(f"✓ Grouping by: {group_col}")

    # Verify group column exists
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe")

    # Extract groups, labels, and features
    groups = df[group_col].values
    y = df["label"].values
    X = df[feature_cols]

    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Total unique {group_col}: {df[group_col].nunique()}")

    # Step 1: Split into train and temp (val + test) using GroupShuffleSplit
    temp_size = test_size + validation_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    print(f"\n✓ Train/Temp split:")
    print(f"  Train samples: {len(df_train)} ({len(df_train)/len(df)*100:.2f}%)")
    print(f"  Temp samples (val+test): {len(df_temp)} ({len(df_temp)/len(df)*100:.2f}%)")
    print(f"  Train {group_col} count: {df_train[group_col].nunique()}")
    print(f"  Temp {group_col} count: {df_temp[group_col].nunique()}")

    # Step 2: Split temp into val and test using GroupShuffleSplit
    val_ratio_within_temp = validation_size / (test_size + validation_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - val_ratio_within_temp,
                             random_state=random_state + 1)
    temp_groups = df_temp[group_col].values
    val_idx, test_idx = next(gss2.split(df_temp[feature_cols],
                                       df_temp["label"].values,
                                       temp_groups))

    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    print(f"\n✓ Val/Test split:")
    print(f"  Val samples: {len(df_val)} ({len(df_val)/len(df)*100:.2f}%)")
    print(f"  Test samples: {len(df_test)} ({len(df_test)/len(df)*100:.2f}%)")
    print(f"  Val {group_col} count: {df_val[group_col].nunique()}")
    print(f"  Test {group_col} count: {df_test[group_col].nunique()}")

    # Step 3: Verify no overlap between groups
    train_groups = set(df_train[group_col].unique())
    val_groups = set(df_val[group_col].unique())
    test_groups = set(df_test[group_col].unique())

    assert train_groups.isdisjoint(val_groups), f"Train and Val share {len(train_groups & val_groups)} groups!"
    assert train_groups.isdisjoint(test_groups), f"Train and Test share {len(train_groups & test_groups)} groups!"
    assert val_groups.isdisjoint(test_groups), f"Val and Test share {len(val_groups & test_groups)} groups!"
    print(f"\n✓ Group separation verified: No {group_col} overlap between train/val/test")

    # Step 4: Build X/y for each split
    X_train = df_train[feature_cols]
    y_train = df_train["label"]

    X_val_external = df_val[feature_cols]
    y_val_external = df_val["label"]

    X_test = df_test[feature_cols]
    y_test = df_test["label"]

    # Print class distributions
    print(f"\n✓ Class distribution in train split (before balancing):")
    for label, count in pd.Series(y_train).value_counts().items():
        print(f"    {label}: {count} ({count/len(y_train)*100:.2f}%)")

    print(f"\n✓ Class distribution in val split (external, not used in this version):")
    for label, count in pd.Series(y_val_external).value_counts().items():
        print(f"    {label}: {count} ({count/len(y_val_external)*100:.2f}%)")

    print(f"\n✓ Class distribution in test split (WITHHELD):")
    for label, count in pd.Series(y_test).value_counts().items():
        print(f"    {label}: {count} ({count/len(y_test)*100:.2f}%)")

    # Step 5: Apply class balancing (RandomOverSampler) only on training split
    ros = RandomOverSampler(random_state=random_state)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

    print(f"\n✓ Training set after oversampling: {X_train_balanced.shape[0]} samples")
    print(f"  Class distribution (balanced):")
    for label, count in pd.Series(y_train_balanced).value_counts().items():
        print(f"    {label}: {count} ({count/len(y_train_balanced)*100:.2f}%)")

    # Step 6: Create internal train_sub/val split for mRMR (from balanced training data)
    X_train_sub, X_val_internal, y_train_sub, y_val_internal = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_balanced
    )

    print(f"\n✓ Internal mRMR splits (from balanced training data):")
    print(f"  Train_sub (for mRMR scoring): {X_train_sub.shape[0]} samples")
    print(f"  Val_internal (for mRMR evaluation): {X_val_internal.shape[0]} samples")
    print(f"\n  Class distribution in train_sub:")
    for label, count in pd.Series(y_train_sub).value_counts().items():
        print(f"    {label}: {count} ({count/len(y_train_sub)*100:.2f}%)")
    print(f"\n  Class distribution in val_internal:")
    for label, count in pd.Series(y_val_internal).value_counts().items():
        print(f"    {label}: {count} ({count/len(y_val_internal)*100:.2f}%)")

    print(f"\n⚠️  NOTE: Test set will be used ONLY ONCE at the end for final evaluation.")
    print(f"⚠️  NOTE: Val_internal (from balanced training) is used for mRMR optimization.")
    print(f"⚠️  NOTE: External val/test splits respect {group_col} grouping.")

    # Return internal val as the validation set to maintain compatibility with existing code
    return X_train_sub, X_val_internal, y_train_sub, y_val_internal, X_train_balanced, X_test, y_train_balanced, y_test


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

    max_per_class = 30000

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
                            target_class: str) -> Dict:
    """
    Evaluate a feature subset of size k using the top-k features from mRMR ranking.

    Args:
        k: Number of top features to select
        ranked_indices: mRMR-ranked feature indices
        X_train_sub: Training subset features
        y_train_sub: Training subset labels
        X_val: Validation features
        y_val: Validation labels
        target_class: Target class name (e.g., 'CRACK')

    Returns:
        Dictionary with evaluation metrics
    """
    # Select top-k features
    selected_indices = ranked_indices[:k]

    # Subset data
    X_train_sub_selected = X_train_sub.iloc[:, selected_indices]
    X_val_selected = X_val.iloc[:, selected_indices]

    # Train LDA
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train_sub_selected, y_train_sub)

    # Predict on validation
    y_pred_val = lda_model.predict(X_val_selected)

    # Compute metrics
    accuracy_val = accuracy_score(y_val, y_pred_val)
    f1_weighted_val = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

    # Target class metrics
    labels = sorted(y_val.unique())
    if target_class in labels:
        target_idx = labels.index(target_class)
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


def run_mrmr_feature_selection(X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               feature_names: List[str],
                               target_class: str,
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
                target_class
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


def evaluate_best_mrmr_model(best_k: int, ranked_indices: List[int],
                             X_train_balanced: pd.DataFrame, y_train_balanced: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             feature_names: List[str],
                             target_class: str,
                             best_val_metrics: Dict,
                             results_folder: str) -> Tuple[Any, Dict]:
    """
    Train final model with best mRMR-selected features on FULL balanced training set.
    Evaluate on test set (USED ONLY ONCE HERE - NO LEAKAGE).
    """
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON TEST SET (USED ONLY ONCE)")
    print("="*80)

    # Get selected features
    selected_indices = ranked_indices[:best_k]
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"✓ Selected {len(selected_features)} features")

    # Subset data using best mRMR features
    X_train_subset = X_train_balanced.iloc[:, selected_indices]
    X_test_subset = X_test.iloc[:, selected_indices]

    print(f"\n✓ Training final model on FULL balanced training set:")
    print(f"  Training samples: {X_train_subset.shape[0]}")
    print(f"  Selected features: {X_train_subset.shape[1]}")

    # Train final model on full balanced training set
    final_model = LinearDiscriminantAnalysis()
    final_model.fit(X_train_subset, y_train_balanced)

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

    # Add LDA coefficients if available
    if hasattr(final_model, 'coef_') and final_model.coef_ is not None:
        coefs = np.abs(final_model.coef_).mean(axis=0)
        selected_features_df['lda_coefficient'] = coefs
        selected_features_df['importance_rank'] = selected_features_df['lda_coefficient'].rank(ascending=False)

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

def plot_mrmr_performance_vs_k(results_df: pd.DataFrame, best_k: int, results_folder: str) -> None:
    """Plot mRMR performance (CRACK F1 on validation) vs number of features."""
    print("\n✓ Generating mRMR performance vs k plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot F1_CRACK vs k
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
    ax.set_title('mRMR Feature Selection: Performance vs Number of Features',
                 fontsize=16, fontweight='bold', pad=20)
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
    ax.set_title('Confusion Matrix: mRMR-Selected LDA Model on Test Set',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'mrmr_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_mrmr_metrics_comparison(results_df: pd.DataFrame, best_k: int, results_folder: str) -> None:
    """Plot multiple metrics (F1, Precision, Recall) vs k."""
    print("\n✓ Generating mRMR metrics comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot multiple metrics
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
    ax.set_title('mRMR Feature Selection: Metrics Comparison',
                 fontsize=16, fontweight='bold', pad=20)
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
    and LDA coefficients for the selected top-k features.

    Args:
        best_k: Number of selected features
        ranked_indices: mRMR-ranked feature indices
        mrmr_scores: mRMR scores for all features
        final_model: Trained LDA model
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

    # Extract LDA coefficients for selected features
    if hasattr(final_model, 'coef_') and final_model.coef_ is not None:
        # Get absolute mean coefficient across all classes
        lda_coefs = np.abs(final_model.coef_).mean(axis=0)
    else:
        lda_coefs = np.ones(best_k)  # Fallback if no coefficients available

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
            'lda_coefficient': lda_coefs[i]
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

    if df_importance['lda_coefficient'].max() > 0:
        df_importance['normalized_lda'] = (
            (df_importance['lda_coefficient'] - df_importance['lda_coefficient'].min()) /
            (df_importance['lda_coefficient'].max() - df_importance['lda_coefficient'].min())
        )
    else:
        df_importance['normalized_lda'] = 0.5

    # Combined importance: 50% mRMR + 50% LDA
    df_importance['normalized_importance'] = (
        0.5 * df_importance['normalized_mrmr'] +
        0.5 * df_importance['normalized_lda']
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = time.time()

    print("\n" + "="*80)
    print(" mRMR (MINIMUM REDUNDANCY MAXIMUM RELEVANCE) FOR CRACK DETECTION ")
    print(" Hyperspectral Imaging - Multi-class Classification ")
    print("="*80)

    # Setup
    setup_results_folder(RESULTS_FOLDER)

    # Load data
    df, X, y, feature_names = load_and_prepare_data(DATA_PATH, EXCLUDE_COLUMNS)

    # Split and balance (creates train_sub/val/test splits with grouped split by hs_dir)
    X_train_sub, X_val, y_train_sub, y_val, X_train_bal, X_test, y_train_bal, y_test = split_and_balance_data(
        df, feature_names,
        MODEL_CONFIG['test_size'],
        MODEL_CONFIG['random_state'],
        validation_size=0.2
    )

    # Run mRMR feature selection (uses ONLY train_sub and validation - NO TEST!)
    results_df, best_k, ranked_indices, mrmr_scores = run_mrmr_feature_selection(
        X_train_sub, y_train_sub,
        X_val, y_val,
        feature_names,
        MODEL_CONFIG['target_class'],
        MRMR_CONFIG,
        RESULTS_FOLDER
    )

    # Get validation metrics for best k
    best_val_metrics = results_df[results_df['k'] == best_k].iloc[0].to_dict()

    # Evaluate best model on TEST SET (used only ONCE here)
    final_model, results = evaluate_best_mrmr_model(
        best_k, ranked_indices,
        X_train_bal, y_train_bal,
        X_test, y_test,
        feature_names,
        MODEL_CONFIG['target_class'],
        best_val_metrics,
        RESULTS_FOLDER
    )

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_mrmr_performance_vs_k(results_df, best_k, RESULTS_FOLDER)
    plot_mrmr_ranked_wavelengths(ranked_indices, mrmr_scores, feature_names, RESULTS_FOLDER, top_n=30)
    plot_confusion_matrix_mrmr(results['confusion_matrix'], final_model.classes_, RESULTS_FOLDER)
    plot_mrmr_metrics_comparison(results_df, best_k, RESULTS_FOLDER)
    plot_feature_importance_summary(best_k, ranked_indices, mrmr_scores, final_model, feature_names, RESULTS_FOLDER)

    # Save summary report
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_FOLDER, 'MRMR_FEATURE_SELECTION_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("mRMR (MINIMUM REDUNDANCY MAXIMUM RELEVANCE) - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Total samples: {len(X)}\n")
        f.write(f"  Total features: {len(feature_names)}\n")
        f.write(f"  Training samples (balanced): {len(X_train_bal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n\n")

        f.write("mRMR Configuration:\n")
        for key, value in MRMR_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Best Configuration Found (Validation Set Performance):\n")
        f.write(f"  Best k: {best_k} features\n")
        f.write(f"  F1-Score (CRACK, validation): {best_val_metrics['f1_crack_val']:.4f}\n")
        f.write(f"  Precision (CRACK, validation): {best_val_metrics['precision_crack_val']:.4f}\n")
        f.write(f"  Recall (CRACK, validation): {best_val_metrics['recall_crack_val']:.4f}\n")
        f.write(f"  F2-Score (CRACK, validation): {best_val_metrics['f2_crack_val']:.4f}\n")
        f.write(f"  Accuracy (validation): {best_val_metrics['accuracy_val']:.4f}\n")
        f.write(f"  Weighted F1 (validation): {best_val_metrics['f1_weighted_val']:.4f}\n\n")

        f.write("Final Model Performance on Test Set (NO LEAKAGE - Used Only Once):\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        crack_report = results['classification_report'].get(MODEL_CONFIG['target_class'], {})
        f.write(f"  CRACK Precision: {crack_report.get('precision', 0):.4f}\n")
        f.write(f"  CRACK Recall: {crack_report.get('recall', 0):.4f}\n")
        f.write(f"  CRACK F1-Score: {crack_report.get('f1-score', 0):.4f}\n\n")

        f.write("Selected Wavelengths (mRMR Ranked):\n")
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
        f.write("Classification Report (Test Set):\n")
        f.write("="*80 + "\n")
        f.write(classification_report(y_test, results['y_pred'], zero_division=0))

        f.write("\n" + "="*80 + "\n")
        f.write(f"Total execution time: {time.time() - start_time:.2f} seconds\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary report saved to: {report_path}")

    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"✓ Best k: {best_k} features")
    print(f"✓ Best F1_CRACK (validation): {best_val_metrics['f1_crack_val']:.4f}")
    print(f"✓ Test F1_CRACK: {results['classification_report'].get(MODEL_CONFIG['target_class'], {}).get('f1-score', 0):.4f}")
    print(f"✓ Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"✓ All results saved to: {RESULTS_FOLDER}")
    print("\n⚠️  IMPORTANT: Test set was used ONLY ONCE for final evaluation - NO LEAKAGE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

