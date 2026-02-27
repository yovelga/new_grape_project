"""
Recursive Feature Elimination (RFE) Feature Selection for Crack Detection
==========================================================================

This script implements Recursive Feature Elimination (RFE) with LDA to optimize
feature (wavelength) selection for multi-class classification of grape hyperspectral
images, with the primary objective of maximizing F1-score for the CRACK class.

Author: ML Research Team
Date: 2025-12-09
Purpose: Thesis research - Optimal wavelength selection for crack detection using RFE

Key Features:
- Multi-class classification with focus on CRACK class F1 optimization
- RFE (Recursive Feature Elimination) with LinearDiscriminantAnalysis
- Handles class imbalance via RandomOverSampler on training set
- NO TEST LEAKAGE: RFE uses only train_sub + validation, test used once at end
- Comprehensive logging and visualization for research thesis
- High-quality plots (300 DPI) for publication
- Systematic evaluation across different feature subset sizes

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
from sklearn.feature_selection import RFE
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
_PROJECT_ROOT = Path(__file__).resolve().parents[6]

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv")
RESULTS_FOLDER = str(_PROJECT_ROOT / r"results/rfe_feature_selection_crack_f1")

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

# RFE Configuration
RFE_CONFIG = {
    'min_features': 5,
    'max_features': 50,
    'step_size': 1,  # How many features to remove per iteration
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


def load_and_prepare_data(data_path: str, exclude_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load dataset and separate features from labels.

    Returns:
        X: Feature DataFrame
        y: Label Series
        feature_names: List of feature column names
    """
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    df = pd.read_csv(data_path)
    print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Separate labels
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")

    y = df['label']
    print(f"✓ Label distribution:\n{y.value_counts()}")

    # Extract features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]

    print(f"✓ Features extracted: {len(feature_cols)} wavelength channels")
    print(f"✓ Feature range: {X.min().min():.2f} to {X.max().max():.2f}")

    return X, y, feature_cols


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


def split_and_balance_data(X: pd.DataFrame, y: pd.Series,
                           test_size: float, random_state: int,
                           validation_size: float = 0.2) -> Tuple:
    """
    Split data into train/test and apply oversampling only to training set.
    Then create internal train_sub/validation split for RFE optimization.

    Returns:
        X_train_sub, X_val, y_train_sub, y_val, X_train_balanced, X_test, y_train_balanced, y_test
    """
    print("\n" + "="*80)
    print("SPLITTING AND BALANCING DATA (NO TEST LEAKAGE)")
    print("="*80)

    # Initial train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Initial train set: {X_train.shape[0]} samples")
    print(f"✓ Test set (WITHHELD): {X_test.shape[0]} samples")

    # Balance training set only
    ros = RandomOverSampler(random_state=random_state)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

    print(f"\n✓ Training set after oversampling: {X_train_balanced.shape[0]} samples")
    print(f"  Class distribution (balanced):\n{pd.Series(y_train_balanced).value_counts()}")

    # Create internal train_sub/validation split for RFE
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_balanced
    )

    print(f"\n✓ Internal RFE splits:")
    print(f"  Train_sub (for RFE fitting): {X_train_sub.shape[0]} samples")
    print(f"  Validation (for evaluation): {X_val.shape[0]} samples")
    print(f"  Class distribution (validation):\n{pd.Series(y_val).value_counts()}")

    print(f"\n✓ Test set (unbalanced, real-world distribution):")
    print(f"  Class distribution:\n{pd.Series(y_test).value_counts()}")
    print(f"\n⚠️  NOTE: Test set will be used ONLY ONCE at the end for final evaluation.")

    return X_train_sub, X_val, y_train_sub, y_val, X_train_balanced, X_test, y_train_balanced, y_test


# ============================================================================
# RFE FEATURE SELECTION FUNCTIONS
# ============================================================================

def run_rfe_feature_selection(X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series,
                              feature_names: List[str],
                              target_class: str,
                              rfe_config: Dict,
                              results_folder: str) -> Tuple[pd.DataFrame, int, np.ndarray]:
    """
    Run RFE for multiple values of k (number of features to select).

    IMPORTANT: NO TEST LEAKAGE
    - RFE uses only train_sub for fitting
    - Validation set used for performance evaluation
    - Test set is never touched

    Returns:
        results_df: DataFrame with results for each k
        best_k: Best number of features (based on validation CRACK F1)
        best_feature_mask: Boolean mask of selected features for best k
    """
    print("\n" + "="*80)
    print("RECURSIVE FEATURE ELIMINATION (RFE)")
    print("="*80)
    print(f"⚠️  RFE EVALUATED ON VALIDATION SET ONLY – TEST WITHHELD")
    print("="*80)
    print(f"Configuration:")
    for key, value in rfe_config.items():
        print(f"  {key}: {value}")
    print("="*80)

    min_features = rfe_config['min_features']
    max_features = rfe_config['max_features']
    step_size = rfe_config['step_size']

    # Create range of k values to test
    k_values = list(range(min_features, max_features + 1, step_size))

    results = []

    print(f"\n✓ Testing {len(k_values)} different feature subset sizes...")
    print(f"  Range: {min_features} to {max_features} features\n")

    # Iterate over different k values
    for k in tqdm(k_values, desc="RFE Progress"):
        try:
            # Create RFE selector with LDA estimator
            estimator = LinearDiscriminantAnalysis()
            selector = RFE(estimator=estimator, n_features_to_select=k, step=step_size)

            # Fit RFE on train_sub (NOT on test!)
            selector.fit(X_train_sub, y_train_sub)

            # Get selected features
            selected_mask = selector.support_
            selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]

            # Transform train_sub and validation sets
            X_train_sub_selected = X_train_sub.iloc[:, selected_mask]
            X_val_selected = X_val.iloc[:, selected_mask]

            # Train LDA on selected features
            lda_model = LinearDiscriminantAnalysis()
            lda_model.fit(X_train_sub_selected, y_train_sub)

            # Evaluate on VALIDATION set (not test!)
            y_pred_val = lda_model.predict(X_val_selected)

            # Compute metrics on validation
            accuracy_val = accuracy_score(y_val, y_pred_val)
            f1_weighted_val = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

            # Target class metrics on validation
            labels = sorted(y_val.unique())
            if target_class in labels:
                target_idx = labels.index(target_class)
                f1_crack_val = f1_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
                precision_crack_val = precision_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
                recall_crack_val = recall_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
                f2_crack_val = fbeta_score(y_val, y_pred_val, beta=2, labels=labels, average=None, zero_division=0)[target_idx]
            else:
                f1_crack_val = precision_crack_val = recall_crack_val = f2_crack_val = 0.0

            # Store results
            result = {
                'k': k,
                'num_features': k,
                'f1_crack_val': f1_crack_val,
                'precision_crack_val': precision_crack_val,
                'recall_crack_val': recall_crack_val,
                'f2_crack_val': f2_crack_val,
                'accuracy_val': accuracy_val,
                'f1_weighted_val': f1_weighted_val,
                'selected_features': ','.join(selected_features)
            }
            results.append(result)

        except Exception as e:
            print(f"\n⚠️  Warning: RFE with k={k} failed: {e}")
            # Store failed result with zero metrics
            result = {
                'k': k,
                'num_features': k,
                'f1_crack_val': 0.0,
                'precision_crack_val': 0.0,
                'recall_crack_val': 0.0,
                'f2_crack_val': 0.0,
                'accuracy_val': 0.0,
                'f1_weighted_val': 0.0,
                'selected_features': ''
            }
            results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best k based on validation CRACK F1
    best_idx = results_df['f1_crack_val'].idxmax()
    best_k = int(results_df.loc[best_idx, 'k'])
    best_f1_val = results_df.loc[best_idx, 'f1_crack_val']

    print("\n" + "="*80)
    print("RFE FEATURE SELECTION COMPLETE")
    print("="*80)
    print(f"✓ Best k: {best_k} features")
    print(f"✓ Best F1_CRACK (validation): {best_f1_val:.4f}")
    print(f"✓ Best accuracy (validation): {results_df.loc[best_idx, 'accuracy_val']:.4f}")

    # Refit RFE with best k to get the feature mask
    print(f"\n✓ Refitting RFE with best k={best_k}...")
    estimator_best = LinearDiscriminantAnalysis()
    selector_best = RFE(estimator=estimator_best, n_features_to_select=best_k, step=step_size)
    selector_best.fit(X_train_sub, y_train_sub)
    best_feature_mask = selector_best.support_

    # Save results log
    results_df.to_csv(os.path.join(results_folder, 'logs', 'rfe_results_by_k.csv'), index=False)
    print(f"✓ Results saved to: {os.path.join(results_folder, 'logs', 'rfe_results_by_k.csv')}")

    return results_df, best_k, best_feature_mask


def evaluate_best_rfe_model(best_feature_mask: np.ndarray,
                            X_train_balanced: pd.DataFrame, y_train_balanced: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            feature_names: List[str],
                            target_class: str,
                            best_k: int,
                            best_val_metrics: Dict,
                            results_folder: str) -> Tuple[Any, Dict]:
    """
    Train final model with best RFE-selected features on FULL balanced training set.
    Evaluate on test set (USED ONLY ONCE HERE - NO LEAKAGE).
    """
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON TEST SET (USED ONLY ONCE)")
    print("="*80)

    # Get selected features
    selected_features = [feature_names[i] for i, selected in enumerate(best_feature_mask) if selected]
    print(f"✓ Selected {len(selected_features)} features")

    # Subset data using best RFE mask
    X_train_subset = X_train_balanced.iloc[:, best_feature_mask]
    X_test_subset = X_test.iloc[:, best_feature_mask]

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
    model_path = os.path.join(results_folder, 'models', 'rfe_selected_lda_model.pkl')
    joblib.dump({
        'model': final_model,
        'selected_features': selected_features,
        'feature_mask': best_feature_mask,
        'best_k': best_k,
        'rfe_validation_metrics': best_val_metrics,  # Metrics from validation during RFE
        'test_metrics': {  # Metrics from final test evaluation
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
        'feature_name': selected_features,
        'feature_index': [i for i, sel in enumerate(best_feature_mask) if sel],
        'wavelength_nm': [wavelength_map.get(f, 'N/A') for f in selected_features]
    })

    # Add LDA coefficients if available
    if hasattr(final_model, 'coef_') and final_model.coef_ is not None:
        # For multi-class, take mean absolute coefficient across all discriminants
        coefs = np.abs(final_model.coef_).mean(axis=0)
        selected_features_df['lda_coefficient'] = coefs
        selected_features_df['importance_rank'] = selected_features_df['lda_coefficient'].rank(ascending=False)

    selected_features_df.to_csv(
        os.path.join(results_folder, 'logs', 'rfe_selected_features.csv'),
        index=False
    )
    print(f"✓ Selected features saved to: {os.path.join(results_folder, 'logs', 'rfe_selected_features.csv')}")

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

def plot_rfe_performance_vs_k(results_df: pd.DataFrame, best_k: int, results_folder: str) -> None:
    """Plot RFE performance (CRACK F1 on validation) vs number of features."""
    print("\n✓ Generating RFE performance vs k plot...")

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
    ax.set_title('RFE Feature Selection: Performance vs Number of Features',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'rfe_performance_vs_k.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_rfe_selected_wavelengths_importance(selected_features_df: pd.DataFrame,
                                             results_folder: str,
                                             top_n: int = 25) -> None:
    """Bar plot showing selected wavelengths with their LDA importance."""
    print("\n✓ Generating selected wavelengths importance plot...")

    # Sort by importance (LDA coefficient)
    if 'lda_coefficient' in selected_features_df.columns:
        plot_df = selected_features_df.sort_values('lda_coefficient', ascending=False).head(top_n)
        y_label = 'LDA Coefficient (Absolute Mean)'
        y_col = 'lda_coefficient'
    else:
        plot_df = selected_features_df.head(top_n)
        y_label = 'Feature Index'
        y_col = 'feature_index'

    # Create labels
    labels = []
    for _, row in plot_df.iterrows():
        wl = row['wavelength_nm']
        if isinstance(wl, (int, float)):
            labels.append(f"{wl:.1f}nm")
        else:
            labels.append(str(row['feature_name'])[:15])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(range(len(plot_df)), plot_df[y_col],
                   color='steelblue', edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, plot_df[y_col])):
        ax.text(val + val*0.02, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(y_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('Wavelength', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {len(plot_df)} RFE-Selected Wavelengths by Importance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'rfe_selected_wavelengths_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_confusion_matrix_rfe(cm: np.ndarray, classes: List[str], results_folder: str) -> None:
    """Plot confusion matrix heatmap for RFE model on test set."""
    print("\n✓ Generating confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=1, linecolor='gray', ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix: RFE-Selected LDA Model on Test Set',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'rfe_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_rfe_metrics_comparison(results_df: pd.DataFrame, best_k: int, results_folder: str) -> None:
    """Plot multiple metrics (F1, Precision, Recall) vs k."""
    print("\n✓ Generating RFE metrics comparison plot...")

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
    ax.set_title('RFE Feature Selection: Metrics Comparison',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'rfe_metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = time.time()

    print("\n" + "="*80)
    print(" RECURSIVE FEATURE ELIMINATION (RFE) FOR CRACK DETECTION ")
    print(" Hyperspectral Imaging - Multi-class Classification ")
    print("="*80)

    # Setup
    setup_results_folder(RESULTS_FOLDER)

    # Load data
    X, y, feature_names = load_and_prepare_data(DATA_PATH, EXCLUDE_COLUMNS)

    # Split and balance (creates train_sub/val/test splits)
    X_train_sub, X_val, y_train_sub, y_val, X_train_bal, X_test, y_train_bal, y_test = split_and_balance_data(
        X, y, MODEL_CONFIG['test_size'], MODEL_CONFIG['random_state'], validation_size=0.2
    )

    # Run RFE feature selection (uses ONLY train_sub and validation - NO TEST!)
    results_df, best_k, best_feature_mask = run_rfe_feature_selection(
        X_train_sub, y_train_sub,
        X_val, y_val,
        feature_names,
        MODEL_CONFIG['target_class'],
        RFE_CONFIG,
        RESULTS_FOLDER
    )

    # Get validation metrics for best k
    best_val_metrics = results_df[results_df['k'] == best_k].iloc[0].to_dict()

    # Evaluate best model on TEST SET (used only ONCE here)
    final_model, results = evaluate_best_rfe_model(
        best_feature_mask,
        X_train_bal, y_train_bal,  # Train on full balanced training set
        X_test, y_test,  # Evaluate on test set
        feature_names,
        MODEL_CONFIG['target_class'],
        best_k,
        best_val_metrics,
        RESULTS_FOLDER
    )

    # Load selected features for plotting
    selected_features_df = pd.read_csv(os.path.join(RESULTS_FOLDER, 'logs', 'rfe_selected_features.csv'))

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_rfe_performance_vs_k(results_df, best_k, RESULTS_FOLDER)
    plot_rfe_selected_wavelengths_importance(selected_features_df, RESULTS_FOLDER, top_n=min(25, len(selected_features_df)))
    plot_confusion_matrix_rfe(results['confusion_matrix'], final_model.classes_, RESULTS_FOLDER)
    plot_rfe_metrics_comparison(results_df, best_k, RESULTS_FOLDER)

    # Save summary report
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_FOLDER, 'RFE_FEATURE_SELECTION_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RECURSIVE FEATURE ELIMINATION (RFE) - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Total samples: {len(X)}\n")
        f.write(f"  Total features: {len(feature_names)}\n")
        f.write(f"  Training samples (balanced): {len(X_train_bal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n\n")

        f.write("RFE Configuration:\n")
        for key, value in RFE_CONFIG.items():
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

        f.write("Selected Wavelengths:\n")
        wavelength_map = create_wavelength_mapping(feature_names)
        selected_feats = [feature_names[i] for i, sel in enumerate(best_feature_mask) if sel]
        for feat in selected_feats:
            wl = wavelength_map[feat]
            if isinstance(wl, (int, float)):
                f.write(f"  - {feat}: {wl:.2f} nm\n")
            else:
                f.write(f"  - {feat}\n")

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

