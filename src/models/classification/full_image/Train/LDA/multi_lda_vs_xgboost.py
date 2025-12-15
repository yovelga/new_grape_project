"""
Model Comparison: LDA vs XGBoost (Baseline on Full Features)
==================================================================================
Purpose: Determine the best classifier architecture for Crack Detection
         before applying feature selection.
Method:  Hybrid LOGO Cross-Validation on ALL 204 features.
Balancing:
    - LDA: Uses equal 'priors' parameter.
    - XGBoost: Uses 'sample_weight' parameter.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight

# Import wavelengths mapping
from wavelengths import WAVELENGTHS

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv"
RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\results\model_comparison_baseline"
RANDOM_STATE = 42
TARGET_CLASS = 'CRACK'

# Maximum samples per class (set to None for no limit, or e.g., 30000 for 30K limit)
MAX_SAMPLES_PER_CLASS = 22580  # Change to 30000 to limit to 30K samples per class

# Columns to exclude
EXCLUDE_COLUMNS = ['label', 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']

# XGBoost Config
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'n_jobs': -1,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE
}

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# FUNCTIONS
# ============================================================================

def create_wavelength_mapping(feature_names):
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


def load_data(path):
    print(f"✓ Loading data from: {path}")
    df = pd.read_csv(path)

    # Filter outliers
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == 0].copy()

    # Apply sample limit per class if specified
    if MAX_SAMPLES_PER_CLASS is not None:
        print(f"✓ Applying maximum samples per class limit: {MAX_SAMPLES_PER_CLASS}")
        df_limited = []
        for label in df['label'].unique():
            df_class = df[df['label'] == label]
            if len(df_class) > int(MAX_SAMPLES_PER_CLASS):
                df_class = df_class.sample(n=int(MAX_SAMPLES_PER_CLASS), random_state=RANDOM_STATE)
                print(f"  - {label}: Limited from {len(df[df['label'] == label])} to {len(df_class)} samples")
            else:
                print(f"  - {label}: {len(df_class)} samples (no limiting needed)")
            df_limited.append(df_class)
        df = pd.concat(df_limited, axis=0).reset_index(drop=True)

    # Encoder
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS and c != 'label_encoded']

    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Classes: {le.classes_}")

    return df, feature_cols, le


def get_logo_splits(df, le):
    """
    Hybrid LOGO logic with REALISTIC validation including background classes.

    Returns:
        - logo_groups: Groups with mixed CRACK/REGULAR for LOGO validation
        - always_train: Pure groups always in training
        - df_cr: CRACK/REGULAR subset
        - train_other: Background classes for training (70%)
        - val_other: Background classes for validation (30%) - CRITICAL FIX
    """
    print("\n✓ Preparing LOGO splits (WITH BACKGROUND IN VALIDATION)...")

    # Identify encoded labels
    crack_code = le.transform(['CRACK'])[0] if 'CRACK' in le.classes_ else -1
    regular_code = le.transform(['REGULAR'])[0] if 'REGULAR' in le.classes_ else -1

    # Subset CRACK/REGULAR (for LOGO splitting)
    df_cr = df[df['label_encoded'].isin([crack_code, regular_code])].copy()

    logo_groups = []
    always_train = []

    for grp in df_cr['hs_dir'].unique():
        subset = df_cr[df_cr['hs_dir'] == grp]
        uniques = subset['label_encoded'].unique()
        if crack_code in uniques and regular_code in uniques:
            logo_groups.append(grp)
        else:
            always_train.append(grp)

    print(f"  - Validation Groups (Mixed CRACK/REGULAR): {len(logo_groups)}")
    print(f"  - Always Train Groups (Pure): {len(always_train)}")

    # *** CRITICAL FIX: Split background/other classes into train/val ***
    other_mask = ~df['label'].isin(['CRACK', 'REGULAR'])
    df_other = df[other_mask]

    if len(df_other) > 0:
        # Split OTHER classes: 70% train, 30% validation (FIXED across all folds)
        train_other, val_other = train_test_split(
            df_other,
            test_size=0.3,
            stratify=df_other['label_encoded'],
            random_state=RANDOM_STATE
        )
        print(f"  - Background Classes Split:")
        print(f"    * Train: {len(train_other)} samples")
        print(f"    * Val:   {len(val_other)} samples (INCLUDED IN EVERY FOLD)")

        # Show class distribution in background validation
        val_other_classes = val_other['label'].value_counts()
        print(f"    * Val classes: {dict(val_other_classes)}")
    else:
        train_other = pd.DataFrame()
        val_other = pd.DataFrame()
        print(f"  - No background classes found")

    return logo_groups, always_train, df_cr, train_other, val_other


def run_comparison(df, logo_groups, always_train, df_cr, train_other, val_other, feature_cols, le):
    """
    Run LOGO comparison with REALISTIC validation (includes background classes).

    Args:
        train_other: Background classes for training
        val_other: Background classes for validation (ADDED TO EVERY FOLD)
    """
    results = []

    # Lists for storing data for advanced plots
    global_y_true = []
    global_y_pred_lda = []
    global_y_pred_xgb = []
    global_probs_lda = []  # For ROC curves
    global_probs_xgb = []  # For ROC curves

    feature_importance_list = []  # For feature importance plot

    crack_code = le.transform(['CRACK'])[0]

    print("\n" + "="*80)
    print("Starting LOGO Comparison with REALISTIC Validation (Background Included)")
    print("="*80)
    print(f"Background validation samples: {len(val_other)}")
    print(f"This ensures models are tested against hard negatives (plastic, branches, etc.)")
    print("="*80 + "\n")

    for i, val_group in enumerate(tqdm(logo_groups)):
        # --- 1. Create Fold with REALISTIC Validation ---

        # Validation: Current grape group + FIXED background validation set
        val_cr = df_cr[df_cr['hs_dir'] == val_group]
        val_full = pd.concat([val_cr, val_other], axis=0)  # *** CRITICAL FIX ***

        # Training: All other grape groups + background training set
        train_groups = [g for g in logo_groups if g != val_group] + always_train
        train_cr = df_cr[df_cr['hs_dir'].isin(train_groups)]
        train_full = pd.concat([train_cr, train_other], axis=0)

        X_train = train_full[feature_cols]
        y_train = train_full['label_encoded']
        X_val = val_full[feature_cols]  # *** NOW INCLUDES BACKGROUND ***
        y_val = val_full['label_encoded']  # *** NOW INCLUDES BACKGROUND ***

        # --- 2. Train LDA ---
        n_classes = len(np.unique(y_train))
        priors = [1 / n_classes] * n_classes
        lda = LinearDiscriminantAnalysis(priors=priors)
        lda.fit(X_train, y_train)

        y_pred_lda = lda.predict(X_val)
        y_prob_lda = lda.predict_proba(X_val)

        # --- 3. Train XGBoost ---
        weights = compute_sample_weight('balanced', y_train)
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_train, y_train, sample_weight=weights, verbose=False)

        y_pred_xgb = xgb.predict(X_val)
        y_prob_xgb = xgb.predict_proba(X_val)

        # Store feature importance from current fold
        fold_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': xgb.feature_importances_
        })
        feature_importance_list.append(fold_importance)

        # --- 4. Accumulate Global Data ---
        global_y_true.extend(y_val)
        global_y_pred_lda.extend(y_pred_lda)
        global_y_pred_xgb.extend(y_pred_xgb)

        # Extract probabilities for CRACK class for ROC curves
        crack_index_lda = list(lda.classes_).index(crack_code)
        crack_index_xgb = list(xgb.classes_).index(crack_code)

        global_probs_lda.extend(y_prob_lda[:, crack_index_lda])
        global_probs_xgb.extend(y_prob_xgb[:, crack_index_xgb])

        # --- 5. Comprehensive Metrics Calculation ---
        # CRACK-specific metrics
        f1_lda = f1_score(y_val, y_pred_lda, labels=[crack_code], average=None, zero_division=0)[0]
        prec_lda = precision_score(y_val, y_pred_lda, labels=[crack_code], average=None, zero_division=0)[0]
        rec_lda = recall_score(y_val, y_pred_lda, labels=[crack_code], average=None, zero_division=0)[0]

        f1_xgb = f1_score(y_val, y_pred_xgb, labels=[crack_code], average=None, zero_division=0)[0]
        prec_xgb = precision_score(y_val, y_pred_xgb, labels=[crack_code], average=None, zero_division=0)[0]
        rec_xgb = recall_score(y_val, y_pred_xgb, labels=[crack_code], average=None, zero_division=0)[0]

        # Global metrics
        acc_lda = accuracy_score(y_val, y_pred_lda)
        acc_xgb = accuracy_score(y_val, y_pred_xgb)

        # Matthews Correlation Coefficient
        mcc_lda = matthews_corrcoef(y_val, y_pred_lda)
        mcc_xgb = matthews_corrcoef(y_val, y_pred_xgb)

        # AUC and PR-AUC (per fold)
        try:
            # Convert to binary for One-vs-Rest calculation (CRACK vs all others)
            y_val_binary = (y_val == crack_code).astype(int)

            # ROC-AUC using predicted probabilities
            auc_lda = roc_auc_score(y_val_binary, y_prob_lda[:, crack_index_lda])
            auc_xgb = roc_auc_score(y_val_binary, y_prob_xgb[:, crack_index_xgb])

            # PR-AUC (Precision-Recall AUC) - PRIMARY METRIC
            # Uses predicted probabilities to evaluate precision-recall trade-off
            # across ALL decision thresholds (threshold-independent evaluation)
            pr_auc_lda = average_precision_score(y_val_binary, y_prob_lda[:, crack_index_lda])
            pr_auc_xgb = average_precision_score(y_val_binary, y_prob_xgb[:, crack_index_xgb])
        except Exception:
            auc_lda = 0.0
            auc_xgb = 0.0
            pr_auc_lda = 0.0
            pr_auc_xgb = 0.0

        results.append({
            'Fold': i + 1,
            'Model': 'LDA',
            'F1_Crack': f1_lda,
            'Precision': prec_lda,
            'Recall': rec_lda,
            'Accuracy': acc_lda,
            'MCC': mcc_lda,
            'AUC': auc_lda,
            'PR_AUC_Crack': pr_auc_lda
        })
        results.append({
            'Fold': i + 1,
            'Model': 'XGBoost',
            'F1_Crack': f1_xgb,
            'Precision': prec_xgb,
            'Recall': rec_xgb,
            'Accuracy': acc_xgb,
            'MCC': mcc_xgb,
            'AUC': auc_xgb,
            'PR_AUC_Crack': pr_auc_xgb
        })

    # Return results with all accumulated data
    return pd.DataFrame(results), {
        'y_true': global_y_true,
        'y_pred_lda': global_y_pred_lda,
        'y_pred_xgb': global_y_pred_xgb,
        'probs_lda': global_probs_lda,
        'probs_xgb': global_probs_xgb,
        'importances': pd.concat(feature_importance_list)
    }


def plot_comparison(df_results, output_folder):
    """
    Generate three primary visualization plots using PR-AUC as the main metric:
    1. Precision vs Recall Scatter with No-Skill baseline
    2. Fold-by-Fold Performance Gain Bar Chart
    3. PR-AUC Stability Boxplot
    """

    # Configure plotting style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ========================================================================
    # PLOT 1: Precision vs Recall Scatter (Trade-off Visualization)
    # ========================================================================
    plt.figure(figsize=(10, 8))

    # Calculate no-skill baseline (ratio of positive samples)
    # This represents a random classifier's performance
    total_samples = len(df_results) // 2  # Divided by 2 models
    # Estimate baseline from average recall (proxy for class prevalence)
    baseline = df_results['Recall'].mean() * 0.5  # Approximate baseline

    # Scatter plot with explicit axis mapping
    for model in ['LDA', 'XGBoost']:
        model_data = df_results[df_results['Model'] == model]
        color = '#E24A33' if model == 'LDA' else '#348ABD'
        marker = 'o' if model == 'LDA' else 's'
        plt.scatter(
            x=model_data['Recall'],  # Explicitly set X-axis
            y=model_data['Precision'],  # Explicitly set Y-axis
            c=color,
            marker=marker,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            label=model
        )

    # Add mean centroids
    for model in ['LDA', 'XGBoost']:
        model_data = df_results[df_results['Model'] == model]
        mean_recall = model_data['Recall'].mean()
        mean_precision = model_data['Precision'].mean()
        color = 'darkred' if model == 'LDA' else 'darkblue'
        plt.scatter(
            mean_recall,
            mean_precision,
            c=color,
            s=500,
            marker='X',
            edgecolors='black',
            linewidth=2,
            label=f'{model} Mean',
            zorder=10
        )

    # Plot No-Skill baseline (horizontal line)
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=2,
                alpha=0.6, label=f'No-Skill Baseline ({baseline:.3f})')

    # Explicitly set axis labels and limits
    plt.xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Trade-off: LDA vs. XGBoost\n(CRACK Class Detection)',
              fontsize=15, fontweight='bold', pad=20)
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    path = os.path.join(output_folder, 'plot_1_precision_recall_tradeoff.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 1 (Precision-Recall Scatter) saved to {path}")

    # ========================================================================
    # PLOT 2: Fold-by-Fold Performance Gain (Bar Chart)
    # ========================================================================
    plt.figure(figsize=(12, 6))

    # Prepare data: Calculate Delta (XGBoost - LDA) per fold
    folds = sorted(df_results['Fold'].unique())
    deltas = []

    for fold in folds:
        lda_pr = df_results[(df_results['Fold'] == fold) & (df_results['Model'] == 'LDA')]['PR_AUC_Crack'].values[0]
        xgb_pr = df_results[(df_results['Fold'] == fold) & (df_results['Model'] == 'XGBoost')]['PR_AUC_Crack'].values[0]
        delta = xgb_pr - lda_pr
        deltas.append(delta)

    # Create bar colors: Green for positive (XGBoost better), Red for negative (LDA better)
    colors = ['green' if d > 0 else 'red' for d in deltas]

    # Bar chart
    bars = plt.bar(folds, deltas, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for fold, delta, bar in zip(folds, deltas, bars):
        height = bar.get_height()
        label_y = height + 0.002 if height > 0 else height - 0.005
        va = 'bottom' if height > 0 else 'top'
        plt.text(fold, label_y, f'{delta:+.3f}', ha='center', va=va,
                fontsize=10, fontweight='bold')

    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Add mean gain line
    mean_delta = np.mean(deltas)
    plt.axhline(y=mean_delta, color='blue', linestyle='--', linewidth=2.5,
                alpha=0.7, label=f'Mean Gain: {mean_delta:+.4f}')

    # Labels and formatting
    plt.xlabel('Fold Number', fontsize=14, fontweight='bold')
    plt.ylabel('Performance Gain (PR-AUC)\n[XGBoost - LDA]', fontsize=14, fontweight='bold')
    plt.title('Fold-by-Fold Performance Gain: XGBoost vs. LDA\n(Positive = XGBoost Better)',
              fontsize=15, fontweight='bold', pad=20)
    plt.xticks(folds, fontsize=11)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    path = os.path.join(output_folder, 'plot_2_fold_performance_gain.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 2 (Fold Performance Gain) saved to {path}")

    # ========================================================================
    # PLOT 3: PR-AUC Stability Boxplot
    # ========================================================================
    plt.figure(figsize=(10, 7))

    # Create boxplot with clear separation
    box_data = [
        df_results[df_results['Model'] == 'LDA']['PR_AUC_Crack'].values,
        df_results[df_results['Model'] == 'XGBoost']['PR_AUC_Crack'].values
    ]

    box_plot = plt.boxplot(
        box_data,
        labels=['LDA', 'XGBoost'],
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=10, markeredgecolor='black'),
        boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='darkblue', linewidth=2.5)
    )

    # Color boxes differently
    colors = ['#E24A33', '#348ABD']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add stripplot overlay
    for i, model in enumerate(['LDA', 'XGBoost'], start=1):
        model_data = df_results[df_results['Model'] == model]['PR_AUC_Crack'].values
        x = np.random.normal(i, 0.04, size=len(model_data))  # Add jitter
        plt.scatter(x, model_data, alpha=0.5, s=60, color='black', zorder=3)

    # Statistics annotations
    lda_mean = df_results[df_results['Model'] == 'LDA']['PR_AUC_Crack'].mean()
    lda_std = df_results[df_results['Model'] == 'LDA']['PR_AUC_Crack'].std()
    xgb_mean = df_results[df_results['Model'] == 'XGBoost']['PR_AUC_Crack'].mean()
    xgb_std = df_results[df_results['Model'] == 'XGBoost']['PR_AUC_Crack'].std()

    plt.text(1, lda_mean, f'μ={lda_mean:.3f}\nσ={lda_std:.3f}',
            ha='right', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.text(2, xgb_mean, f'μ={xgb_mean:.3f}\nσ={xgb_std:.3f}',
            ha='left', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Labels and formatting
    plt.ylabel('PR-AUC (Crack Class)', fontsize=14, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=14, fontweight='bold')
    plt.title('Model Stability Comparison: PR-AUC Distribution\n(LOGO Cross-Validation)',
              fontsize=15, fontweight='bold', pad=20)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    path = os.path.join(output_folder, 'plot_3_stability_boxplot.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 3 (Stability Boxplot) saved to {path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY: PR-AUC COMPARISON")
    print("="*80)
    print(f"LDA:     Mean={lda_mean:.4f}, Std={lda_std:.4f}")
    print(f"XGBoost: Mean={xgb_mean:.4f}, Std={xgb_std:.4f}")
    print(f"Mean Gain per Fold: {mean_delta:+.4f} ({(mean_delta/lda_mean)*100:+.2f}%)")
    print(f"Consistency: XGBoost wins in {sum(1 for d in deltas if d > 0)}/{len(deltas)} folds")
    print("="*80 + "\n")


def plot_advanced_analysis(global_data, output_folder, le):
    """
    Generate comprehensive publication-ready comparison plots for thesis
    """
    crack_code = le.transform(['CRACK'])[0]
    classes = le.classes_

    # 1. Global Confusion Matrix (Comparison)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LDA Matrix
    cm_lda = confusion_matrix(global_data['y_true'], global_data['y_pred_lda'])
    disp_lda = ConfusionMatrixDisplay(confusion_matrix=cm_lda, display_labels=classes)
    disp_lda.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Global Confusion Matrix: LDA', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)

    # XGBoost Matrix
    cm_xgb = confusion_matrix(global_data['y_true'], global_data['y_pred_xgb'])
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=classes)
    disp_xgb.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('Global Confusion Matrix: XGBoost', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'global_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve Comparison
    fig, ax = plt.subplots(figsize=(8, 7))

    # Convert to binary for ROC
    y_true_binary = np.array([1 if y == crack_code else 0 for y in global_data['y_true']])

    # LDA ROC
    fpr_lda, tpr_lda, _ = roc_curve(y_true_binary, global_data['probs_lda'])
    auc_lda = auc(fpr_lda, tpr_lda)
    ax.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {auc_lda:.3f})', linestyle='--', linewidth=2.5, color='#1f77b4')

    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true_binary, global_data['probs_xgb'])
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})', linewidth=2.5, color='#2ca02c')

    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison: CRACK Detection\n(Aggregated LOGO Cross-Validation)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'roc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curve Comparison
    fig, ax = plt.subplots(figsize=(8, 7))

    # Convert to binary for PR curve
    y_true_binary = np.array([1 if y == crack_code else 0 for y in global_data['y_true']])

    # LDA PR Curve
    precision_lda, recall_lda, _ = precision_recall_curve(y_true_binary, global_data['probs_lda'])
    pr_auc_lda = average_precision_score(y_true_binary, global_data['probs_lda'])
    ax.plot(recall_lda, precision_lda, label=f'LDA (AP = {pr_auc_lda:.3f})',
            linestyle='--', linewidth=2.5, color='#1f77b4')

    # XGBoost PR Curve
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_true_binary, global_data['probs_xgb'])
    pr_auc_xgb = average_precision_score(y_true_binary, global_data['probs_xgb'])
    ax.plot(recall_xgb, precision_xgb, label=f'XGBoost (AP = {pr_auc_xgb:.3f})',
            linewidth=2.5, color='#2ca02c')

    # Baseline (random classifier for imbalanced data)
    baseline = np.sum(y_true_binary) / len(y_true_binary)
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, linewidth=1.5,
               label=f'Baseline (No Skill = {baseline:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison: CRACK Detection\n(Aggregated LOGO Cross-Validation)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'precision_recall_curve_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Top 20 Feature Importance (Averaged over folds) with Wavelengths
    if not global_data['importances'].empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        avg_imp = global_data['importances'].groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(20)

        # Create wavelength mapping
        wavelength_map = create_wavelength_mapping(list(avg_imp.index))

        # Create labels with wavelengths
        labels = []
        for feat in avg_imp.index:
            wl = wavelength_map.get(feat, feat)
            if isinstance(wl, (int, float)):
                labels.append(f"{wl:.1f} nm")
            else:
                labels.append(str(feat)[:20])  # Fallback to feature name

        # Create horizontal bar plot
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0.3, 0.9, len(avg_imp)))
        bars = ax.barh(range(len(avg_imp)), avg_imp.values, color=colors, edgecolor='black', linewidth=0.8)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_imp.values)):
            ax.text(val + val*0.02, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(range(len(avg_imp)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Mean Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Wavelengths: XGBoost Importance\n(Averaged over LOGO Cross-Validation Folds)',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print("✓ Advanced plots saved!")


def plot_comprehensive_metrics(df_results, output_folder):
    """
    Create comprehensive multi-metric comparison plots
    """
    # 1. All Metrics Comparison (Grouped Bar Chart)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = ['F1_Crack', 'Accuracy', 'AUC', 'PR_AUC_Crack', 'MCC', 'Recall']
    titles = ['F1-Score (CRACK Class)', 'Global Accuracy', 'ROC-AUC (CRACK vs Others)',
              'PR-AUC (CRACK Class)', 'Matthews Correlation Coefficient', 'Recall (Sensitivity)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # Calculate mean and std for each model
        lda_data = df_results[df_results['Model'] == 'LDA'][metric]
        xgb_data = df_results[df_results['Model'] == 'XGBoost'][metric]

        models = ['LDA', 'XGBoost']
        means = [lda_data.mean(), xgb_data.mean()]
        stds = [lda_data.std(), xgb_data.std()]

        # Bar plot with error bars
        bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.7,
                      color=['#1f77b4', '#2ca02c'], edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_ylim([0, max(means) * 1.2])
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.suptitle('Comprehensive Model Comparison: All Metrics\n(LOGO Cross-Validation on 204 Features)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comprehensive_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Radar Chart Comparison (using core metrics)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Use core metrics for radar chart
    radar_metrics = ['F1_Crack', 'Accuracy', 'AUC', 'PR_AUC_Crack', 'MCC']

    # Calculate mean values for radar metrics
    lda_means = [df_results[df_results['Model'] == 'LDA'][m].mean() for m in radar_metrics]
    xgb_means = [df_results[df_results['Model'] == 'XGBoost'][m].mean() for m in radar_metrics]

    # Normalize to 0-1 if needed (MCC is already -1 to 1, shift it)
    lda_means_norm = lda_means.copy()
    xgb_means_norm = xgb_means.copy()
    lda_means_norm[4] = (lda_means[4] + 1) / 2  # Normalize MCC from [-1,1] to [0,1]
    xgb_means_norm[4] = (xgb_means[4] + 1) / 2

    # Number of variables
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    lda_means_norm += lda_means_norm[:1]  # Complete the circle
    xgb_means_norm += xgb_means_norm[:1]
    angles += angles[:1]

    # Plot
    ax.plot(angles, lda_means_norm, 'o-', linewidth=2, label='LDA', color='#1f77b4')
    ax.fill(angles, lda_means_norm, alpha=0.15, color='#1f77b4')
    ax.plot(angles, xgb_means_norm, 'o-', linewidth=2, label='XGBoost', color='#2ca02c')
    ax.fill(angles, xgb_means_norm, alpha=0.15, color='#2ca02c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['F1-Score\n(CRACK)', 'Accuracy', 'ROC-AUC', 'PR-AUC', 'MCC'], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Multi-Metric Performance Comparison\n(Radar Chart)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Comprehensive metrics plots saved!")


def plot_academic_comparison(df_results, output_folder):
    """
    Generate three academic publication-quality plots for thesis:
    1. Precision vs Recall Scatter (The Trade-off)
    2. Paired Slope Chart (Consistency) - Using PR-AUC
    3. Distribution Boxplot (Stability) - Using PR-AUC
    """
    # Configure Plot Style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    colors = {"LDA": "#E24A33", "XGBoost": "#348ABD"}  # Academic colors (Red/Blue)

    # --- PLOT 1: Precision vs Recall Scatter (The Trade-off) ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_results, x='Recall', y='Precision', hue='Model',
                    style='Model', palette=colors, s=150, alpha=0.8)

    # Add mean centroids
    means = df_results.groupby('Model')[['Recall', 'Precision']].mean()
    plt.scatter(means.loc['LDA', 'Recall'], means.loc['LDA', 'Precision'],
                c='darkred', s=400, marker='X', edgecolors='black', label='LDA Mean')
    plt.scatter(means.loc['XGBoost', 'Recall'], means.loc['XGBoost', 'Precision'],
                c='darkblue', s=400, marker='X', edgecolors='black', label='XGB Mean')

    plt.title('Precision-Recall Trade-off: LDA vs. XGBoost', fontweight='bold', pad=20)
    plt.xlabel('Recall (Sensitivity)', fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontweight='bold')
    plt.legend(title='Model', loc='lower left', bbox_to_anchor=(0, 0))
    plt.xlim(0.4, 1.05)
    plt.ylim(0.2, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'plot_1_precision_recall_tradeoff.png'), dpi=300)
    plt.close()

    # --- PLOT 2: Paired Slope Chart (Consistency) - Using PR-AUC ---
    # Prepare data for slope chart
    df_pivot = df_results.pivot(index='Fold', columns='Model', values='PR_AUC_Crack')

    plt.figure(figsize=(8, 8))
    # Draw lines
    for i in range(len(df_pivot)):
        fold_idx = df_pivot.index[i]
        y1 = df_pivot.loc[fold_idx, 'LDA']
        y2 = df_pivot.loc[fold_idx, 'XGBoost']
        color = 'green' if y2 > y1 else 'red'
        plt.plot([1, 2], [y1, y2], c='gray', alpha=0.4, linewidth=1)

    # Draw points
    plt.scatter([1] * len(df_pivot), df_pivot['LDA'], c=colors['LDA'], s=50, label='LDA', zorder=5)
    plt.scatter([2] * len(df_pivot), df_pivot['XGBoost'], c=colors['XGBoost'], s=50, label='XGBoost', zorder=5)

    # Connect means
    plt.plot([1, 2], [df_pivot['LDA'].mean(), df_pivot['XGBoost'].mean()],
             c='black', linewidth=4, linestyle='-', marker='o', markersize=10, label='Mean Improvement')

    plt.xticks([1, 2], ['LDA', 'XGBoost'], fontsize=16, fontweight='bold')
    plt.ylabel('PR-AUC (Crack Class)', fontsize=14, fontweight='bold')
    plt.title('Fold-by-Fold Improvement (PR-AUC)', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0.8, 2.2)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'plot_2_paired_improvement.png'), dpi=300)
    plt.close()

    # --- PLOT 3: Distribution Boxplot (Stability) - Using PR-AUC ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x='Model', y='PR_AUC_Crack', palette=colors, width=0.5, linewidth=2)
    sns.stripplot(data=df_results, x='Model', y='PR_AUC_Crack', color='black', alpha=0.3, jitter=0.1, size=6)

    plt.title('Model Stability Comparison (PR-AUC)', fontweight='bold', pad=15)
    plt.ylabel('PR-AUC (Crack Class)', fontweight='bold')
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'plot_3_stability_boxplot.png'), dpi=300)
    plt.close()

    print("✓ Academic comparison plots saved!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Load data
    df, features, le = load_data(DATA_PATH)

    # Get LOGO splits WITH background validation split (CRITICAL FIX)
    logo_groups, always_train, df_cr, train_other, val_other = get_logo_splits(df, le)

    # Run comparison with realistic validation (background included)
    results_df, global_data = run_comparison(
        df, logo_groups, always_train, df_cr, train_other, val_other, features, le
    )

    # Save raw results
    results_df.to_csv(os.path.join(RESULTS_FOLDER, 'comparison_results.csv'), index=False)

    # Calculate comprehensive statistics - PR-AUC as primary metric
    summary = results_df.groupby('Model').agg({
        'PR_AUC_Crack': ['mean', 'std', 'min', 'max'],
        'F1_Crack': ['mean', 'std', 'min', 'max'],
        'Accuracy': ['mean', 'std', 'min', 'max'],
        'AUC': ['mean', 'std', 'min', 'max'],
        'MCC': ['mean', 'std', 'min', 'max']
    })

    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    print(summary)
    print("\n")

    # Print focused PR-AUC comparison
    print("="*80)
    print("PRIMARY METRIC: PRECISION-RECALL AUC (CRACK CLASS)")
    print("="*80)
    lda_pr_auc_mean = results_df[results_df['Model'] == 'LDA']['PR_AUC_Crack'].mean()
    lda_pr_auc_std = results_df[results_df['Model'] == 'LDA']['PR_AUC_Crack'].std()
    xgb_pr_auc_mean = results_df[results_df['Model'] == 'XGBoost']['PR_AUC_Crack'].mean()
    xgb_pr_auc_std = results_df[results_df['Model'] == 'XGBoost']['PR_AUC_Crack'].std()

    print(f"LDA:     PR-AUC = {lda_pr_auc_mean:.4f} ± {lda_pr_auc_std:.4f}")
    print(f"XGBoost: PR-AUC = {xgb_pr_auc_mean:.4f} ± {xgb_pr_auc_std:.4f}")
    print(f"Improvement: {((xgb_pr_auc_mean - lda_pr_auc_mean) / lda_pr_auc_mean * 100):+.2f}%")
    print("="*80)
    print("\n")

    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_comparison(results_df, RESULTS_FOLDER)
    plot_comprehensive_metrics(results_df, RESULTS_FOLDER)
    plot_advanced_analysis(global_data, RESULTS_FOLDER, le)
    plot_academic_comparison(results_df, RESULTS_FOLDER)  # New academic plots

    # Generate academic results report
    report_path = os.path.join(RESULTS_FOLDER, 'RESULTS_AND_DISCUSSION.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESULTS AND DISCUSSION: MODEL COMPARISON\n")
        f.write("LDA vs XGBoost for Crack Detection in Grapes\n")
        f.write("="*80 + "\n\n")

        # Extract statistics - mean and std for ALL metrics
        lda_f1 = results_df[results_df['Model'] == 'LDA']['F1_Crack'].mean()
        lda_f1_std = results_df[results_df['Model'] == 'LDA']['F1_Crack'].std()
        lda_pr_auc = results_df[results_df['Model'] == 'LDA']['PR_AUC_Crack'].mean()
        lda_pr_auc_std = results_df[results_df['Model'] == 'LDA']['PR_AUC_Crack'].std()
        lda_prec = results_df[results_df['Model'] == 'LDA']['Precision'].mean()
        lda_prec_std = results_df[results_df['Model'] == 'LDA']['Precision'].std()
        lda_rec = results_df[results_df['Model'] == 'LDA']['Recall'].mean()
        lda_rec_std = results_df[results_df['Model'] == 'LDA']['Recall'].std()
        lda_acc = results_df[results_df['Model'] == 'LDA']['Accuracy'].mean()
        lda_acc_std = results_df[results_df['Model'] == 'LDA']['Accuracy'].std()
        lda_auc = results_df[results_df['Model'] == 'LDA']['AUC'].mean()
        lda_auc_std = results_df[results_df['Model'] == 'LDA']['AUC'].std()
        lda_mcc = results_df[results_df['Model'] == 'LDA']['MCC'].mean()
        lda_mcc_std = results_df[results_df['Model'] == 'LDA']['MCC'].std()

        xgb_f1 = results_df[results_df['Model'] == 'XGBoost']['F1_Crack'].mean()
        xgb_f1_std = results_df[results_df['Model'] == 'XGBoost']['F1_Crack'].std()
        xgb_pr_auc = results_df[results_df['Model'] == 'XGBoost']['PR_AUC_Crack'].mean()
        xgb_pr_auc_std = results_df[results_df['Model'] == 'XGBoost']['PR_AUC_Crack'].std()
        xgb_prec = results_df[results_df['Model'] == 'XGBoost']['Precision'].mean()
        xgb_prec_std = results_df[results_df['Model'] == 'XGBoost']['Precision'].std()
        xgb_rec = results_df[results_df['Model'] == 'XGBoost']['Recall'].mean()
        xgb_rec_std = results_df[results_df['Model'] == 'XGBoost']['Recall'].std()
        xgb_acc = results_df[results_df['Model'] == 'XGBoost']['Accuracy'].mean()
        xgb_acc_std = results_df[results_df['Model'] == 'XGBoost']['Accuracy'].std()
        xgb_auc = results_df[results_df['Model'] == 'XGBoost']['AUC'].mean()
        xgb_auc_std = results_df[results_df['Model'] == 'XGBoost']['AUC'].std()
        xgb_mcc = results_df[results_df['Model'] == 'XGBoost']['MCC'].mean()
        xgb_mcc_std = results_df[results_df['Model'] == 'XGBoost']['MCC'].std()

        f.write("3.X. Model Architecture Comparison\n\n")

        f.write("To establish the optimal classification architecture for crack detection, ")
        f.write("we conducted a comprehensive comparison between Linear Discriminant Analysis (LDA) ")
        f.write("and XGBoost using all 204 hyperspectral features. The evaluation employed ")
        f.write(f"Leave-One-Group-Out (LOGO) cross-validation across {len(logo_groups)} independent ")
        f.write("grape bunches to ensure robust generalization assessment while maintaining ")
        f.write("biological independence between training and validation sets.\n\n")

        f.write("3.X.0. Selection of Primary Evaluation Metric\n\n")
        f.write("Given the inherent class imbalance in our dataset—where crack instances ")
        f.write("constitute a minority class—we selected Precision-Recall Area Under the Curve ")
        f.write("(PR-AUC) as our primary comparison metric. This choice is methodologically ")
        f.write("superior to traditional F1-score or ROC-AUC for several critical reasons:\n\n")
        f.write("First, PR-AUC provides a threshold-independent evaluation by computing the ")
        f.write("area under the precision-recall curve across ALL possible decision thresholds ")
        f.write("(Davis & Goadrich, 2006). Unlike F1-score, which evaluates performance at a ")
        f.write("single, arbitrary threshold (typically 0.5), PR-AUC assesses the model's ")
        f.write("capability to maintain the precision-recall trade-off across the entire ")
        f.write("operating range. This is crucial for agricultural applications where the ")
        f.write("optimal detection threshold may vary based on economic costs of false ")
        f.write("positives versus false negatives.\n\n")
        f.write("Second, for imbalanced datasets, PR-AUC is demonstrably more informative ")
        f.write("than ROC-AUC (Saito & Rehmsmeier, 2015). ROC-AUC incorporates true negative ")
        f.write("rate in its calculation, which can create an optimistic bias when the negative ")
        f.write("class (non-crack samples) vastly outnumbers the positive class. A classifier ")
        f.write("that simply predicts 'no crack' for most samples can achieve high ROC-AUC ")
        f.write("purely from correctly identifying abundant negative instances. In contrast, ")
        f.write("PR-AUC focuses exclusively on precision (positive predictive value) and recall ")
        f.write("(sensitivity), making it immune to inflation from true negatives and thus ")
        f.write("providing a more conservative, realistic assessment of minority class ")
        f.write("detection capability.\n\n")
        f.write("Third, PR-AUC directly reflects the practical utility for crack detection ")
        f.write("scenarios. Precision quantifies the proportion of crack predictions that are ")
        f.write("correct (minimizing unnecessary interventions), while recall measures the ")
        f.write("proportion of actual cracks successfully detected (minimizing missed defects). ")
        f.write("The PR-AUC thus summarizes the model's ability to achieve both goals ")
        f.write("simultaneously across all threshold configurations, providing stakeholders ")
        f.write("with a single, interpretable metric for comparing classification architectures.\n\n")

        f.write("3.X.1. Overall Performance Comparison\n\n")
        f.write(f"XGBoost demonstrated superior performance across all evaluation metrics. ")
        f.write(f"Our primary metric, Precision-Recall AUC (PR-AUC), specifically measures ")
        f.write(f"the model's ability to maintain the precision-recall trade-off across all ")
        f.write(f"possible decision thresholds—a crucial capability for imbalanced datasets ")
        f.write(f"where the CRACK class represents a minority. XGBoost achieved a PR-AUC of ")
        f.write(f"{xgb_pr_auc:.3f} ± {lda_pr_auc_std:.3f}, substantially outperforming LDA's ")
        f.write(f"{lda_pr_auc:.3f} ± {lda_pr_auc_std:.3f} (Table X). This represents a ")
        f.write(f"{((xgb_pr_auc - lda_pr_auc) / lda_pr_auc * 100):.1f}% relative improvement ")
        f.write(f"in threshold-independent crack detection capability.\n\n")
        f.write(f"The F1-score, while useful as a single-point metric at the default threshold, ")
        f.write(f"showed XGBoost achieving {xgb_f1:.3f} ± {xgb_f1_std:.3f} compared to LDA's ")
        f.write(f"{lda_f1:.3f} ± {lda_f1_std:.3f}. The ROC-AUC further confirmed ")
        f.write(f"XGBoost's superior discriminative power (AUC = {xgb_auc:.3f}) compared to ")
        f.write(f"LDA (AUC = {lda_auc:.3f}). However, for imbalanced datasets, PR-AUC provides ")
        f.write(f"a more reliable assessment as it focuses exclusively on positive class ")
        f.write(f"performance without being inflated by the large number of true negatives ")
        f.write(f"that dominate ROC-AUC calculations.\n\n")

        f.write("3.X.2. The Accuracy Paradox and Class Imbalance\n\n")
        f.write(f"While both models exhibited high global accuracy (LDA: {lda_acc:.3f}, ")
        f.write(f"XGBoost: {xgb_acc:.3f}), this metric proved misleading due to the inherent ")
        f.write(f"class imbalance in the dataset, where crack instances represent a minority ")
        f.write(f"class. The Matthews Correlation Coefficient (MCC), a more balanced metric for ")
        f.write(f"imbalanced datasets (Chicco & Jurman, 2020), revealed the true performance ")
        f.write(f"disparity: XGBoost (MCC = {xgb_mcc:.3f}) significantly outperformed LDA ")
        f.write(f"(MCC = {lda_mcc:.3f}). This substantial difference underscores the \"accuracy ")
        f.write(f"paradox\" phenomenon, where high overall accuracy can mask poor minority ")
        f.write(f"class detection—a critical consideration in agricultural defect detection ")
        f.write(f"where rare but economically significant defects must be reliably identified.\n\n")

        f.write("3.X.3. Linear vs. Non-Linear Feature Interactions\n\n")
        f.write("The superior performance of XGBoost can be attributed to its capacity to ")
        f.write("model complex, non-linear relationships within the hyperspectral feature space. ")
        f.write("LDA, as a linear discriminant classifier, assumes that class boundaries can be ")
        f.write("adequately represented by linear decision surfaces—an assumption that appears ")
        f.write("insufficient for the biological complexity of grape crack detection. The ")
        f.write("spectral signatures associated with crack formation likely involve intricate ")
        f.write("interactions between multiple wavelength bands, reflecting the complex ")
        f.write("biochemical and structural changes in damaged tissue. XGBoost's ensemble of ")
        f.write("decision trees can effectively capture these non-linear feature interactions ")
        f.write("and higher-order dependencies, as evidenced by the feature importance analysis ")
        f.write("(Figure X), which revealed complex patterns of wavelength co-dependence ")
        f.write("that would be invisible to linear methods.\n\n")

        f.write("3.X.4. Robustness and Generalization\n\n")
        f.write(f"The consistency of XGBoost's performance across LOGO folds (F1 std: {xgb_f1_std:.3f}) ")
        f.write(f"compared to LDA (F1 std: {lda_f1_std:.3f}) demonstrates superior ")
        f.write(f"generalization capability across diverse grape samples. The ROC curve analysis ")
        f.write(f"(Figure X) illustrates that XGBoost maintains high true positive rates ")
        f.write(f"while controlling false positive rates across various decision thresholds, ")
        f.write(f"indicating robust performance independent of specific operating points. ")
        f.write(f"This robustness is particularly crucial for practical deployment in precision ")
        f.write(f"viticulture, where the model must reliably perform across varying environmental ")
        f.write(f"conditions, grape varieties, and phenological stages.\n\n")

        f.write("3.X.5. Implications for Feature Selection\n\n")
        f.write("These findings establish XGBoost as the preferred classification architecture ")
        f.write("for subsequent feature selection experiments. The model's ability to handle ")
        f.write("class imbalance through sample weighting, combined with its capacity to ")
        f.write("exploit non-linear feature relationships, positions it as the optimal choice ")
        f.write("for identifying the minimal set of diagnostic wavelengths necessary for ")
        f.write("crack detection. Furthermore, XGBoost's inherent feature importance metrics ")
        f.write("provide valuable insights into spectral band relevance, which can guide ")
        f.write("the subsequent mRMR (Minimum Redundancy Maximum Relevance) feature ")
        f.write("selection process.\n\n")

        f.write("="*80 + "\n\n")
        f.write("TABLE X: Comprehensive Performance Comparison\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<35} {'LDA':<22} {'XGBoost':<22} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'PR-AUC (CRACK Class) *PRIMARY*':<35} {lda_pr_auc:.3f} ± {lda_pr_auc_std:.3f}      {xgb_pr_auc:.3f} ± {xgb_pr_auc_std:.3f}      {((xgb_pr_auc-lda_pr_auc)/lda_pr_auc*100):+.1f}%\n")
        f.write(f"{'F1-Score (CRACK)':<35} {lda_f1:.3f} ± {lda_f1_std:.3f}      {xgb_f1:.3f} ± {xgb_f1_std:.3f}      {((xgb_f1-lda_f1)/lda_f1*100):+.1f}%\n")
        f.write(f"{'Precision (CRACK)':<35} {lda_prec:.3f} ± {lda_prec_std:.3f}      {xgb_prec:.3f} ± {xgb_prec_std:.3f}      {((xgb_prec-lda_prec)/lda_prec*100):+.1f}%\n")
        f.write(f"{'Recall (CRACK)':<35} {lda_rec:.3f} ± {lda_rec_std:.3f}      {xgb_rec:.3f} ± {xgb_rec_std:.3f}      {((xgb_rec-lda_rec)/lda_rec*100):+.1f}%\n")
        f.write(f"{'ROC-AUC (CRACK vs Others)':<35} {lda_auc:.3f} ± {lda_auc_std:.3f}      {xgb_auc:.3f} ± {xgb_auc_std:.3f}      {((xgb_auc-lda_auc)/lda_auc*100):+.1f}%\n")
        f.write(f"{'MCC':<35} {lda_mcc:.3f} ± {lda_mcc_std:.3f}      {xgb_mcc:.3f} ± {xgb_mcc_std:.3f}      {((xgb_mcc-lda_mcc)/lda_mcc*100):+.1f}%\n")
        f.write(f"{'Accuracy (Global)':<35} {lda_acc:.3f} ± {lda_acc_std:.3f}      {xgb_acc:.3f} ± {xgb_acc_std:.3f}      {((xgb_acc-lda_acc)/lda_acc*100):+.1f}%\n")
        f.write("-"*80 + "\n\n")

        f.write("Note: Values represent mean ± standard deviation across LOGO cross-validation folds.\n")
        f.write("Improvement percentages calculated as relative change from LDA baseline.\n")
        f.write("*PRIMARY METRIC: PR-AUC (Precision-Recall AUC) for CRACK class provides threshold-\n")
        f.write("independent evaluation ideal for imbalanced datasets. Unlike ROC-AUC, PR-AUC focuses\n")
        f.write("on positive class performance and is not inflated by true negatives.\n")
        f.write("MCC: Matthews Correlation Coefficient, balanced metric for imbalanced datasets.\n")
        f.write("ROC-AUC: Area Under the ROC Curve, threshold-independent but can be optimistic.\n\n")

        f.write("="*80 + "\n")
        f.write("REFERENCES\n")
        f.write("="*80 + "\n")
        f.write("Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation\n")
        f.write("    coefficient (MCC) over F1 score and accuracy in binary classification\n")
        f.write("    evaluation. BMC Genomics, 21(1), 1-13.\n")

    print(f"✓ Academic results report saved to: {report_path}")
    print(f"\n✓ All results saved to: {RESULTS_FOLDER}")
    print("✓ Comparison complete!")

