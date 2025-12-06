import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, fbeta_score)
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import label_binarize
import joblib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Path to dataset
DATASET_PATH = r'C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv'
RESULTS_FOLDER = r'results_multi_class_reduce_F1_weighted'
MODEL_OUTPUT_PATH = os.path.join(RESULTS_FOLDER, 'lda_model_best_f1_weighted.joblib')
FEATURE_LOG_CSV = os.path.join(RESULTS_FOLDER, 'feature_selection_log_f1_weighted.csv')
BEST_MODEL_REPORT = os.path.join(RESULTS_FOLDER, 'best_model_report_f1_weighted.txt')
ACCURACY_PLOT = os.path.join(RESULTS_FOLDER, 'f1_weighted_plot.png')
METRICS_COMPARISON_PLOT = os.path.join(RESULTS_FOLDER, 'metrics_comparison_plot.png')
LABEL_COLUMN = 'label'  # Change this if your label column is named differently

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Print available columns for verification
print('Columns in CSV:', df.columns.tolist())

# Use all columns except label and non-feature columns as features
feature_columns = [col for col in df.columns if col not in [LABEL_COLUMN, 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']]
X = df[feature_columns].values
y = df[LABEL_COLUMN].values

# Print unique classes for verification
unique_classes = np.unique(y)
print('Unique classes in label column:', unique_classes)
if len(unique_classes) <= 1:
    raise ValueError(f'LDA requires at least 2 classes, found {len(unique_classes)}: {unique_classes}')

# Handle missing values if any
if np.any(pd.isnull(X)) or np.any(pd.isnull(y)):
    print('Warning: Missing values detected. Dropping rows with missing values.')
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

# Option to balance classes
BALANCE_CLASSES = True  # Set to True to balance classes using RandomOverSampler

if BALANCE_CLASSES:
    print('Balancing classes using RandomOverSampler...')
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    print('Class distribution after balancing:', dict(zip(*np.unique(y, return_counts=True))))
else:
    print('Class balancing is OFF. Using original class distribution.')
    print('Class distribution:', dict(zip(*np.unique(y, return_counts=True))))

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create results folder
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# OPTIMIZATION: Create a smaller subset for fast SFS feature selection
SELECTION_SAMPLE_SIZE = 200000 # Use full training set (no subsampling)
print('\n' + '='*70)
print('DATASET OPTIMIZATION')
print('='*70)
print(f'Full training set size: {len(X_train)} samples')
print(f'Full test set size: {len(X_test)} samples')

if len(X_train) > SELECTION_SAMPLE_SIZE:
    print(f'\nCreating stratified subsample of {SELECTION_SAMPLE_SIZE} samples for SFS feature selection...')
    from sklearn.model_selection import train_test_split as subsample_split
    X_train_sub, _, y_train_sub, _ = subsample_split(
        X_train, y_train,
        train_size=SELECTION_SAMPLE_SIZE,
        random_state=42,
        stratify=y_train
    )
    print(f'Subsample size: {len(X_train_sub)} samples')
    print(f'Subsample class distribution: {dict(zip(*np.unique(y_train_sub, return_counts=True)))}')
else:
    print(f'\nTraining set is smaller than {SELECTION_SAMPLE_SIZE}, using full set for SFS.')
    X_train_sub = X_train
    y_train_sub = y_train

# Feature Selection Experiment: Find optimal number of features (1-20)
print('\n' + '='*70)
print('COMPREHENSIVE BENCHMARKING: Feature Selection with ALL Metrics')
print('='*70)
print('Optimization Target: Weighted F1-Score')
print('Testing 1-20 features using SFS (Forward)')
print('Strategy: SFS on subset → Train & Evaluate on FULL data')
print('Metrics Tracked: Accuracy, Precision, Recall, F1, F2, ROC-AUC (Global + CRACK)')
print('='*70 + '\n')

best_f1_score = 0
best_n_features = 0
best_model = None
best_features = []
best_y_pred = None
best_y_test = None
best_accuracy = 0
best_all_metrics = {}

results = []

# Loop through different numbers of features
for n_features in tqdm(range(1, 21), desc="Testing feature counts"):
    print(f'\n{"="*70}')
    print(f'Processing k={n_features}...')
    print(f'{"="*70}')

    # Create LDA estimator for SFS
    lda_estimator = LinearDiscriminantAnalysis()
    
    # Perform SFS to select top k features (using SUBSET for speed)
    print(f'  Running SFS on {len(X_train_sub)} samples...')
    sfs = SequentialFeatureSelector(
        estimator=lda_estimator,
        n_features_to_select=n_features,
        direction='forward',
        scoring='f1_weighted',  # Optimize for weighted F1-score
        n_jobs=-1,  # Use all CPU cores for parallel processing
        cv=3  # Use 3-fold cross-validation for scoring
    )
    print(f'  SFS will perform forward selection with 3-fold CV (F1-weighted)...')
    sfs.fit(X_train_sub, y_train_sub)
    print(f'  ✓ SFS completed!')

    # Get selected features
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if sfs.support_[i]]
    print(f'  ✓ Selected features: {selected_features}')

    # Transform FULL data to use only selected features
    print(f'  Training LDA on FULL training set ({len(X_train)} samples)...')
    X_train_selected = sfs.transform(X_train)
    X_test_selected = sfs.transform(X_test)

    # Train a new LDA model on FULL selected features
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_selected, y_train)
    
    # Evaluate on FULL test set
    print(f'  Evaluating on test set ({len(X_test)} samples)...')
    y_pred = lda.predict(X_test_selected)
    y_pred_proba = lda.predict_proba(X_test_selected)

    # === GLOBAL METRICS (Weighted) ===
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (One-vs-Rest for multi-class)
    try:
        y_test_bin = label_binarize(y_test, classes=lda.classes_)
        roc_auc_weighted = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
    except Exception as e:
        roc_auc_weighted = 0.0
        print(f'  ⚠ Warning: Could not calculate ROC-AUC: {str(e)}')

    # === CRACK-SPECIFIC METRICS ===
    try:
        crack_idx = np.where(lda.classes_ == 'CRACK')[0][0]

        # Per-class metrics (for CRACK)
        precision_per_class = precision_score(y_test, y_pred, average=None, labels=lda.classes_, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, labels=lda.classes_, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=lda.classes_, zero_division=0)
        f2_per_class = fbeta_score(y_test, y_pred, beta=2, average=None, labels=lda.classes_, zero_division=0)

        crack_precision = precision_per_class[crack_idx]
        crack_recall = recall_per_class[crack_idx]
        crack_f1 = f1_per_class[crack_idx]
        crack_f2 = f2_per_class[crack_idx]

        # CRACK ROC-AUC (One-vs-Rest)
        try:
            crack_auc = roc_auc_score((y_test == 'CRACK').astype(int), y_pred_proba[:, crack_idx])
        except:
            crack_auc = 0.0

    except (IndexError, ValueError) as e:
        print(f'  ⚠ WARNING: CRACK class not found or metrics calculation failed!')
        crack_precision = crack_recall = crack_f1 = crack_f2 = crack_auc = 0.0

    # Store comprehensive results
    results.append({
        'num_features': n_features,
        # Global Weighted Metrics
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'roc_auc_weighted': roc_auc_weighted,
        # CRACK-Specific Metrics
        'crack_precision': crack_precision,
        'crack_recall': crack_recall,
        'crack_f1': crack_f1,
        'crack_f2': crack_f2,
        'crack_auc': crack_auc,
        'selected_features': str(selected_features)
    })
    
    # Print progress with ALL metrics (matching format)
    print(f'  ✓ RESULT: k={n_features}')
    print(f'    Global: Acc={accuracy:.4f}, Prec={precision_weighted:.4f}, Rec={recall_weighted:.4f}, F1={f1_weighted:.4f}, AUC={roc_auc_weighted:.4f}')
    print(f'    CRACK:  Acc=N/A   , Prec={crack_precision:.4f}, Rec={crack_recall:.4f}, F1={crack_f1:.4f}, AUC={crack_auc:.4f}')

    # Track best model (based on Weighted F1-Score)
    if f1_weighted > best_f1_score:
        best_f1_score = f1_weighted
        best_n_features = n_features
        best_model = (sfs, lda)
        best_features = selected_features
        best_y_pred = y_pred
        best_y_test = y_test
        best_accuracy = accuracy
        best_all_metrics = {
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'roc_auc_weighted': roc_auc_weighted,
            'crack_precision': crack_precision,
            'crack_recall': crack_recall,
            'crack_f1': crack_f1,
            'crack_f2': crack_f2,
            'crack_auc': crack_auc
        }
        print(f'  🌟 NEW BEST MODEL! F1-Weighted={f1_weighted:.4f}')

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(FEATURE_LOG_CSV, index=False)
print(f'\n✓ Results logged to CSV: {FEATURE_LOG_CSV}')

# Print summary of all results
print('\n' + '='*90)
print('EXPERIMENT SUMMARY - ALL METRICS')
print('='*90)
print(f"{'k':<4} {'F1':<7} {'Acc':<7} {'Prec':<7} {'Rec':<7} {'AUC':<7} | {'C_F1':<7} {'C_F2':<7} {'C_Rec':<7}")
print('-'*90)
for result in results:
    print(f"{result['num_features']:<4} "
          f"{result['f1_weighted']:<7.4f} "
          f"{result['accuracy']:<7.4f} "
          f"{result['precision_weighted']:<7.4f} "
          f"{result['recall_weighted']:<7.4f} "
          f"{result['roc_auc_weighted']:<7.4f} | "
          f"{result['crack_f1']:<7.4f} "
          f"{result['crack_f2']:<7.4f} "
          f"{result['crack_recall']:<7.4f}")

# Print best model details
print('\n' + '='*70)
print('BEST MODEL FOUND (Optimized for Weighted F1-Score)')
print('='*70)
print(f'🎯 Best Weighted F1-Score: {best_f1_score:.4f} ⭐')
print(f'Optimal number of features: {best_n_features}')
print(f'\nGLOBAL METRICS (Weighted):')
print(f'  Accuracy:           {best_accuracy:.4f}')
print(f'  Precision:          {best_all_metrics.get("precision_weighted", 0):.4f}')
print(f'  Recall:             {best_all_metrics.get("recall_weighted", 0):.4f}')
print(f'  ROC-AUC:            {best_all_metrics.get("roc_auc_weighted", 0):.4f}')
print(f'\nCRACK-SPECIFIC METRICS:')
print(f'  Precision:          {best_all_metrics.get("crack_precision", 0):.4f}')
print(f'  Recall:             {best_all_metrics.get("crack_recall", 0):.4f}')
print(f'  F1-Score:           {best_all_metrics.get("crack_f1", 0):.4f}')
print(f'  F2-Score:           {best_all_metrics.get("crack_f2", 0):.4f}')
print(f'  ROC-AUC:            {best_all_metrics.get("crack_auc", 0):.4f}')
print(f'\nSelected features: {best_features}')
print('\nClassification Report (Best Model):')
best_class_report = classification_report(best_y_test, best_y_pred)
print(best_class_report)
print('\nConfusion Matrix (Best Model):')
best_conf_matrix = confusion_matrix(best_y_test, best_y_pred)
print(best_conf_matrix)

# Save the best model (both SFS and LDA)
model_package = {
    'sfs': best_model[0],
    'lda': best_model[1],
    'selected_features': best_features,
    'n_features': best_n_features,
    'f1_score_weighted': best_f1_score,
    'accuracy': best_accuracy,
    'feature_columns': feature_columns,
    'optimization_metric': 'Weighted F1-Score'
}
joblib.dump(model_package, MODEL_OUTPUT_PATH)
print(f'\n✓ Best LDA model (F1-weighted optimized) saved to: {MODEL_OUTPUT_PATH}')

# Save detailed report for the best model
with open(BEST_MODEL_REPORT, 'w', encoding='utf-8') as f:
    f.write('='*70 + '\n')
    f.write('BEST MODEL REPORT - Weighted F1-Score Optimization\n')
    f.write('='*70 + '\n\n')
    f.write(f'Date: 2025-12-02\n')
    f.write(f'Dataset: {DATASET_PATH}\n')
    f.write(f'Total features available: {len(feature_columns)}\n')
    f.write(f'Feature range tested: 1-20\n')
    f.write(f'Selection method: Sequential Feature Selection (Forward)\n')
    f.write(f'SFS Scoring: f1_weighted (with 3-fold CV)\n')
    f.write(f'Optimization metric: Weighted F1-Score\n')
    f.write(f'Note: F1-weighted balances precision and recall across all classes\n')
    f.write(f'Test set size: {len(best_y_test)} samples\n\n')

    f.write('='*70 + '\n')
    f.write('BEST MODEL DETAILS\n')
    f.write('='*70 + '\n')
    f.write(f'*** This model was selected to MAXIMIZE WEIGHTED F1-SCORE ***\n\n')
    f.write(f'Optimal number of features: {best_n_features}\n')
    f.write(f'Weighted F1-Score: {best_f1_score:.4f} ⭐\n')
    f.write(f'Accuracy: {best_accuracy:.4f}\n\n')
    f.write(f'Selected features:\n')
    for i, feat in enumerate(best_features, 1):
        f.write(f'  {i}. {feat}\n')

    f.write('\n' + '='*70 + '\n')
    f.write('CLASSIFICATION REPORT\n')
    f.write('='*70 + '\n')
    f.write(best_class_report)

    f.write('\n' + '='*70 + '\n')
    f.write('CONFUSION MATRIX\n')
    f.write('='*70 + '\n')
    f.write('Rows: True labels | Columns: Predicted labels\n\n')
    f.write(str(best_conf_matrix))
    f.write('\n')

print(f'✓ Best model report saved to: {BEST_MODEL_REPORT}')

# Create visualization: Number of Features vs Weighted F1-Score
plt.figure(figsize=(12, 6))
num_features_list = [r['num_features'] for r in results]
f1_list = [r['f1_score_weighted'] for r in results]
accuracy_list = [r['accuracy'] for r in results]

# Plot F1-weighted as primary metric (bold line)
plt.plot(num_features_list, f1_list, marker='D', linewidth=3, markersize=10,
         label='F1-Score (weighted)', color='#2ca02c', zorder=3)
plt.plot(num_features_list, accuracy_list, marker='o', linewidth=2, markersize=8,
         label='Accuracy', alpha=0.6)

# Mark the best point (based on F1-weighted)
plt.scatter([best_n_features], [best_f1_score], color='gold', s=300, marker='*',
            edgecolors='black', linewidths=2,
            label=f'Best (k={best_n_features}, F1={best_f1_score:.4f})', zorder=5)

plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Feature Selection Optimization: Weighted F1-Score vs Number of Features', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='best')
plt.xticks(range(1, 21))
plt.ylim([min(min(f1_list), min(accuracy_list)) - 0.02,
          max(max(f1_list), max(accuracy_list)) + 0.02])
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=300, bbox_inches='tight')
plt.close()

print(f'✓ Accuracy plot saved to: {ACCURACY_PLOT}')

# Create comprehensive metrics comparison plot
print(f'Creating comprehensive metrics comparison plot...')
fig, ax = plt.subplots(figsize=(14, 8))

# Extract all metrics
num_features_list = [r['num_features'] for r in results]
# Global metrics
accuracy_list = [r['accuracy'] for r in results]
f1_weighted_list = [r['f1_weighted'] for r in results]
recall_weighted_list = [r['recall_weighted'] for r in results]
precision_weighted_list = [r['precision_weighted'] for r in results]
roc_auc_weighted_list = [r['roc_auc_weighted'] for r in results]
# CRACK metrics
crack_precision_list = [r['crack_precision'] for r in results]
crack_recall_list = [r['crack_recall'] for r in results]
crack_f1_list = [r['crack_f1'] for r in results]
crack_f2_list = [r['crack_f2'] for r in results]
crack_auc_list = [r['crack_auc'] for r in results]

# Plot GLOBAL metrics with solid lines
ax.plot(num_features_list, accuracy_list,
        marker='o', linestyle='-', linewidth=2, markersize=6,
        label='Global: Accuracy', color='gray', alpha=0.8)

ax.plot(num_features_list, f1_weighted_list,
        marker='D', linewidth=4, markersize=10,
        label='Global: F1-Weighted (Main)', color='#1f77b4', zorder=4)

ax.plot(num_features_list, recall_weighted_list,
        marker='s', linewidth=2.5, markersize=7,
        label='Global: Recall', color='#2ca02c', zorder=3)

ax.plot(num_features_list, precision_weighted_list,
        marker='^', linewidth=2.5, markersize=7,
        label='Global: Precision', color='#9467bd', zorder=3)

ax.plot(num_features_list, roc_auc_weighted_list,
        marker='p', linewidth=2.5, markersize=7,
        label='Global: ROC-AUC', color='#8c564b', zorder=3)

# Plot CRACK metrics with dashed lines
ax.plot(num_features_list, crack_precision_list,
        marker='>', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: Precision', color='#e377c2', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_recall_list,
        marker='v', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: Recall', color='#d62728', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_f1_list,
        marker='<', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: F1', color='#ff7f0e', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_f2_list,
        marker='*', linestyle='--', linewidth=2, markersize=8,
        label='CRACK: F2', color='#bcbd22', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_auc_list,
        marker='h', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: ROC-AUC', color='#17becf', zorder=3, alpha=0.8)

# Mark the best F1 point
ax.scatter([best_n_features], [best_f1_score],
          color='gold', s=400, marker='*',
          edgecolors='black', linewidths=2.5,
          label=f'Best F1 (k={best_n_features})', zorder=5)

# Formatting
ax.set_xlabel('Number of Features (k)', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Comprehensive Benchmarking: Global & CRACK Metrics (F1-Weighted Optimization)',
            fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(fontsize=9, loc='best', framealpha=0.95, shadow=True, ncol=2)
ax.set_xticks(range(1, 21))
ax.set_ylim([0, 1.05])

# Add text box with best model info
textstr = f'Best Model (k={best_n_features}):\n'
textstr += f'Global F1: {best_f1_score:.4f}\n'
textstr += f'CRACK F1: {best_all_metrics.get("crack_f1", 0):.4f}\n'
textstr += f'CRACK F2: {best_all_metrics.get("crack_f2", 0):.4f}'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig(METRICS_COMPARISON_PLOT, dpi=300, bbox_inches='tight')
plt.close()

print(f'✓ Comprehensive metrics plot saved to: {METRICS_COMPARISON_PLOT}')

print('\n' + '='*70)
print('EXPERIMENT COMPLETE')
print('='*70)
print(f'All results saved to folder: {RESULTS_FOLDER}')
print(f'\nFiles generated:')
print(f'  1. {FEATURE_LOG_CSV} - Detailed metrics for each feature count (ALL METRICS)')
print(f'  2. {MODEL_OUTPUT_PATH} - Best model package')
print(f'  3. {BEST_MODEL_REPORT} - Detailed report for best model')
print(f'  4. {ACCURACY_PLOT} - F1-Weighted performance visualization')
print(f'  5. {METRICS_COMPARISON_PLOT} - Comprehensive metrics comparison (ALL METRICS)')
print(f'\nTo use the model:')
print(f'  model_pkg = joblib.load("{MODEL_OUTPUT_PATH}")')
print(f'  X_selected = model_pkg["sfs"].transform(X)')
print(f'  pred = model_pkg["lda"].predict(X_selected)')

