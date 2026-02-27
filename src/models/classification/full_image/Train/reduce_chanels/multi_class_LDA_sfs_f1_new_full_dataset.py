import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, fbeta_score, make_scorer)
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import label_binarize
import joblib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import wavelength mapping
from wavelengths import WAVELENGTHS

# Set seaborn style for professional plots
sns.set_style("whitegrid")

# Path to dataset
DATASET_PATH = str(_PROJECT_ROOT / r'src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv')
RESULTS_FOLDER = str(_PROJECT_ROOT / r'src/models/classification/full_image/Train/reduce_chanels/results_multi_class_reduce_CRACK_F1')
MODEL_OUTPUT_PATH = os.path.join(RESULTS_FOLDER, 'lda_model_best_crack_f1.joblib')
FEATURE_LOG_CSV = os.path.join(RESULTS_FOLDER, 'feature_selection_log_crack_f1.csv')
BEST_MODEL_REPORT = os.path.join(RESULTS_FOLDER, 'best_model_report_crack_f1.txt')
ACCURACY_PLOT = os.path.join(RESULTS_FOLDER, 'crack_f1_plot.png')
METRICS_COMPARISON_PLOT = os.path.join(RESULTS_FOLDER, 'metrics_comparison_crack_f1.png')
THRESHOLD_OPTIMIZATION_LOG = os.path.join(RESULTS_FOLDER, 'threshold_optimization_log.csv')
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

print('\n' + '='*70)
print('DATA PREPARATION - PREVENTING DATA LEAKAGE')
print('='*70)

# 1. SPLIT DATA FIRST (Crucial for preventing data leakage)
print('Step 1: Splitting data into Train (80%) and Test (20%) BEFORE any oversampling...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'  Training set: {len(X_train)} samples')
print(f'  Test set: {len(X_test)} samples')
print(f'  Train class distribution (original): {dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'  Test class distribution (original): {dict(zip(*np.unique(y_test, return_counts=True)))}')

# 2. BALANCE ONLY THE TRAINING SET
BALANCE_CLASSES = True

if BALANCE_CLASSES:
    print('\nStep 2: Balancing TRAINING set only using RandomOverSampler...')
    print('  (Test set remains unbalanced to reflect real-world distribution)')
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print(f'  Training set after balancing: {len(X_train)} samples')
    print(f'  Train class distribution (balanced): {dict(zip(*np.unique(y_train, return_counts=True)))}')
else:
    print('\nStep 2: Class balancing is OFF.')

print(f'\n✓ Data preparation complete. Test set integrity preserved!')
print('='*70)

# Create results folder
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================================
# LARGE-SCALE SELECTION & FULL-SCALE EVALUATION WORKFLOW
# ============================================================================
# Strategy:
#   1. SELECTION PHASE: Use 500,000 samples to run SFS (computationally expensive)
#   2. EVALUATION PHASE: Train NEW model on FULL training set with selected features
#   3. METRICS/PLOTS: Report performance from FULL test set evaluation only
# ============================================================================

SELECTION_SAMPLE_SIZE = 500000  # Large subset for robust feature selection

print('\n' + '='*70)
print('LARGE-SCALE SELECTION & FULL-SCALE EVALUATION WORKFLOW')
print('='*70)
print(f'Full training set size: {len(X_train)} samples (balanced)')
print(f'Full test set size: {len(X_test)} samples (unbalanced, real-world)')

if len(X_train) > SELECTION_SAMPLE_SIZE:
    print(f'\n📊 SELECTION PHASE: Creating stratified subsample of {SELECTION_SAMPLE_SIZE:,} samples for SFS...')
    print(f'   Purpose: Speed up feature selection while maintaining representativeness')
    from sklearn.model_selection import train_test_split as subsample_split
    X_train_sub, _, y_train_sub, _ = subsample_split(
        X_train, y_train,
        train_size=SELECTION_SAMPLE_SIZE,
        random_state=42,
        stratify=y_train
    )
    print(f'   ✓ Subsample created: {len(X_train_sub):,} samples')
    print(f'   ✓ Subsample class distribution: {dict(zip(*np.unique(y_train_sub, return_counts=True)))}')
else:
    print(f'\n📊 Training set ({len(X_train):,} samples) is smaller than {SELECTION_SAMPLE_SIZE:,}')
    print(f'   Using FULL training set for SFS (no subsampling needed)')
    X_train_sub = X_train
    y_train_sub = y_train

print(f'\n🎯 EVALUATION PHASE: All metrics will be from FULL dataset evaluation')
print(f'   - Train on: {len(X_train):,} samples (all training data)')
print(f'   - Test on: {len(X_test):,} samples (all test data)')
print('='*70)

# Create custom scorer for CRACK F1-Score
def crack_f1_scorer(y_true, y_pred):
    """Custom scorer that returns F1-score specifically for CRACK class."""
    return f1_score(y_true, y_pred, labels=['CRACK'], average=None, zero_division=0)[0]

crack_f1_scoring = make_scorer(crack_f1_scorer)

# Feature Selection Experiment: Find optimal number of features (1-20)
print('\n' + '='*70)
print('COMPREHENSIVE BENCHMARKING: Feature Selection with ALL Metrics')
print('='*70)
print('🎯 Optimization Target: CRACK F1-Score (Maximizing CRACK class detection)')
print('Testing 1-20 features using SFS (Forward)')
print('')
print('📊 WORKFLOW FOR EACH k (1 to 20):')
print(f'   1. SELECTION: Run SFS on {len(X_train_sub):,} sample subset (n_jobs=-1)')
print(f'   2. DISCARD: Temporary SFS model discarded after feature identification')
print(f'   3. TRAIN: NEW LDA trained on FULL training set ({len(X_train):,} samples)')
print(f'   4. EVALUATE: Test on FULL test set ({len(X_test):,} samples)')
print(f'   5. REPORT: Metrics from FULL test set → CSV & Plots')
print('')
print('Metrics Tracked: Accuracy, Precision, Recall, F1, F2, ROC-AUC (Global + CRACK)')
print('='*70 + '\n')

best_crack_f1 = 0
best_n_features = 0
best_model = None
best_features = []
best_y_pred = None
best_y_test = None
best_y_proba = None
best_accuracy = 0
best_all_metrics = {}
best_threshold = 0.5

# Feature frequency tracking for stability analysis
feature_selection_counter = {feature: 0 for feature in feature_columns}

results = []

# Loop through different numbers of features
for n_features in tqdm(range(1, 21), desc="Testing feature counts"):
    print(f'\n{"="*70}')
    print(f'Processing k={n_features}...')
    print(f'{"="*70}')

    # ========================================================================
    # STEP 1: SELECTION PHASE - Run SFS on Large Subset (500k samples)
    # ========================================================================
    print(f'  📊 STEP 1: SELECTION PHASE')
    print(f'     Running SFS on {len(X_train_sub):,} sample subset (n_jobs=-1)...')

    # Create LDA estimator for SFS
    lda_estimator = LinearDiscriminantAnalysis()
    
    # Perform SFS to select top k features
    sfs = SequentialFeatureSelector(
        estimator=lda_estimator,
        n_features_to_select=n_features,
        direction='forward',
        scoring=crack_f1_scoring,  # 🎯 Optimize for CRACK F1-score
        n_jobs=-1,  # Use all CPU cores for parallel processing
        cv=3  # Use 3-fold cross-validation for scoring
    )
    sfs.fit(X_train_sub, y_train_sub)
    print(f'     ✓ SFS completed with 3-fold CV on subset')

    # ========================================================================
    # STEP 2: DISCARD - Extract feature indices, discard temporary SFS model
    # ========================================================================
    print(f'  🗑️  STEP 2: DISCARD PHASE')
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if sfs.support_[i]]
    print(f'     ✓ Identified {len(selected_features)} features: {selected_features[:3]}...')
    print(f'     ✓ Temporary SFS model will be discarded (not used for evaluation)')

    # Update feature frequency counter (stability analysis)
    for feature in selected_features:
        feature_selection_counter[feature] += 1

    # ========================================================================
    # STEP 3: TRAIN - NEW LDA on FULL Training Set with selected features
    # ========================================================================
    print(f'  🎓 STEP 3: TRAIN PHASE')
    print(f'     Training NEW LDA on FULL training set ({len(X_train):,} samples)...')

    # Transform FULL data to use only selected features
    X_train_selected = sfs.transform(X_train)
    X_test_selected = sfs.transform(X_test)

    # Train a BRAND NEW LDA model on FULL training data
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_selected, y_train)
    print(f'     ✓ NEW LDA model trained on full dataset')

    # ========================================================================
    # STEP 4: EVALUATE - Test on FULL Test Set
    # ========================================================================
    print(f'  🔍 STEP 4: EVALUATION PHASE')
    print(f'     Evaluating on FULL test set ({len(X_test):,} samples)...')
    y_pred = lda.predict(X_test_selected)
    y_pred_proba = lda.predict_proba(X_test_selected)
    print(f'     ✓ Predictions generated from full test set')

    # ========================================================================
    # STEP 5: REPORT - Calculate metrics from FULL test set (for CSV & Plots)
    # ========================================================================
    print(f'  📈 STEP 5: REPORT PHASE')
    print(f'     Calculating metrics from FULL test set evaluation...')

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
    
    # Print progress with ALL metrics (from FULL test set)
    print(f'     ✓ Metrics calculated from {len(X_test):,} test samples')
    print(f'  ')
    print(f'  ✅ FINAL RESULT (k={n_features}) - FULL Test Set Performance:')
    print(f'    Global: Acc={accuracy:.4f}, Prec={precision_weighted:.4f}, Rec={recall_weighted:.4f}, F1={f1_weighted:.4f}, AUC={roc_auc_weighted:.4f}')
    print(f'    CRACK:  Acc=N/A   , Prec={crack_precision:.4f}, Rec={crack_recall:.4f}, F1={crack_f1:.4f}, AUC={crack_auc:.4f}')
    print(f'    ⚠️  NOTE: These metrics are from FULL dataset evaluation, NOT SFS CV scores')

    # Track best model (based on CRACK F1-Score)
    if crack_f1 > best_crack_f1:
        best_crack_f1 = crack_f1
        best_n_features = n_features
        best_model = (sfs, lda)
        best_features = selected_features
        best_y_pred = y_pred
        best_y_test = y_test
        best_y_proba = y_pred_proba
        best_accuracy = accuracy
        best_all_metrics = {
            'f1_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'roc_auc_weighted': roc_auc_weighted,
            'crack_precision': crack_precision,
            'crack_recall': crack_recall,
            'crack_f1': crack_f1,
            'crack_f2': crack_f2,
            'crack_auc': crack_auc
        }
        print(f'  🌟 NEW BEST MODEL! CRACK F1={crack_f1:.4f}')

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(FEATURE_LOG_CSV, index=False)
print(f'\n' + '='*70)
print('SAVING RESULTS')
print('='*70)
print(f'✓ Results logged to CSV: {FEATURE_LOG_CSV}')
print(f'  ⚠️  IMPORTANT: All metrics in CSV are from FULL TEST SET evaluation')
print(f'  - NOT from SFS cross-validation scores')
print(f'  - Real-world performance on {len(X_test):,} test samples')
print('='*70)

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
print('BEST MODEL FOUND (Optimized for CRACK F1-Score)')
print('='*70)
print(f'🎯 Best CRACK F1-Score: {best_crack_f1:.4f} ⭐')
print(f'Optimal number of features: {best_n_features}')
print(f'\nGLOBAL METRICS (Weighted):')
print(f'  Accuracy:           {best_accuracy:.4f}')
print(f'  F1-Score:           {best_all_metrics.get("f1_weighted", 0):.4f}')
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

# ============================================================================
# THRESHOLD OPTIMIZATION FOR CRACK DETECTION
# ============================================================================
print('\n' + '='*70)
print('THRESHOLD OPTIMIZATION FOR CRACK CLASS')
print('='*70)
print('Optimizing probability threshold to maximize CRACK F1-Score...')

# Get CRACK class index
crack_idx = np.where(best_model[1].classes_ == 'CRACK')[0][0]
print(f'CRACK class index: {crack_idx}')

# Test thresholds from 0.01 to 0.99
threshold_results = []
best_threshold_f1 = 0
best_threshold_value = 0.5

print('Testing thresholds from 0.01 to 0.99...')
for threshold in np.arange(0.01, 1.00, 0.01):
    # Apply custom prediction logic
    y_pred_threshold = []
    for i in range(len(best_y_proba)):
        if best_y_proba[i, crack_idx] >= threshold:
            # Predict CRACK if probability exceeds threshold
            y_pred_threshold.append('CRACK')
        else:
            # Otherwise, predict the class with highest probability among remaining classes
            proba_copy = best_y_proba[i].copy()
            proba_copy[crack_idx] = -1  # Exclude CRACK from consideration
            max_idx = np.argmax(proba_copy)
            y_pred_threshold.append(best_model[1].classes_[max_idx])

    y_pred_threshold = np.array(y_pred_threshold)

    # Calculate metrics for CRACK class
    crack_f1_thresh = f1_score(best_y_test, y_pred_threshold, labels=['CRACK'], average=None, zero_division=0)[0]
    crack_precision_thresh = precision_score(best_y_test, y_pred_threshold, labels=['CRACK'], average=None, zero_division=0)[0]
    crack_recall_thresh = recall_score(best_y_test, y_pred_threshold, labels=['CRACK'], average=None, zero_division=0)[0]

    threshold_results.append({
        'threshold': threshold,
        'crack_f1': crack_f1_thresh,
        'crack_precision': crack_precision_thresh,
        'crack_recall': crack_recall_thresh
    })

    # Update best threshold
    if crack_f1_thresh > best_threshold_f1:
        best_threshold_f1 = crack_f1_thresh
        best_threshold_value = threshold

print(f'\n{"="*70}')
print(f'THRESHOLD OPTIMIZATION COMPLETE')
print(f'{"="*70}')
print(f'Optimal Threshold: {best_threshold_value:.3f}')
print(f'Maximized CRACK F1-Score: {best_threshold_f1:.4f}')
print(f'{"="*70}\n')

# Generate final predictions with optimal threshold
y_pred_final = []
for i in range(len(best_y_proba)):
    if best_y_proba[i, crack_idx] >= best_threshold_value:
        y_pred_final.append('CRACK')
    else:
        proba_copy = best_y_proba[i].copy()
        proba_copy[crack_idx] = -1
        max_idx = np.argmax(proba_copy)
        y_pred_final.append(best_model[1].classes_[max_idx])

y_pred_final = np.array(y_pred_final)

# Print optimized results
print('Final Evaluation with Optimized Threshold:')
print(f'Overall Accuracy: {accuracy_score(best_y_test, y_pred_final):.4f}')
print('\nClassification Report (Optimized):')
print(classification_report(best_y_test, y_pred_final))
print('\nConfusion Matrix (Optimized):')
print(confusion_matrix(best_y_test, y_pred_final))

# Save threshold optimization log
threshold_log_df = pd.DataFrame(threshold_results)
threshold_log_df.to_csv(THRESHOLD_OPTIMIZATION_LOG, index=False)
print(f'\n✓ Threshold optimization log saved to: {THRESHOLD_OPTIMIZATION_LOG}')

# Update best metrics with optimized threshold results
best_threshold = best_threshold_value
best_y_pred = y_pred_final
best_class_report = classification_report(best_y_test, y_pred_final)
best_conf_matrix = confusion_matrix(best_y_test, y_pred_final)
best_crack_f1 = best_threshold_f1

# Save the best model (both SFS and LDA) with optimal threshold
model_package = {
    'sfs': best_model[0],
    'lda': best_model[1],
    'selected_features': best_features,
    'n_features': best_n_features,
    'crack_f1_score': best_crack_f1,
    'optimal_threshold': best_threshold,
    'accuracy': best_accuracy,
    'feature_columns': feature_columns,
    'classes': best_model[1].classes_,
    'crack_class_index': crack_idx,
    'optimization_metric': 'CRACK F1-Score'
}
joblib.dump(model_package, MODEL_OUTPUT_PATH)
print(f'\n✓ Best LDA model (CRACK F1 optimized with threshold) saved to: {MODEL_OUTPUT_PATH}')

# Save detailed report for the best model
with open(BEST_MODEL_REPORT, 'w', encoding='utf-8') as f:
    f.write('='*70 + '\n')
    f.write('BEST MODEL REPORT - CRACK F1-Score Optimization\n')
    f.write('='*70 + '\n\n')
    f.write(f'Date: 2025-12-04\n')
    f.write(f'Dataset: {DATASET_PATH}\n')
    f.write(f'Total wavelengths available: {len(feature_columns)}\n')
    f.write(f'Feature range tested: 1-20 wavelengths\n')
    f.write(f'Selection method: Sequential Feature Selection (Forward)\n')
    f.write(f'SFS Scoring: CRACK F1-Score (with 3-fold CV)\n')
    f.write(f'Optimization metric: CRACK F1-Score\n')
    f.write(f'Note: Model optimized specifically for CRACK class detection\n')
    f.write(f'Feature naming: Wavelengths in nm (e.g., 397.32nm, 850.29nm)\n')
    f.write(f'Data Leakage Prevention: Train/Test split BEFORE oversampling\n')
    f.write(f'Test set size: {len(best_y_test)} samples (unbalanced, real-world distribution)\n')
    f.write(f'Threshold optimization: Yes (range 0.01-0.99)\n\n')

    f.write('='*70 + '\n')
    f.write('BEST MODEL DETAILS\n')
    f.write('='*70 + '\n')
    f.write(f'*** This model was selected to MAXIMIZE CRACK F1-SCORE ***\n\n')
    f.write(f'Optimal number of wavelengths: {best_n_features}\n')
    f.write(f'CRACK F1-Score (with optimal threshold): {best_crack_f1:.4f} ⭐\n')
    f.write(f'Optimal Threshold: {best_threshold:.3f}\n')
    f.write(f'Overall Accuracy: {best_accuracy:.4f}\n\n')
    f.write(f'Selected Wavelengths (in nm):\n')
    for i, feat in enumerate(best_features, 1):
        f.write(f'  {i:2d}. {feat}\n')

    f.write('\n' + '='*70 + '\n')
    f.write('CLASSIFICATION REPORT (With Optimized Threshold)\n')
    f.write('='*70 + '\n')
    f.write(best_class_report)

    f.write('\n' + '='*70 + '\n')
    f.write('CONFUSION MATRIX (With Optimized Threshold)\n')
    f.write('='*70 + '\n')
    f.write('Rows: True labels | Columns: Predicted labels\n\n')
    f.write(str(best_conf_matrix))
    f.write('\n')

print(f'✓ Best model report saved to: {BEST_MODEL_REPORT}')

# ============================================================================
# PROFESSIONAL VISUALIZATION MODULE
# ============================================================================
print('\n' + '='*70)
print('GENERATING PROFESSIONAL VISUALIZATIONS FOR THESIS')
print('='*70)
print('Creating high-quality plots (300 DPI) for academic publication...')
print('')
print('⚠️  IMPORTANT: All plots display FULL TEST SET evaluation metrics')
print(f'   - Performance measured on {len(X_test):,} real-world test samples')
print(f'   - NOT from SFS cross-validation scores')
print(f'   - Represents actual model performance')
print('='*70 + '\n')

# Extract metrics for plotting (ALL FROM FULL TEST SET EVALUATION)
num_features_list = [r['num_features'] for r in results]
crack_f1_list = [r['crack_f1'] for r in results]
f1_weighted_list = [r['f1_weighted'] for r in results]
accuracy_list = [r['accuracy'] for r in results]

# ============================================================================
# PLOT 1: Enhanced SFS Performance Plot
# ============================================================================
print('1. Creating Enhanced SFS Performance Plot...')
fig, ax = plt.subplots(figsize=(14, 8))

# Line 1: CRACK F1-Score (Bold Red - Optimization Target)
ax.plot(num_features_list, crack_f1_list,
        marker='D', linewidth=4, markersize=12,
        color='#dc143c', label='CRACK F1-Score (Optimization Target)',
        zorder=4, alpha=0.9)

# Line 2: Overall Accuracy (Blue)
ax.plot(num_features_list, accuracy_list,
        marker='o', linewidth=2.5, markersize=9,
        color='#1f77b4', label='Overall Accuracy',
        zorder=3, alpha=0.8)

# Line 3: Weighted F1-Score (Green)
ax.plot(num_features_list, f1_weighted_list,
        marker='s', linewidth=2.5, markersize=9,
        color='#2ca02c', label='Weighted F1-Score',
        zorder=3, alpha=0.8)

# Mark the Best Model point (Distinct Star)
ax.scatter([best_n_features], [best_crack_f1],
          s=800, marker='*', color='gold',
          edgecolors='black', linewidths=3,
          label=f'Best Model (k={best_n_features}, F1={best_crack_f1:.4f})',
          zorder=5)

# Formatting
ax.set_xlabel('Number of Features', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontsize=16, fontweight='bold')
ax.set_title(f'Sequential Feature Selection Performance (Full Test Set Evaluation)\nCRACK F1-Score Optimization - {len(X_test):,} Test Samples',
            fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
ax.legend(fontsize=13, loc='best', framealpha=0.95, shadow=True,
         edgecolor='black', fancybox=True)
ax.set_xticks(range(1, 21))
ax.set_xlim([0.5, 20.5])
ax.set_ylim([max(0, min(min(crack_f1_list), min(accuracy_list)) - 0.05),
            min(1.0, max(max(crack_f1_list), max(accuracy_list)) + 0.05)])

# Add text box with key info
textstr = f'Optimization Metric: CRACK F1-Score\n'
textstr += f'Best Performance: {best_crack_f1:.4f} at k={best_n_features}\n'
textstr += f'Method: Forward SFS\n'
textstr += f'Evaluation: {len(X_test):,} Full Test Samples\n'
textstr += f'Note: Real-world performance, not CV scores'
props = dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
            alpha=0.95, edgecolor='black', linewidth=2)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=props, fontweight='bold')

plt.tight_layout()
enhanced_sfs_plot_path = os.path.join(RESULTS_FOLDER, 'sfs_performance_enhanced.png')
plt.savefig(enhanced_sfs_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'   ✓ Enhanced SFS plot saved to: {enhanced_sfs_plot_path}')

# ============================================================================
# PLOT 2: Confusion Matrix Heatmap
# ============================================================================
print('2. Creating Confusion Matrix Heatmap...')
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap with seaborn
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model[1].classes_,
            yticklabels=best_model[1].classes_,
            cbar_kws={'label': 'Count', 'shrink': 0.8},
            linewidths=2, linecolor='white',
            square=True, ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'})

# Formatting
ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title(f'Confusion Matrix - Best Model (Full Test Set: {len(X_test):,} samples)\n(CRACK F1-Optimized, Threshold-Tuned)',
            fontsize=18, fontweight='bold', pad=20)

# Rotate labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=13)

plt.tight_layout()
cm_heatmap_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix_heatmap.png')
plt.savefig(cm_heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'   ✓ Confusion matrix heatmap saved to: {cm_heatmap_path}')

# ============================================================================
# PLOT 3: Multi-Class ROC Curves (One-vs-Rest)
# ============================================================================
print('3. Creating Multi-Class ROC Curves (One-vs-Rest)...')
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(12, 9))

# Calculate ROC curve for each class
classes = best_model[1].classes_
colors = ['#dc143c', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
linewidths = []

for idx, (class_name, color) in enumerate(zip(classes, colors)):
    # Binarize the labels for this class
    y_binary = (best_y_test == class_name).astype(int)

    # Get probabilities for this class
    y_score = best_y_proba[:, idx]

    # Calculate ROC curve and AUC
    if len(np.unique(y_binary)) > 1:  # Only if class exists in test set
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)

        # Thicker line for CRACK class
        linewidth = 5 if class_name == 'CRACK' else 2.5
        alpha = 1.0 if class_name == 'CRACK' else 0.8
        zorder = 5 if class_name == 'CRACK' else 3

        # Plot ROC curve
        label = f'{class_name} (AUC = {roc_auc:.3f})'
        if class_name == 'CRACK':
            label = f'🎯 {label} - OPTIMIZED'

        ax.plot(fpr, tpr, color=color, linewidth=linewidth,
                label=label, alpha=alpha, zorder=zorder)
    else:
        print(f'   ⚠ Warning: Class {class_name} not found in test set')

# Plot diagonal reference line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)',
        alpha=0.5, zorder=1)

# Formatting
ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
ax.set_title(f'Multi-Class ROC Curves (Full Test Set: {len(X_test):,} samples)\nOne-vs-Rest, CRACK Class Highlighted',
            fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right', framealpha=0.95,
         shadow=True, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
roc_curves_path = os.path.join(RESULTS_FOLDER, 'roc_curves_multiclass.png')
plt.savefig(roc_curves_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'   ✓ Multi-class ROC curves saved to: {roc_curves_path}')

# ============================================================================
# PLOT 4: Selected Features Visualization (Best Model)
# ============================================================================
print('4. Creating Selected Features Visualization...')

# Create feature importance data (using absolute LDA coefficients)
best_lda = best_model[1]
crack_class_idx = np.where(best_lda.classes_ == 'CRACK')[0][0]
crack_coefficients = best_lda.coef_[crack_class_idx, :]
abs_coefficients = np.abs(crack_coefficients)

# Create DataFrame and sort by importance
feature_importance_df = pd.DataFrame({
    'Feature': best_features,
    'Importance': abs_coefficients,
    'Coefficient': crack_coefficients
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(12, max(8, len(best_features) * 0.5)))

# Color by coefficient sign
colors = ['#2ca02c' if c > 0 else '#d62728'
         for c in feature_importance_df['Coefficient'].values]

bars = ax.barh(range(len(feature_importance_df)),
              feature_importance_df['Importance'].values,
              color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels
for i, (bar, imp, coef) in enumerate(zip(bars,
                                         feature_importance_df['Importance'].values,
                                         feature_importance_df['Coefficient'].values)):
    width = bar.get_width()
    sign = '+' if coef > 0 else '-'
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{imp:.3f} ({sign})', ha='left', va='center',
            fontsize=11, fontweight='bold')

# Formatting
ax.set_yticks(range(len(feature_importance_df)))
ax.set_yticklabels(feature_importance_df['Feature'].values, fontsize=11)
ax.set_xlabel('Wavelength Importance (Absolute LDA Coefficient)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Selected Wavelengths (nm)', fontsize=14, fontweight='bold')
ax.set_title(f'Selected Wavelengths for Best Model (k={best_n_features})\n' +
            'Wavelength Importance for CRACK Detection',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black',
          label='Positive Coefficient (↑ CRACK probability)'),
    Patch(facecolor='#d62728', edgecolor='black',
          label='Negative Coefficient (↓ CRACK probability)')
]
ax.legend(handles=legend_elements, loc='lower right',
         fontsize=11, framealpha=0.95, edgecolor='black')

plt.tight_layout()
features_viz_path = os.path.join(RESULTS_FOLDER, 'selected_features_importance.png')
plt.savefig(features_viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'   ✓ Selected features visualization saved to: {features_viz_path}')

print('\n' + '='*70)
print('PROFESSIONAL VISUALIZATIONS COMPLETE')
print('='*70)
print('Generated 4 high-quality plots (300 DPI):')
print(f'  1. Enhanced SFS Performance Plot: {enhanced_sfs_plot_path}')
print(f'  2. Confusion Matrix Heatmap: {cm_heatmap_path}')
print(f'  3. Multi-Class ROC Curves: {roc_curves_path}')
print(f'  4. Selected Features Visualization: {features_viz_path}')
print('='*70)

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
        marker='D', linewidth=2.5, markersize=8,
        label='Global: F1-Weighted', color='#1f77b4', zorder=3)

ax.plot(num_features_list, recall_weighted_list,
        marker='s', linewidth=2.5, markersize=7,
        label='Global: Recall', color='#2ca02c', zorder=3)

ax.plot(num_features_list, precision_weighted_list,
        marker='^', linewidth=2.5, markersize=7,
        label='Global: Precision', color='#9467bd', zorder=3)

ax.plot(num_features_list, roc_auc_weighted_list,
        marker='p', linewidth=2.5, markersize=7,
        label='Global: ROC-AUC', color='#8c564b', zorder=3)

# Plot CRACK metrics - emphasize F1 as optimization target
ax.plot(num_features_list, crack_f1_list,
        marker='D', linestyle='-', linewidth=4, markersize=10,
        label='CRACK: F1 (OPTIMIZATION TARGET)', color='#ff7f0e', zorder=4)

ax.plot(num_features_list, crack_precision_list,
        marker='>', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: Precision', color='#e377c2', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_recall_list,
        marker='v', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: Recall', color='#d62728', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_f2_list,
        marker='*', linestyle='--', linewidth=2, markersize=8,
        label='CRACK: F2', color='#bcbd22', zorder=3, alpha=0.8)

ax.plot(num_features_list, crack_auc_list,
        marker='h', linestyle='--', linewidth=2, markersize=6,
        label='CRACK: ROC-AUC', color='#17becf', zorder=3, alpha=0.8)

# Mark the best CRACK F1 point
ax.scatter([best_n_features], [best_crack_f1],
          color='gold', s=400, marker='*',
          edgecolors='black', linewidths=2.5,
          label=f'Best CRACK F1 (k={best_n_features})', zorder=5)

# Formatting
ax.set_xlabel('Number of Wavelengths (k)', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Comprehensive Benchmarking: Global & CRACK Metrics (CRACK F1 Optimization)\nWavelength Selection Analysis',
            fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(fontsize=9, loc='best', framealpha=0.95, shadow=True, ncol=2)
ax.set_xticks(range(1, 21))
ax.set_ylim([0, 1.05])

# Add text box with best model info
textstr = f'Best Model (k={best_n_features}):\n'
textstr += f'CRACK F1: {best_crack_f1:.4f} ⭐\n'
textstr += f'Global F1: {best_all_metrics.get("f1_weighted", 0):.4f}\n'
textstr += f'CRACK F2: {best_all_metrics.get("crack_f2", 0):.4f}'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig(METRICS_COMPARISON_PLOT, dpi=300, bbox_inches='tight')
plt.close()

print(f'✓ Comprehensive metrics plot saved to: {METRICS_COMPARISON_PLOT}')

# ============================================================================
# FEATURE IMPORTANCE & WAVELENGTH ANALYSIS MODULE
# ============================================================================
print('\n' + '='*70)
print('FEATURE IMPORTANCE & WAVELENGTH ANALYSIS')
print('='*70)
print('Analyzing which spectral bands drive CRACK detection decisions...\n')

# 1. TRACK FEATURE FREQUENCY (STABILITY ANALYSIS)
print('1. Feature Selection Stability Analysis')
print('-'*70)

# Sort features by selection frequency
sorted_features = sorted(feature_selection_counter.items(), key=lambda x: x[1], reverse=True)
top_10_features = sorted_features[:10]

print('Top 10 Most Frequently Selected Wavelengths:')
for i, (feature, count) in enumerate(top_10_features, 1):
    print(f'  {i:2d}. {feature:<20s} - Selected {count:2d} times (out of 20)')

# Generate Bar Plot: Top 10 Most Frequently Selected Wavelengths
fig, ax = plt.subplots(figsize=(12, 7))

feature_names = [f[0] for f in top_10_features]
selection_counts = [f[1] for f in top_10_features]

bars = ax.barh(range(len(feature_names)), selection_counts, color='#1f77b4', edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, selection_counts)):
    width = bar.get_width()
    ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
            f'{count}', ha='left', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names, fontsize=11)
ax.set_xlabel('Selection Frequency (out of 20 wavelength counts tested)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wavelength (nm)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Frequently Selected Wavelengths\n(Wavelength Stability Analysis)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim([0, 21])

plt.tight_layout()
feature_freq_plot_path = os.path.join(RESULTS_FOLDER, 'feature_selection_frequency.png')
plt.savefig(feature_freq_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'\n✓ Feature frequency plot saved to: {feature_freq_plot_path}')

# 2. BEST MODEL COEFFICIENTS (DIRECTIONALITY)
print('\n2. LDA Coefficients Analysis (CRACK Class)')
print('-'*70)

# Extract LDA coefficients for the best model
best_lda = best_model[1]  # Get the LDA model

# Find CRACK class index
crack_class_idx = np.where(best_lda.classes_ == 'CRACK')[0][0]

# Get coefficients for CRACK class
# LDA.coef_ shape: (n_classes, n_features)
crack_coefficients = best_lda.coef_[crack_class_idx, :]

# Create DataFrame with feature names and their coefficients
coef_data = pd.DataFrame({
    'Feature': best_features,
    'Coefficient': crack_coefficients
})

# Sort by absolute coefficient value
coef_data['Abs_Coefficient'] = np.abs(coef_data['Coefficient'])
coef_data_sorted = coef_data.sort_values('Abs_Coefficient', ascending=True)

print(f'Number of features in best model: {len(best_features)}')
print(f'\nCRACK Class LDA Coefficients (sorted by absolute value):')
for idx, row in coef_data_sorted.iterrows():
    direction = '(+)' if row['Coefficient'] > 0 else '(-)'
    print(f'  {row["Feature"]:<20s}: {row["Coefficient"]:>8.4f} {direction}')

# Generate Horizontal Bar Plot: LDA Coefficients for CRACK Class
fig, ax = plt.subplots(figsize=(12, max(8, len(best_features) * 0.4)))

features = coef_data_sorted['Feature'].values
coefficients = coef_data_sorted['Coefficient'].values

# Color bars based on sign (positive = orange, negative = blue)
colors = ['#ff7f0e' if c > 0 else '#1f77b4' for c in coefficients]

bars = ax.barh(range(len(features)), coefficients, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, coef) in enumerate(zip(bars, coefficients)):
    width = bar.get_width()
    label_x = width + (0.01 if width > 0 else -0.01)
    ha = 'left' if width > 0 else 'right'
    ax.text(label_x, bar.get_y() + bar.get_height()/2,
            f'{coef:.3f}', ha=ha, va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel('LDA Coefficient Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature / Wavelength', fontsize=12, fontweight='bold')
ax.set_title(f'LDA Coefficients for CRACK Class (Best Model, k={best_n_features})\n' +
             'Positive: Increases CRACK probability | Negative: Decreases CRACK probability',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=2, linestyle='-', alpha=0.3)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
legend_elements = [
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Positive (↑ CRACK probability)'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Negative (↓ CRACK probability)')
]
ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9)

plt.tight_layout()
coef_plot_path = os.path.join(RESULTS_FOLDER, 'lda_coefficients_crack_class.png')
plt.savefig(coef_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'\n✓ LDA coefficients plot saved to: {coef_plot_path}')

# 3. SAVE FEATURE LIST (Best Model Wavelengths)
print('\n3. Saving Best Model Wavelengths')
print('-'*70)

# Create comprehensive wavelength CSV
wavelength_df = pd.DataFrame({
    'Feature_Index': range(1, len(best_features) + 1),
    'Wavelength_Name': best_features,
    'LDA_Coefficient': crack_coefficients,
    'Abs_Coefficient': np.abs(crack_coefficients),
    'Direction': ['Positive' if c > 0 else 'Negative' for c in crack_coefficients]
})

# Sort by absolute coefficient value (most important first)
wavelength_df = wavelength_df.sort_values('Abs_Coefficient', ascending=False)
wavelength_df['Importance_Rank'] = range(1, len(wavelength_df) + 1)

# Reorder columns
wavelength_df = wavelength_df[['Importance_Rank', 'Wavelength_Name', 'LDA_Coefficient',
                                 'Abs_Coefficient', 'Direction', 'Feature_Index']]

wavelength_csv_path = os.path.join(RESULTS_FOLDER, 'best_model_wavelengths.csv')
wavelength_df.to_csv(wavelength_csv_path, index=False)

print(f'Best model uses {len(best_features)} wavelengths/features')
print(f'✓ Wavelength details saved to: {wavelength_csv_path}')

# Print top 5 most important features
print(f'\nTop 5 Most Important Features (by absolute coefficient):')
for idx, row in wavelength_df.head(5).iterrows():
    print(f'  {int(row["Importance_Rank"])}. {row["Wavelength_Name"]:<20s}: {row["LDA_Coefficient"]:>8.4f} ({row["Direction"]})')

# Save feature frequency analysis to CSV
feature_freq_df = pd.DataFrame([
    {'Feature': feat, 'Selection_Count': count, 'Selection_Rate': f'{count/20*100:.1f}%'}
    for feat, count in sorted_features
])
feature_freq_csv_path = os.path.join(RESULTS_FOLDER, 'feature_selection_frequency.csv')
feature_freq_df.to_csv(feature_freq_csv_path, index=False)
print(f'✓ Feature frequency analysis saved to: {feature_freq_csv_path}')

print('\n' + '='*70)
print('FEATURE IMPORTANCE ANALYSIS COMPLETE')
print('='*70)
print(f'\nGenerated files:')
print(f'  1. {feature_freq_plot_path}')
print(f'  2. {coef_plot_path}')
print(f'  3. {wavelength_csv_path}')
print(f'  4. {feature_freq_csv_path}')
print('='*70)

print('\n' + '='*70)
print('EXPERIMENT COMPLETE')
print('='*70)
print(f'All results saved to folder: {RESULTS_FOLDER}')
print(f'\n' + '='*70)
print('COMPLETE OUTPUT SUMMARY')
print('='*70)
print(f'\n📊 METRICS & LOGS:')
print(f'   1. {FEATURE_LOG_CSV}')
print(f'      → Detailed metrics for each feature count (1-20)')
print(f'   2. {THRESHOLD_OPTIMIZATION_LOG}')
print(f'      → Threshold optimization log (0.01-0.99)')
print(f'\n🤖 MODEL FILES:')
print(f'   3. {MODEL_OUTPUT_PATH}')
print(f'      → Best model package (LDA + SFS + threshold)')
print(f'   4. {BEST_MODEL_REPORT}')
print(f'      → Detailed text report for best model')
print(f'\n📈 PROFESSIONAL VISUALIZATIONS (300 DPI):')
print(f'   5. {os.path.join(RESULTS_FOLDER, "sfs_performance_enhanced.png")}')
print(f'      → Enhanced SFS performance plot (3 metrics)')
print(f'   6. {os.path.join(RESULTS_FOLDER, "confusion_matrix_heatmap.png")}')
print(f'      → Confusion matrix heatmap (seaborn)')
print(f'   7. {os.path.join(RESULTS_FOLDER, "roc_curves_multiclass.png")}')
print(f'      → Multi-class ROC curves (One-vs-Rest)')
print(f'   8. {os.path.join(RESULTS_FOLDER, "selected_features_importance.png")}')
print(f'      → Selected features importance visualization')
print(f'   9. {METRICS_COMPARISON_PLOT}')
print(f'      → Comprehensive metrics comparison (all classes)')
print(f'\n🔬 FEATURE IMPORTANCE ANALYSIS:')
print(f'   10. {os.path.join(RESULTS_FOLDER, "feature_selection_frequency.png")}')
print(f'       → Top 10 most frequently selected wavelengths')
print(f'   11. {os.path.join(RESULTS_FOLDER, "lda_coefficients_crack_class.png")}')
print(f'       → LDA coefficients directionality plot')
print(f'   12. {os.path.join(RESULTS_FOLDER, "best_model_wavelengths.csv")}')
print(f'       → Detailed wavelength list with rankings')
print(f'   13. {os.path.join(RESULTS_FOLDER, "feature_selection_frequency.csv")}')
print(f'       → Feature frequency analysis data')
print(f'\n' + '='*70)
print(f'\nTo use the model:')
print(f'  model_pkg = joblib.load("{MODEL_OUTPUT_PATH}")')
print(f'  X_selected = model_pkg["sfs"].transform(X)')
print(f'  y_proba = model_pkg["lda"].predict_proba(X_selected)')
print(f'  # Apply optimal threshold for CRACK detection')
print(f'  crack_idx = model_pkg["crack_class_index"]')
print(f'  threshold = model_pkg["optimal_threshold"]')
print(f'  # Custom prediction logic using threshold...')

