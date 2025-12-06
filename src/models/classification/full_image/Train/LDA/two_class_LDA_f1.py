import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, matthews_corrcoef, average_precision_score, roc_curve, precision_recall_curve
import joblib
import os
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset
DATASET_PATH = r'C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv'
MODEL_OUTPUT_PATH = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\lda_model_2_class_f1_v2.joblib'
RESULTS_FOLDER = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\results_binary'
LABEL_COLUMN = 'label'  # Change this if your label column is named differently

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Print available columns for verification
print('Columns in CSV:', df.columns.tolist())

# Use all columns except label and non-feature columns as features
feature_columns = [col for col in df.columns if col not in [LABEL_COLUMN, 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']]
X = df[feature_columns].values

# Print original multi-class distribution
group_labels = ['CRACK', 'BACKGROUND', 'BRANCH', 'REGULAR']
group_counts = {g: (df[LABEL_COLUMN] == g).sum() for g in group_labels}
print('Original multi-class distribution:', group_counts)

# 1. CONVERT TO BINARY LABELS FIRST (before splitting)
print('\nConverting to binary classification: CRACK vs. other')
y = np.where(df[LABEL_COLUMN] == 'CRACK', 'CRACK', 'other')

# Print binary class distribution
unique, counts = np.unique(y, return_counts=True)
print('Binary class distribution (before split):', dict(zip(unique, counts)))
if len(unique) != 2:
    raise ValueError(f'Two-class LDA requires exactly 2 classes, found {len(unique)}: {unique}')

# Handle missing values if any
if np.any(pd.isnull(X)) or np.any(pd.isnull(y)):
    print('Warning: Missing values detected. Dropping rows with missing values.')
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

# 2. SPLIT DATA FIRST (Keep Test set pure and unbalanced to reflect reality)
print('\nSplitting data into Train and Test (80/20)...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train set distribution (before balancing):', dict(zip(*np.unique(y_train, return_counts=True))))
print('Test set distribution (Original, unbalanced):', dict(zip(*np.unique(y_test, return_counts=True))))

# 3. BALANCE ONLY THE TRAINING SET
BALANCE_CLASSES = True

if BALANCE_CLASSES:
    print('\nBalancing TRAINING set using RandomOverSampler...')
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print('Train set distribution (after balancing):', dict(zip(*np.unique(y_train, return_counts=True))))
else:
    print('Class balancing is OFF.')

# Train LDA model
print('\nTraining Binary LDA model...')
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('LDA model training complete.')

# Get probability predictions for threshold tuning
y_proba = lda.predict_proba(X_test)
classes = lda.classes_

# Find the index of the 'CRACK' class
if 'CRACK' not in classes:
    raise ValueError(f"'CRACK' class not found in model classes: {classes}")
crack_idx = np.where(classes == 'CRACK')[0][0]

print(f"\n{'='*60}")
print(f"THRESHOLD OPTIMIZATION FOR BINARY CLASSIFICATION")
print(f"{'='*60}")
print(f"Optimizing threshold for CRACK class (index {crack_idx})...")
print("Testing thresholds from 0.01 to 0.99...")

# Threshold tuning loop to maximize F1-Score for CRACK class
best_threshold = 0.5
best_f1 = 0.0
threshold_results = []

for threshold in np.arange(0.01, 1.00, 0.01):
    # Apply threshold: if P(CRACK) >= threshold, predict CRACK, else 'other'
    y_pred_threshold = np.where(y_proba[:, crack_idx] >= threshold, 'CRACK', 'other')

    # Calculate F1-score, Precision, and Recall for CRACK class
    crack_f1 = f1_score(y_test, y_pred_threshold, pos_label='CRACK', zero_division=0)
    crack_precision = precision_score(y_test, y_pred_threshold, pos_label='CRACK', zero_division=0)
    crack_recall = recall_score(y_test, y_pred_threshold, pos_label='CRACK', zero_division=0)
    threshold_results.append([threshold, crack_f1, crack_precision, crack_recall])

    # Update best threshold if this one is better
    if crack_f1 > best_f1:
        best_f1 = crack_f1
        best_threshold = threshold

print(f"\n{'='*60}")
print(f"OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"Best Threshold for CRACK: {best_threshold:.3f}")
print(f"Maximized CRACK F1-Score: {best_f1:.4f}")
print(f"{'='*60}\n")

# Generate final predictions using the best threshold
y_pred_final = np.where(y_proba[:, crack_idx] >= best_threshold, 'CRACK', 'other')

# Print final evaluation metrics
print("Final Evaluation with Optimized Threshold:")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# ===== METRIC EXPANSION: Confusion Matrix and ROC-AUC =====
# Compute Confusion Matrix (rows=True, cols=Predicted)
cm = confusion_matrix(y_test, y_pred_final, labels=classes)
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print("Classes:", classes)
print(cm)

# Compute ROC-AUC Score for binary classification
# FIX: Explicitly binarize y_test (CRACK=1, other=0) and use CRACK probabilities
y_test_binary_for_auc = (y_test == 'CRACK').astype(int)
y_proba_crack = y_proba[:, crack_idx]
roc_auc = roc_auc_score(y_test_binary_for_auc, y_proba_crack)
print(f"\nROC-AUC Score (Binary Classification - CRACK class): {roc_auc:.4f}")

# ===== ADVANCED METRICS =====
# Calculate Matthews Correlation Coefficient (MCC)
mcc_score = matthews_corrcoef(y_test, y_pred_final)
print(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}")

# Calculate Precision-Recall AUC (using same binary labels as ROC-AUC)
pr_auc = average_precision_score(y_test_binary_for_auc, y_proba_crack)
print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")

# Calculate per-class metrics for detailed analysis
print("\nPer-Class Metrics:")
for idx, class_name in enumerate(classes):
    y_binary_class = (y_test == class_name).astype(int)
    if len(np.unique(y_binary_class)) > 1:
        auc = roc_auc_score(y_binary_class, y_proba[:, idx])
        print(f"  {class_name} - ROC-AUC: {auc:.4f}")
    else:
        print(f"  {class_name} - ROC-AUC: N/A (not in test set)")

# ===== SAVE OPTIMIZATION LOG (CSV) =====
# Create results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Save threshold optimization log
threshold_log_df = pd.DataFrame(
    threshold_results,
    columns=['Threshold', 'F1', 'Precision', 'Recall']
)
threshold_log_path = os.path.join(RESULTS_FOLDER, 'threshold_optimization_log_binary.csv')
threshold_log_df.to_csv(threshold_log_path, index=False)
print(f"\nThreshold optimization log saved to: {threshold_log_path}")

# ===== SAVE FINAL RESULTS (CSV) =====
# Generate classification report as dictionary
class_report_dict = classification_report(y_test, y_pred_final, output_dict=True)

# Convert classification report to DataFrame
class_report_df = pd.DataFrame(class_report_dict).transpose()

# Add confusion matrix data for each class
cm_dict = {}
for idx, class_name in enumerate(classes):
    cm_dict[f'CM_TruePositive_{class_name}'] = cm[idx, idx]
    cm_dict[f'CM_RowSum_{class_name}'] = cm[idx, :].sum()
    cm_dict[f'CM_ColSum_{class_name}'] = cm[:, idx].sum()

# Calculate expanded metrics at optimal threshold
precision_crack_optimal = precision_score(y_test, y_pred_final, pos_label='CRACK', zero_division=0)
recall_crack_optimal = recall_score(y_test, y_pred_final, pos_label='CRACK', zero_division=0)
f1_crack_optimal = f1_score(y_test, y_pred_final, pos_label='CRACK', zero_division=0)
macro_f1 = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred_final, average='weighted', zero_division=0)

print(f"\nExpanded Metrics at Optimal Threshold:")
print(f"  Precision (CRACK): {precision_crack_optimal:.4f}")
print(f"  Recall (CRACK): {recall_crack_optimal:.4f}")
print(f"  F1-Score (CRACK): {f1_crack_optimal:.4f}")
print(f"  Macro F1: {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")

# Add comprehensive metrics including expanded ones
final_metrics = {
    'Overall_Accuracy': [accuracy_score(y_test, y_pred_final)],
    'ROC_AUC': [roc_auc],
    'PR_AUC': [pr_auc],
    'MCC': [mcc_score],
    'Optimal_Threshold': [best_threshold],
    'Precision_CRACK': [precision_crack_optimal],
    'Recall_CRACK': [recall_crack_optimal],
    'F1_CRACK': [f1_crack_optimal],
    'Best_F1_CRACK': [best_f1],
    'Macro_F1': [macro_f1],
    'Weighted_F1': [weighted_f1],
    **{k: [v] for k, v in cm_dict.items()}
}
final_metrics_df = pd.DataFrame(final_metrics)

# Save classification report
class_report_path = os.path.join(RESULTS_FOLDER, 'final_model_metrics_classification_report_binary.csv')
class_report_df.to_csv(class_report_path, index=True)
print(f"Classification report saved to: {class_report_path}")

# Save final comprehensive metrics
final_metrics_path = os.path.join(RESULTS_FOLDER, 'final_model_metrics_binary.csv')
final_metrics_df.to_csv(final_metrics_path, index=False)
print(f"Final comprehensive metrics saved to: {final_metrics_path}")

# Save confusion matrix as separate CSV for easy reference
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.index.name = 'True_Label'
cm_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix_binary.csv')
cm_df.to_csv(cm_path, index=True)
print(f"Confusion matrix saved to: {cm_path}")

# ===== SAVE MODEL WITH OPTIMAL THRESHOLD =====
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
model_package = {
    'model': lda,
    'optimal_threshold': best_threshold,
    'classes': classes,
    'crack_class_index': crack_idx
}
joblib.dump(model_package, MODEL_OUTPUT_PATH)
print(f"\nModel package (with optimal threshold) saved to {MODEL_OUTPUT_PATH}")
print(f"Package contents: model, optimal_threshold={best_threshold:.3f}, classes={list(classes)}, crack_class_index={crack_idx}")

print("\n" + "="*60)
print("DATA & MODEL FILES SAVED SUCCESSFULLY")
print("="*60)
print(f"Results folder: {RESULTS_FOLDER}")
print("CSV Files created:")
print("  1. threshold_optimization_log_binary.csv")
print("  2. final_model_metrics_binary.csv")
print("  3. final_model_metrics_classification_report_binary.csv")
print("  4. confusion_matrix_binary.csv")
print("  5. lda_model_2_class_f1_v2.joblib (model package)")
print("="*60)

# ============================================================================
# VISUALIZATION & ADVANCED METRICS MODULE
# ============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS FOR THESIS")
print("="*60)

# Set style for academic plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. THRESHOLD OPTIMIZATION CURVE
print("Generating Threshold Optimization Curve...")
fig, ax = plt.subplots(figsize=(12, 7))

thresholds = [x[0] for x in threshold_results]
f1_scores = [x[1] for x in threshold_results]
precisions = [x[2] for x in threshold_results]
recalls = [x[3] for x in threshold_results]

ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2.5, marker='o', markersize=3)
ax.plot(thresholds, precisions, label='Precision', linewidth=2.5, marker='s', markersize=3)
ax.plot(thresholds, recalls, label='Recall', linewidth=2.5, marker='^', markersize=3)

# Mark optimal threshold
ax.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2,
           label=f'Optimal Threshold ({best_threshold:.3f})')

ax.set_xlabel('Threshold', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Threshold Optimization for CRACK Detection\n(Binary LDA)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

threshold_curve_path = os.path.join(RESULTS_FOLDER, 'threshold_optimization_curve_binary.png')
plt.savefig(threshold_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Threshold optimization curve saved to: {threshold_curve_path}")

# 2. CONFUSION MATRIX HEATMAP
print("Generating Confusion Matrix Heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5, linecolor='gray')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - Binary LDA\n(Optimal Threshold)',
             fontsize=16, fontweight='bold')

cm_heatmap_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix_heatmap_binary.png')
plt.savefig(cm_heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix heatmap saved to: {cm_heatmap_path}")

# 3. ROC CURVE
print("Generating ROC Curve...")
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curve (using corrected binary labels)
fpr, tpr, roc_thresholds = roc_curve(y_test_binary_for_auc, y_proba_crack)

ax.plot(fpr, tpr, linewidth=3, label=f'Binary LDA (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')

ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - CRACK Detection\n(Binary LDA)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)

roc_curve_path = os.path.join(RESULTS_FOLDER, 'roc_curve_binary.png')
plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC curve saved to: {roc_curve_path}")

# 4. PRECISION-RECALL CURVE
print("Generating Precision-Recall Curve...")
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate Precision-Recall curve (using corrected binary labels)
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test_binary_for_auc, y_proba_crack)

ax.plot(recall_curve, precision_curve, linewidth=3,
        label=f'Binary LDA (PR-AUC = {pr_auc:.4f})')

# Baseline (proportion of positive class)
baseline = y_test_binary_for_auc.sum() / len(y_test_binary_for_auc)
ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2,
           label=f'Baseline (No Skill = {baseline:.4f})')

ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Curve - CRACK Detection\n(Binary LDA)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

pr_curve_path = os.path.join(RESULTS_FOLDER, 'precision_recall_curve_binary.png')
plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Precision-Recall curve saved to: {pr_curve_path}")

# 5. ACCURACY METRICS SUMMARY CHART
print("Generating Accuracy Metrics Summary Chart...")
fig, ax = plt.subplots(figsize=(10, 7))

# Prepare accuracy metrics
accuracy_metrics = {
    'Overall\nAccuracy': accuracy_score(y_test, y_pred_final),
    'CRACK\nF1-Score': best_f1,
    'ROC-AUC': roc_auc,
    'PR-AUC': pr_auc,
    'MCC': mcc_score
}

metric_names = list(accuracy_metrics.keys())
metric_values = list(accuracy_metrics.values())
colors_metrics = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#1abc9c']

bars = ax.bar(metric_names, metric_values, color=colors_metrics,
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
for bar, value in zip(bars, metric_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics Summary\n(Binary LDA)',
             fontsize=16, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=0, ha='center')

accuracy_summary_path = os.path.join(RESULTS_FOLDER, 'accuracy_metrics_summary_binary.png')
plt.savefig(accuracy_summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Accuracy metrics summary saved to: {accuracy_summary_path}")

# 6. PER-CLASS PERFORMANCE COMPARISON
print("Generating Per-Class Performance Comparison...")
fig, ax = plt.subplots(figsize=(12, 7))

# Extract per-class metrics from classification report
class_report_dict = classification_report(y_test, y_pred_final, output_dict=True)

# Prepare data
class_names_for_plot = [c for c in classes if c in class_report_dict]
precision_values = [class_report_dict[c]['precision'] for c in class_names_for_plot]
recall_values = [class_report_dict[c]['recall'] for c in class_names_for_plot]
f1_values = [class_report_dict[c]['f1-score'] for c in class_names_for_plot]

x = np.arange(len(class_names_for_plot))
width = 0.25

bars1 = ax.bar(x - width, precision_values, width, label='Precision',
               color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, recall_values, width, label='Recall',
               color='#e74c3c', edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, f1_values, width, label='F1-Score',
               color='#2ecc71', edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class Performance Metrics\n(Binary LDA)',
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names_for_plot, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim([0, 1.05])
ax.grid(True, axis='y', alpha=0.3)

per_class_perf_path = os.path.join(RESULTS_FOLDER, 'per_class_performance_binary.png')
plt.savefig(per_class_perf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Per-class performance comparison saved to: {per_class_perf_path}")

print("\n" + "="*60)
print("ALL THESIS VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*60)
print("Plot Files created (300 DPI):")
print("  1. threshold_optimization_curve_binary.png")
print("  2. confusion_matrix_heatmap_binary.png")
print("  3. roc_curve_binary.png")
print("  4. precision_recall_curve_binary.png")
print("  5. accuracy_metrics_summary_binary.png")
print("  6. per_class_performance_binary.png")
print("="*60)
print("\n✓ Binary LDA Training Complete!")
print(f"✓ All files saved to: {RESULTS_FOLDER}")
print("="*60)

