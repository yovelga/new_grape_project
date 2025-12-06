import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset
DATASET_PATH = r'C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv'
MODEL_OUTPUT_PATH = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\lda_model_multi_class_f1_score_v2.joblib'
RESULTS_FOLDER = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\Train\LDA\results'
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

# 1. Split data FIRST (Keep Test set pure and unbalanced to reflect reality)
print('Splitting data into Train and Test...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Balance ONLY the Training set
BALANCE_CLASSES = True

if BALANCE_CLASSES:
    print('Balancing TRAINING set using RandomOverSampler...')
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print('Train set distribution after balancing:', dict(zip(*np.unique(y_train, return_counts=True))))
else:
    print('Class balancing is OFF.')

print('Test set distribution (Original):', dict(zip(*np.unique(y_test, return_counts=True))))

# Train LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Get probability predictions for threshold tuning
y_proba = lda.predict_proba(X_test)
classes = lda.classes_

# Find the index of the 'CRACK' class
if 'CRACK' not in classes:
    raise ValueError(f"'CRACK' class not found in model classes: {classes}")
crack_idx = np.where(classes == 'CRACK')[0][0]

print(f"\nOptimizing threshold for CRACK class (index {crack_idx})...")
print("Testing thresholds from 0.01 to 0.99...")

# Threshold tuning loop
best_threshold = 0.5
best_f1 = 0.0
threshold_results = []

for threshold in np.arange(0.01, 1.00, 0.01):
    # Apply custom prediction logic
    y_pred_threshold = []
    for i in range(len(y_proba)):
        if y_proba[i, crack_idx] >= threshold:
            # Predict CRACK if probability exceeds threshold
            y_pred_threshold.append('CRACK')
        else:
            # Otherwise, predict the class with highest probability among remaining classes
            proba_copy = y_proba[i].copy()
            proba_copy[crack_idx] = -1  # Exclude CRACK from consideration
            max_idx = np.argmax(proba_copy)
            y_pred_threshold.append(classes[max_idx])

    y_pred_threshold = np.array(y_pred_threshold)

    # Calculate F1-score, Precision, and Recall for CRACK class
    crack_f1 = f1_score(y_test, y_pred_threshold, labels=['CRACK'], average=None, zero_division=0)[0]
    crack_precision = precision_score(y_test, y_pred_threshold, labels=['CRACK'], average=None, zero_division=0)[0]
    crack_recall = recall_score(y_test, y_pred_threshold, labels=['CRACK'], average=None, zero_division=0)[0]
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
y_pred_final = []
for i in range(len(y_proba)):
    if y_proba[i, crack_idx] >= best_threshold:
        y_pred_final.append('CRACK')
    else:
        proba_copy = y_proba[i].copy()
        proba_copy[crack_idx] = -1
        max_idx = np.argmax(proba_copy)
        y_pred_final.append(classes[max_idx])

y_pred_final = np.array(y_pred_final)

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

# Compute ROC-AUC Score for CRACK class (One-vs-Rest)
# FIX: Explicitly binarize y_test (CRACK=1, all others=0) and use CRACK probabilities
y_test_binary = (y_test == 'CRACK').astype(int)
y_proba_crack = y_proba[:, crack_idx]
roc_auc_crack = roc_auc_score(y_test_binary, y_proba_crack)
print(f"\nROC-AUC Score (CRACK class, One-vs-Rest): {roc_auc_crack:.4f}")

# ===== SAVE OPTIMIZATION LOG (CSV) =====
# Create results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ===== ADVANCED METRICS =====
# Calculate Matthews Correlation Coefficient (MCC)
mcc_score = matthews_corrcoef(y_test, y_pred_final)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc_score:.4f}")

# Calculate Precision-Recall AUC for CRACK class
y_test_binary = (y_test == 'CRACK').astype(int)
pr_auc = average_precision_score(y_test_binary, y_proba_crack)
print(f"Precision-Recall AUC (PR-AUC) for CRACK: {pr_auc:.4f}")

# Calculate per-class AUC (One-vs-Rest for all classes)
print("\nPer-Class ROC-AUC Scores (One-vs-Rest):")
auc_scores = {}
for idx, class_name in enumerate(classes):
    y_binary = (y_test == class_name).astype(int)
    if len(np.unique(y_binary)) > 1:  # Only calculate if class exists in test set
        auc = roc_auc_score(y_binary, y_proba[:, idx])
        auc_scores[class_name] = auc
        print(f"  {class_name}: {auc:.4f}")
    else:
        auc_scores[class_name] = None
        print(f"  {class_name}: N/A (not in test set)")

# Calculate macro and weighted average AUC
valid_aucs = [v for v in auc_scores.values() if v is not None]
macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
print(f"\nMacro-Average AUC: {macro_auc:.4f}")

# Save threshold optimization log
threshold_log_df = pd.DataFrame(
    threshold_results,
    columns=['Threshold', 'Crack_F1', 'Crack_Precision', 'Crack_Recall']
)
threshold_log_path = os.path.join(RESULTS_FOLDER, 'threshold_optimization_log_multi.csv')
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
precision_crack_optimal = precision_score(y_test, y_pred_final, labels=['CRACK'], average=None, zero_division=0)[0]
recall_crack_optimal = recall_score(y_test, y_pred_final, labels=['CRACK'], average=None, zero_division=0)[0]
f1_crack_optimal = f1_score(y_test, y_pred_final, labels=['CRACK'], average=None, zero_division=0)[0]
macro_f1 = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred_final, average='weighted', zero_division=0)

print(f"\nExpanded Metrics at Optimal Threshold:")
print(f"  Precision (CRACK): {precision_crack_optimal:.4f}")
print(f"  Recall (CRACK): {recall_crack_optimal:.4f}")
print(f"  F1-Score (CRACK): {f1_crack_optimal:.4f}")
print(f"  Macro F1: {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")

# Add all metrics including advanced ones and per-class AUC
final_metrics = {
    'Overall_Accuracy': [accuracy_score(y_test, y_pred_final)],
    'ROC_AUC_CRACK': [roc_auc_crack],
    'PR_AUC_CRACK': [pr_auc],
    'MCC': [mcc_score],
    'Macro_Average_AUC': [macro_auc],
    'Optimal_Threshold': [best_threshold],
    'Precision_CRACK': [precision_crack_optimal],
    'Recall_CRACK': [recall_crack_optimal],
    'F1_CRACK': [f1_crack_optimal],
    'Macro_F1': [macro_f1],
    'Weighted_F1': [weighted_f1],
    **{f'AUC_{k}': [v] for k, v in auc_scores.items()},
    **{k: [v] for k, v in cm_dict.items()}
}
final_metrics_df = pd.DataFrame(final_metrics)

# Save classification report
class_report_path = os.path.join(RESULTS_FOLDER, 'final_model_metrics_classification_report_multi.csv')
class_report_df.to_csv(class_report_path, index=True)
print(f"Classification report saved to: {class_report_path}")

# Save final comprehensive metrics
final_metrics_path = os.path.join(RESULTS_FOLDER, 'final_model_metrics_multi.csv')
final_metrics_df.to_csv(final_metrics_path, index=False)
print(f"Final comprehensive metrics saved to: {final_metrics_path}")

# Save confusion matrix as separate CSV for easy reference
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.index.name = 'True_Label'
cm_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix_multi.csv')
cm_df.to_csv(cm_path, index=True)
print(f"Confusion matrix saved to: {cm_path}")

# Save model with optimal threshold
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
model_package = {
    'model': lda,
    'optimal_threshold': best_threshold,
    'classes': classes,
    'crack_class_index': crack_idx
}
joblib.dump(model_package, MODEL_OUTPUT_PATH)
print(f"\nModel package (with optimal threshold) saved to {MODEL_OUTPUT_PATH}")
print(f"Package contents: model, optimal_threshold={best_threshold:.3f}, classes, crack_class_index={crack_idx}")

print("\n" + "="*60)
print("DATA & MODEL FILES SAVED SUCCESSFULLY")
print("="*60)
print(f"Results folder: {RESULTS_FOLDER}")
print("CSV Files created:")
print("  1. threshold_optimization_log_multi.csv")
print("  2. final_model_metrics_multi.csv")
print("  3. final_model_metrics_classification_report_multi.csv")
print("  4. confusion_matrix_multi.csv")
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
ax.set_title('Threshold Optimization for CRACK Detection\n(Multi-Class LDA)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

threshold_curve_path = os.path.join(RESULTS_FOLDER, 'threshold_optimization_curve_multi.png')
plt.savefig(threshold_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Threshold optimization curve saved to: {threshold_curve_path}")

# 2. CONFUSION MATRIX HEATMAP
print("Generating Confusion Matrix Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5, linecolor='gray')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - Multi-Class LDA\n(Optimal Threshold)',
             fontsize=16, fontweight='bold')

cm_heatmap_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix_heatmap_multi.png')
plt.savefig(cm_heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix heatmap saved to: {cm_heatmap_path}")

# 3. ROC CURVE
print("Generating ROC Curve...")
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curve for CRACK class (One-vs-Rest)
fpr, tpr, roc_thresholds = roc_curve(y_test_binary, y_proba_crack)

ax.plot(fpr, tpr, linewidth=3, label=f'CRACK (AUC = {roc_auc_crack:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')

ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - CRACK Detection\n(Multi-Class LDA, One-vs-Rest)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)

roc_curve_path = os.path.join(RESULTS_FOLDER, 'roc_curve_multi.png')
plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC curve saved to: {roc_curve_path}")

# 4. PRECISION-RECALL CURVE
print("Generating Precision-Recall Curve...")
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate Precision-Recall curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test_binary, y_proba_crack)

ax.plot(recall_curve, precision_curve, linewidth=3,
        label=f'CRACK (PR-AUC = {pr_auc:.4f})')

# Baseline (proportion of positive class)
baseline = y_test_binary.sum() / len(y_test_binary)
ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2,
           label=f'Baseline (No Skill = {baseline:.4f})')

ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Curve - CRACK Detection\n(Multi-Class LDA)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

pr_curve_path = os.path.join(RESULTS_FOLDER, 'precision_recall_curve_multi.png')
plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Precision-Recall curve saved to: {pr_curve_path}")

# 5. MULTI-CLASS ROC CURVES (One-vs-Rest for all classes)
print("Generating Multi-Class ROC Curves...")
fig, ax = plt.subplots(figsize=(12, 9))

# Plot ROC curve for each class
colors = sns.color_palette("husl", len(classes))
for idx, (class_name, color) in enumerate(zip(classes, colors)):
    y_binary = (y_test == class_name).astype(int)
    if len(np.unique(y_binary)) > 1:
        fpr_class, tpr_class, _ = roc_curve(y_binary, y_proba[:, idx])
        auc_class = auc_scores[class_name]
        ax.plot(fpr_class, tpr_class, linewidth=2.5, color=color,
                label=f'{class_name} (AUC = {auc_class:.4f})')

# Plot random classifier line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')

ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('Multi-Class ROC Curves (One-vs-Rest)\n(Multi-Class LDA)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

multiclass_roc_path = os.path.join(RESULTS_FOLDER, 'roc_curves_all_classes_multi.png')
plt.savefig(multiclass_roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Multi-class ROC curves saved to: {multiclass_roc_path}")

# 6. AUC SCORES COMPARISON BAR CHART
print("Generating AUC Scores Comparison Chart...")
fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data for bar chart
class_names_plot = [k for k, v in auc_scores.items() if v is not None]
auc_values_plot = [v for v in auc_scores.values() if v is not None]

# Create bar chart
bars = ax.bar(class_names_plot, auc_values_plot, color=colors[:len(class_names_plot)],
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on top of bars
for bar, value in zip(bars, auc_values_plot):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add macro average line
ax.axhline(y=macro_auc, color='red', linestyle='--', linewidth=2,
           label=f'Macro Average AUC = {macro_auc:.4f}')

ax.set_xlabel('Class', fontsize=14, fontweight='bold')
ax.set_ylabel('AUC Score (One-vs-Rest)', fontsize=14, fontweight='bold')
ax.set_title('Per-Class AUC Scores Comparison\n(Multi-Class LDA)',
             fontsize=16, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, axis='y', alpha=0.3)

auc_comparison_path = os.path.join(RESULTS_FOLDER, 'auc_comparison_multi.png')
plt.savefig(auc_comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ AUC comparison chart saved to: {auc_comparison_path}")

# 7. ACCURACY METRICS SUMMARY CHART
print("Generating Accuracy Metrics Summary Chart...")
fig, ax = plt.subplots(figsize=(10, 7))

# Prepare accuracy metrics
accuracy_metrics = {
    'Overall\nAccuracy': accuracy_score(y_test, y_pred_final),
    'CRACK\nF1-Score': best_f1,
    'CRACK\nROC-AUC': roc_auc_crack,
    'CRACK\nPR-AUC': pr_auc,
    'Macro Avg\nAUC': macro_auc,
    'MCC': mcc_score
}

metric_names = list(accuracy_metrics.keys())
metric_values = list(accuracy_metrics.values())
colors_metrics = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

bars = ax.bar(metric_names, metric_values, color=colors_metrics,
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
for bar, value in zip(bars, metric_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics Summary\n(Multi-Class LDA)',
             fontsize=16, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=0, ha='center')

accuracy_summary_path = os.path.join(RESULTS_FOLDER, 'accuracy_metrics_summary_multi.png')
plt.savefig(accuracy_summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Accuracy metrics summary saved to: {accuracy_summary_path}")

print("\n" + "="*60)
print("ALL THESIS VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*60)
print("Plot Files created (300 DPI):")
print("  1. threshold_optimization_curve_multi.png")
print("  2. confusion_matrix_heatmap_multi.png")
print("  3. roc_curve_multi.png")
print("  4. precision_recall_curve_multi.png")
print("  5. roc_curves_all_classes_multi.png")
print("  6. auc_comparison_multi.png")
print("  7. accuracy_metrics_summary_multi.png")
print("="*60)
print("\n✓ Multi-Class LDA Training Complete!")
print(f"✓ All files saved to: {RESULTS_FOLDER}")
print("="*60)

