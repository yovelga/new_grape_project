import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Path to dataset
DATASET_PATH = str(_PROJECT_ROOT / r'src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv')
RESULTS_FOLDER = r'results_multi_class_reduce'
MODEL_OUTPUT_PATH = os.path.join(RESULTS_FOLDER, 'lda_model_best.joblib')
FEATURE_LOG_CSV = os.path.join(RESULTS_FOLDER, 'feature_selection_log.csv')
BEST_MODEL_REPORT = os.path.join(RESULTS_FOLDER, 'best_model_report.txt')
ACCURACY_PLOT = os.path.join(RESULTS_FOLDER, 'accuracy_plot.png')
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

# OPTIMIZATION: Create a smaller subset for fast RFE feature selection
SELECTION_SAMPLE_SIZE =  float('inf')
print('\n' + '='*70)
print('DATASET OPTIMIZATION')
print('='*70)
print(f'Full training set size: {len(X_train)} samples')
print(f'Full test set size: {len(X_test)} samples')

if len(X_train) > SELECTION_SAMPLE_SIZE:
    print(f'\nCreating stratified subsample of {SELECTION_SAMPLE_SIZE} samples for RFE feature selection...')
    from sklearn.model_selection import train_test_split as subsample_split
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
    X_train_sub, _, y_train_sub, _ = subsample_split(
        X_train, y_train,
        train_size=SELECTION_SAMPLE_SIZE,
        random_state=42,
        stratify=y_train
    )
    print(f'Subsample size: {len(X_train_sub)} samples')
    print(f'Subsample class distribution: {dict(zip(*np.unique(y_train_sub, return_counts=True)))}')
else:
    print(f'\nTraining set is smaller than {SELECTION_SAMPLE_SIZE}, using full set for RFE.')
    X_train_sub = X_train
    y_train_sub = y_train

# Feature Selection Experiment: Find optimal number of features (5-20)
print('\n' + '='*70)
print('FEATURE SELECTION EXPERIMENT: Testing 5-20 features using RFE')
print('='*70)
print('Strategy: RFE on subset → Train & Evaluate on FULL data')
print('='*70 + '\n')

best_accuracy = 0
best_n_features = 0
best_model = None
best_features = []
best_y_pred = None
best_y_test = None
best_f1_score = 0

results = []

# Loop through different numbers of features
for n_features in tqdm(range(5, 21), desc="Testing feature counts"):
    print(f'\n--- Testing k={n_features} features ---')

    # Create LDA estimator for RFE
    lda_estimator = LinearDiscriminantAnalysis()
    
    # Perform RFE to select top k features (using SUBSET for speed)
    print(f'  Running RFE on {len(X_train_sub)} samples...')
    rfe = RFE(estimator=lda_estimator, n_features_to_select=n_features, step=1, verbose=1)
    rfe.fit(X_train_sub, y_train_sub)

    # Get selected features
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
    print(f'  Selected features: {selected_features}')

    # Transform FULL data to use only selected features
    print(f'  Training LDA on FULL training set ({len(X_train)} samples)...')
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    
    # Train a new LDA model on FULL selected features
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_selected, y_train)
    
    # Evaluate on FULL test set
    print(f'  Evaluating on test set ({len(X_test)} samples)...')
    y_pred = lda.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Store results
    results.append({
        'num_features': n_features,
        'accuracy': accuracy,
        'f1_score_weighted': f1_weighted,
        'selected_features': str(selected_features)  # Convert to string for CSV
    })
    
    # Print progress
    print(f'k={n_features}, Accuracy={accuracy:.4f}, F1={f1_weighted:.4f}, Features={selected_features}')

    # Track best model (based on accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_features = n_features
        best_model = (rfe, lda)  # Store both RFE and LDA for the best model
        best_features = selected_features
        best_y_pred = y_pred
        best_y_test = y_test
        best_f1_score = f1_weighted

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(FEATURE_LOG_CSV, index=False)
print(f'\n✓ Results logged to CSV: {FEATURE_LOG_CSV}')

# Print summary of all results
print('\n' + '='*70)
print('EXPERIMENT SUMMARY')
print('='*70)
print(f"{'Features':<12} {'Accuracy':<12} {'F1-Score':<12}")
print('-'*70)
for result in results:
    print(f"{result['num_features']:<12} {result['accuracy']:<12.4f} {result['f1_score_weighted']:<12.4f}")

# Print best model details
print('\n' + '='*70)
print('BEST MODEL FOUND')
print('='*70)
print(f'Optimal number of features: {best_n_features}')
print(f'Best accuracy: {best_accuracy:.4f}')
print(f'Best F1-score (weighted): {best_f1_score:.4f}')
print(f'Selected features: {best_features}')
print('\nClassification Report (Best Model):')
best_class_report = classification_report(best_y_test, best_y_pred)
print(best_class_report)
print('\nConfusion Matrix (Best Model):')
best_conf_matrix = confusion_matrix(best_y_test, best_y_pred)
print(best_conf_matrix)

# Save the best model (both RFE and LDA)
model_package = {
    'rfe': best_model[0],
    'lda': best_model[1],
    'selected_features': best_features,
    'n_features': best_n_features,
    'accuracy': best_accuracy,
    'f1_score_weighted': best_f1_score,
    'feature_columns': feature_columns
}
joblib.dump(model_package, MODEL_OUTPUT_PATH)
print(f'\n✓ Best LDA model saved to: {MODEL_OUTPUT_PATH}')

# Save detailed report for the best model
with open(BEST_MODEL_REPORT, 'w') as f:
    f.write('='*70 + '\n')
    f.write('BEST MODEL REPORT - Feature Selection Experiment\n')
    f.write('='*70 + '\n\n')
    f.write(f'Date: 2025-12-02\n')
    f.write(f'Dataset: {DATASET_PATH}\n')
    f.write(f'Total features available: {len(feature_columns)}\n')
    f.write(f'Feature range tested: 5-20\n')
    f.write(f'Test set size: {len(best_y_test)} samples\n\n')

    f.write('='*70 + '\n')
    f.write('BEST MODEL DETAILS\n')
    f.write('='*70 + '\n')
    f.write(f'Optimal number of features: {best_n_features}\n')
    f.write(f'Accuracy: {best_accuracy:.4f}\n')
    f.write(f'F1-Score (weighted): {best_f1_score:.4f}\n\n')
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

# Create visualization: Number of Features vs Accuracy
plt.figure(figsize=(10, 6))
num_features_list = [r['num_features'] for r in results]
accuracy_list = [r['accuracy'] for r in results]
f1_list = [r['f1_score_weighted'] for r in results]

plt.plot(num_features_list, accuracy_list, marker='o', linewidth=2, markersize=8, label='Accuracy')
plt.plot(num_features_list, f1_list, marker='s', linewidth=2, markersize=8, label='F1-Score (weighted)', alpha=0.7)

# Mark the best point
plt.scatter([best_n_features], [best_accuracy], color='red', s=200, marker='*',
            label=f'Best (k={best_n_features}, Acc={best_accuracy:.4f})', zorder=5)

plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Feature Selection Performance: LDA with RFE', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(range(5, 21))
plt.ylim([min(min(accuracy_list), min(f1_list)) - 0.02, max(max(accuracy_list), max(f1_list)) + 0.02])
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=300, bbox_inches='tight')
plt.close()

print(f'✓ Accuracy plot saved to: {ACCURACY_PLOT}')

print('\n' + '='*70)
print('EXPERIMENT COMPLETE')
print('='*70)
print(f'All results saved to folder: {RESULTS_FOLDER}')
print(f'\nFiles generated:')
print(f'  1. {FEATURE_LOG_CSV} - Detailed metrics for each feature count')
print(f'  2. {MODEL_OUTPUT_PATH} - Best model package')
print(f'  3. {BEST_MODEL_REPORT} - Detailed report for best model')
print(f'  4. {ACCURACY_PLOT} - Performance visualization')
print(f'\nTo use the model:')
print(f'  model_pkg = joblib.load("{MODEL_OUTPUT_PATH}")')
print(f'  X_selected = model_pkg["rfe"].transform(X)')
print(f'  pred = model_pkg["lda"].predict(X_selected)')

