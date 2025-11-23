import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from imblearn.over_sampling import RandomOverSampler

# Path to dataset
DATASET_PATH = r'/src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv'
MODEL_OUTPUT_PATH = r'/src/models/classification/Train/LDA/lda_model_2_class.joblib'
LABEL_COLUMN = 'label'  # Change this if your label column is named differently

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Print available columns for verification
print('Columns in CSV:', df.columns.tolist())

# Use all columns except label and non-feature columns as features
feature_columns = [col for col in df.columns if col not in [LABEL_COLUMN, 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']]
X = df[feature_columns].values

# Option to balance classes
BALANCE_CLASSES = True  # Set to True to balance classes (equal sample size for each group)

# Print original group sizes
group_labels = ['CRACK', 'BACKGROUND', 'BRANCH', 'REGULAR']
group_counts = {g: (df[LABEL_COLUMN] == g).sum() for g in group_labels}
print('Original group sizes:', group_counts)

if BALANCE_CLASSES:
    print('Balancing groups: equal sample size for each group (CRACK, BACKGROUND, BRANCH, REGULAR)')
    min_count = min(group_counts.values())
    balanced_df = pd.concat([
        df[df[LABEL_COLUMN] == g].sample(n=min_count, random_state=42)
        for g in group_labels
    ], ignore_index=True)
    print('Balanced group sizes:', {g: (balanced_df[LABEL_COLUMN] == g).sum() for g in group_labels})
else:
    balanced_df = df.copy()
    print('Group balancing is OFF. Using original group sizes.')

# Convert label column to binary: 'CRACK' vs. 'other'
y = np.where(balanced_df[LABEL_COLUMN] == 'CRACK', 'CRACK', 'other')
X = balanced_df[feature_columns].values

# Print binary class distribution for verification
unique, counts = np.unique(y, return_counts=True)
print('Binary class distribution:', dict(zip(unique, counts)))
if len(unique) != 2:
    raise ValueError(f'Two-class LDA requires exactly 2 classes, found {len(unique)}: {unique}')

# Handle missing values if any
if np.any(pd.isnull(X)) or np.any(pd.isnull(y)):
    print('Warning: Missing values detected. Dropping rows with missing values.')
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Evaluate
y_pred = lda.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(lda, MODEL_OUTPUT_PATH)
print(f'LDA model saved to {MODEL_OUTPUT_PATH}')

