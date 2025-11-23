import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

# Path to dataset
DATASET_PATH = r'/src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv'
MODEL_OUTPUT_PATH = r'/src/models/classification/Train/LDA/lda_model_multi_class.joblib'
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

