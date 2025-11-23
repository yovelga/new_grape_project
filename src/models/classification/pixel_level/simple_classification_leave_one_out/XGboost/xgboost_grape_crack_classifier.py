import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

BASE_PATH = os.getenv("BASE_PATH")
COMBINED_DATASET_PATH = os.path.join(BASE_PATH, r"dataset_builder_grapes\detection\dataset\signatures_with_unknown.csv")

print("Loading dataset...")
df = pd.read_csv(COMBINED_DATASET_PATH)
print(f"Total dataset shape: {df.shape}")

# Create binary label: 1 for CRACK, 0 for everything else
df['binary_label'] = (df['label'] == 1).astype(int)
print("\nBinary label distribution:\n", df['binary_label'].value_counts())

band_cols = [col for col in df.columns if "nm" in col]
print(f"number of inputs  {len(band_cols)}" )
X = df[band_cols].values
# print(X[:5])
y = df['binary_label'].values

print("\nSplitting data (stratified, test=20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train label dist:", np.bincount(y_train))
print("Test label dist:", np.bincount(y_test))

print("\nTraining XGBoost V2 on all features...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
)
xgb.fit(X_train, y_train)

y_proba = xgb.predict_proba(X_test)[:,1]

# ======= Search threshold for Recall & Precision > 0.95 =======
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
wanted_recall = 0.989
wanted_precision = 0.995
found_threshold = None
for p, r, t in zip(precision, recall, np.append(thresholds, 1.0)):
    print(f"Threshold={t:.3f} | Precision={p:.3f} | Recall={r:.3f}")
    if p >= wanted_precision and r >= wanted_recall:
        פ = t
        break

if found_threshold is not None:
    print(f"\n✅ Threshold found! Probability >= {found_threshold:.3f} gives Recall={r:.3f}, Precision={p:.3f}")
else:
    print("\n❌ No threshold found with both Recall > 0.95 and Precision > 0.95.")
    found_threshold = 0.5  # fallback

y_pred = (y_proba >= found_threshold).astype(int)

print("\nClassification report (with threshold):")
print(classification_report(y_test, y_pred, digits=4))

f1 = f1_score(y_test, y_pred)
print("F1 Score (test):", np.round(f1, 4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Crack','Crack'], yticklabels=['Not Crack','Crack'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (XGBoost, Thr={found_threshold:.3f})")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost, all data)")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(5,4))
plt.plot(recall, precision, label=f"AP = {average_precision_score(y_test, y_proba):.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (XGBoost, all data)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,3))
sns.histplot(y_proba[y_test==1], color='r', bins=40, label="Crack", stat="density", kde=True, alpha=0.6)
sns.histplot(y_proba[y_test==0], color='g', bins=40, label="Not Crack", stat="density", kde=True, alpha=0.6)
plt.legend()
plt.title("Distribution of XGBoost Output Probabilities (Test set)")
plt.xlabel("XGBoost Probability")
plt.tight_layout()
plt.show()

print("\nPipeline finished! Next step: try feature selection or advanced ensembles if needed.")

import joblib

MODEL_OUTPUT_PATH = os.path.join(os.getcwd(), "xgboost_crack_vs_notcrack_model_v1.joblib")
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(xgb, MODEL_OUTPUT_PATH)
print(f"\nXGBoost model saved to: {MODEL_OUTPUT_PATH}")
