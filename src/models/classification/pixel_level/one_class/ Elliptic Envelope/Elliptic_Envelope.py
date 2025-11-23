import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import joblib
import os
from dotenv import load_dotenv
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 1. Load environment and data
load_dotenv(dotenv_path=os.getenv('ENV_PATH', r'/.env'))
BASE_PATH = Path(os.getenv('BASE_PATH'))
DATA_PATH = os.getenv('DATASET_FOR_TRAIN_PATH')
CSV_DIR = BASE_PATH / DATA_PATH

print(f"Loading data from: {CSV_DIR}")
df = pd.read_csv(CSV_DIR)
print(f"Data shape: {df.shape}")

# 2. Feature selection
spectral_cols = df.columns[6:-3]
X = df[spectral_cols].values
y = df["label"].values  # 0 = regular, 1 = crack

X_cr = X[y == 1]
X_reg = X[y == 0]

# 3. Balance classes (downsample regulars)
n = min(len(X_cr), len(X_reg))
X_cr_bal = resample(X_cr, replace=False, n_samples=n, random_state=42)
X_reg_bal = resample(X_reg, replace=False, n_samples=n, random_state=42)
print(f"Balanced: Crack: {len(X_cr_bal)}, Regular: {len(X_reg_bal)}")

# 4. Split - Train/Val/Test
# נאמן רק על Crack, נבדוק על Crack + Regular
X_cr_train, X_cr_val = train_test_split(X_cr_bal, train_size=0.7, random_state=42)
X_test = np.vstack([X_cr_val, X_reg_bal])
y_test = np.hstack([np.ones(len(X_cr_val)), np.zeros(len(X_reg_bal))])
print(f"Train crack: {X_cr_train.shape}, Val crack: {X_cr_val.shape}, Test regular: {X_reg_bal.shape}")

# 5. Standardization
scaler = StandardScaler()
X_cr_train_scaled = scaler.fit_transform(X_cr_train)
X_cr_val_scaled = scaler.transform(X_cr_val)
X_test_scaled = scaler.transform(X_test)

# 6. Train EllipticEnvelope על Crack בלבד
ee = EllipticEnvelope(contamination=0.1, random_state=42)
ee.fit(X_cr_train_scaled)
print("Model fitted.")

# 7. Evaluation and confusion matrix
def eval_and_plot(model, X_data, y_true, name='Test'):
    print(f"--- {name} ---")
    y_pred = model.predict(X_data)
    y_pred_bin = (y_pred == 1).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin)
    rec = recall_score(y_true, y_pred_bin)
    cm = confusion_matrix(y_true, y_pred_bin)
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print("Confusion Matrix:\n", cm)
    scores = -model.decision_function(X_data)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    avg_prec = average_precision_score(y_true, scores)
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average Precision: {avg_prec:.3f}")

    plt.figure()
    plt.hist(scores[y_true==1], bins=40, alpha=0.6, label="Crack")
    plt.hist(scores[y_true==0], bins=40, alpha=0.6, label="Regular")
    plt.title(f"Decision Function Score Distribution ({name})")
    plt.xlabel("Anomaly Score (lower is more anomalous)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve ({name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"AP = {avg_prec:.3f}")
    plt.title(f"Precision–Recall Curve ({name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "roc_auc": roc_auc,
        "avg_prec": avg_prec,
        "cm": cm
    }

metrics_test = eval_and_plot(ee, X_test_scaled, y_test, name="Test (Balanced)")

# 8. Save model pipeline and metrics
joblib.dump({
    "model": ee,
    "scaler": scaler,
    "spectral_cols": list(spectral_cols),
    "metrics_test": metrics_test,
}, "elliptic_envelope_balanced_pipeline.joblib")
print("Model pipeline saved as elliptic_envelope_balanced_pipeline.joblib")

pd.DataFrame([metrics_test]).to_excel("elliptic_envelope_balanced_metrics.xlsx", index=False)
print("All done. Results saved.")
