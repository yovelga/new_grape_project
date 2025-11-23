import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# Suppress convergence warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# 1. Load environment and data
load_dotenv(dotenv_path=os.getenv('ENV_PATH', r'/.env'))
BASE_PATH = Path(os.getenv('BASE_PATH'))
DATA_PATH = os.getenv('DATASET_FOR_TRAIN_PATH')

# 2) Build every other file/dir from that single root
CSV_DIR = BASE_PATH / DATA_PATH
df = pd.read_csv(CSV_DIR)
spectral_cols = df.columns[6:-3]
wavelengths = [float(col.rstrip("nm")) for col in spectral_cols]
X = df[spectral_cols].values
y = df["label"].values  # 0 = regular, 1 = cracked

X_reg = X[y == 0]
X_cr = X[y == 1]
X_reg_train, X_reg_test = train_test_split(X_reg, train_size=0.2, random_state=42)
X_test = np.vstack([X_reg_test, X_cr])
y_test = np.hstack([np.zeros(len(X_reg_test)), np.ones(len(X_cr))])

# 2. Standardize
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_reg_train)
# X_test_scaled  = scaler.transform(X_test)

# 3. Define & train AE, record loss each epoch
ae = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 64, 128),
    activation="relu",
    solver="adam",
    max_iter=1,
    warm_start=True,
    random_state=42,
)
train_losses = []
for _ in tqdm(range(500), desc="Training Epochs"):
    ae.fit(X_reg_train, X_reg_train)
    train_losses.append(ae.loss_)

joblib.dump(ae, "autoencoder_trained_model.joblib")
print("Autoencoder saved as autoencoder_trained_model.joblib")

# 4. Compute reconstruction error
reconstructed_test = ae.predict(X_test)
re_error = np.mean((X_test - reconstructed_test) ** 2, axis=1)

# 4.1. Spectral signature plots
#  - Predict separately for healthy test set and cracked set
reconstructed_reg = ae.predict(X_reg_test)
reconstructed_cr = ae.predict(X_cr)

# Line plot of 3 healthy signatures
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(
        wavelengths, X_reg_test[i], linestyle="-", label=f"Healthy Original #{i+1}"
    )
    plt.plot(
        wavelengths,
        reconstructed_reg[i],
        linestyle="--",
        label=f"Healthy Reconstructed #{i+1}",
    )
    plt.title("Line Plot: Original vs Reconstructed Spectral Signatures (Healthy)")
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.3)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Line plot of 3 cracked signatures
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(wavelengths, X_cr[i], linestyle="-", label=f"Cracked Original #{i+1}")
    plt.plot(
        wavelengths,
        reconstructed_cr[i],
        linestyle="--",
        label=f"Cracked Reconstructed #{i+1}",
    )
    plt.title("Line Plot: Original vs Reconstructed Spectral Signatures (Cracked)")
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.3)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 5. Sweep thresholds for accuracy
thresholds = np.arange(0.0, 0.0021, 0.0001)
acc_scores = [
    accuracy_score(y_test, (re_error > thr).astype(int)) for thr in thresholds
]
best_idx = np.argmax(acc_scores)
best_thr = thresholds[best_idx]
best_acc = acc_scores[best_idx]
print(f"Optimal Threshold = {best_thr:.4f} → Accuracy = {best_acc:.3f}")

# 6. ROC & AUC
fpr, tpr, _ = roc_curve(y_test, re_error)
roc_auc = auc(fpr, tpr)

# 7. Precision–Recall & AP
precision, recall, _ = precision_recall_curve(y_test, re_error)
avg_prec = average_precision_score(y_test, re_error)

# 7.1 Training Loss per Epoch
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.2 Error Distribution (Histogram)
plt.figure(figsize=(6, 4))
plt.hist(re_error[y_test == 0], bins=50, alpha=0.6, label="Regular", color="blue")
plt.hist(re_error[y_test == 1], bins=50, alpha=0.6, label="Cracked", color="red")
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.3 Boxplot by Class
plt.figure(figsize=(5, 4))
bp = plt.boxplot(
    [re_error[y_test == 0], re_error[y_test == 1]],
    tick_labels=["Regular", "Cracked"],
    patch_artist=True,
)
for patch, color in zip(bp["boxes"], ["lightblue", "lightcoral"]):
    patch.set_facecolor(color)
plt.title("Reconstruction Error by Class (Boxplot)")
plt.ylabel("Reconstruction Error")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# 7.4 Accuracy vs. Reconstruction Threshold
plt.figure(figsize=(6, 4))
plt.plot(thresholds, acc_scores, marker=".", label="Accuracy")
plt.axvline(
    best_thr,
    color="red",
    linestyle="--",
    label=f"Best Thr = {best_thr:.4f}\nAcc = {best_acc:.3f}",
)
plt.title("Accuracy vs. Reconstruction Threshold")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.5 ROC Curve for Reconstruction Error
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve for Reconstruction Error")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.6 Precision–Recall Curve
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, lw=2, label=f"Average Precision = {avg_prec:.3f}")
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()
