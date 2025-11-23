import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Load environment and data
load_dotenv(dotenv_path=os.getenv('ENV_PATH', r'/.env'))
BASE_PATH = Path(os.getenv('BASE_PATH'))
DATA_PATH = os.getenv('DATASET_FOR_TRAIN_PATH')
CSV_DIR = BASE_PATH / DATA_PATH
df = pd.read_csv(CSV_DIR)
spectral_cols = df.columns[6:-3]
wavelengths = [float(col.rstrip("nm")) for col in spectral_cols]
X = df[spectral_cols].values.astype(np.float32)
y = df["label"].values  # 0 = regular, 1 = cracked

# *** KEY CHANGE: train on cracks (label==1), not regular ***
X_cr = X[y == 1]
X_reg = X[y == 0]

# Train/test split
X_cr_train, X_cr_test = train_test_split(X_cr, train_size=0.8, random_state=42)
X_test = np.vstack([X_cr_test, X_reg])
y_test = np.hstack([np.ones(len(X_cr_test)), np.zeros(len(X_reg))])

# Choose device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# 2. PyTorch Autoencoder Definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

input_dim = X.shape[1]
autoencoder = Autoencoder(input_dim).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 3. Training
n_epochs = 500
batch_size = 1024
train_losses = []
X_cr_train_tensor = torch.from_numpy(X_cr_train).float().to(device)
dataset_size = X_cr_train_tensor.size(0)

for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
    autoencoder.train()
    perm = torch.randperm(dataset_size)
    batch_losses = []
    for i in range(0, dataset_size, batch_size):
        idx = perm[i:i + batch_size]
        batch = X_cr_train_tensor[idx]
        optimizer.zero_grad()
        outputs = autoencoder(batch)
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

# Save model
torch.save(autoencoder.state_dict(), "autoencoder_trained_model_crack.pt")
print("Autoencoder saved as autoencoder_trained_model_crack.pt")

# 4. Compute reconstruction error
def compute_reconstruction(model, X_data, device, batch_size=1024):
    model.eval()
    errors = []
    recon = []
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_data).float().to(device)
        for i in range(0, X_tensor.size(0), batch_size):
            batch = X_tensor[i:i+batch_size]
            out = model(batch)
            recon.append(out.cpu().numpy())
            err = torch.mean((batch - out) ** 2, dim=1)
            errors.append(err.cpu().numpy())
    recon = np.vstack(recon)
    errors = np.concatenate(errors)
    return recon, errors

reconstructed_test, re_error = compute_reconstruction(autoencoder, X_test, device)
reconstructed_cr, _ = compute_reconstruction(autoencoder, X_cr_test, device)
reconstructed_reg, _ = compute_reconstruction(autoencoder, X_reg, device)

# 4.1. Spectral signature plots (Crack)
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(wavelengths, X_cr_test[i], linestyle="-", label=f"Crack Original #{i+1}")
    plt.plot(wavelengths, reconstructed_cr[i], linestyle="--", label=f"Crack Reconstructed #{i+1}")
plt.title("Line Plot: Original vs Reconstructed Spectral Signatures (Crack)")
plt.xlabel("Spectral Band Index")
plt.ylabel("Reflectance")
plt.ylim(0, 1.3)
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4.2. Spectral signature plots (Regular)
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(wavelengths, X_reg[i], linestyle="-", label=f"Regular Original #{i+1}")
    plt.plot(wavelengths, reconstructed_reg[i], linestyle="--", label=f"Regular Reconstructed #{i+1}")
plt.title("Line Plot: Original vs Reconstructed Spectral Signatures (Regular)")
plt.xlabel("Spectral Band Index")
plt.ylabel("Reflectance")
plt.ylim(0, 1.3)
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Sweep thresholds for accuracy
thresholds = np.arange(0.0, 0.1, 0.0001)
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
