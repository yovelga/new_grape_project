"""
Autoencoder-based Anomaly Detection for Crack Detection
=========================================================

THESIS-GRADE IMPLEMENTATION
---------------------------

This script implements a one-class classification approach using an autoencoder
trained exclusively on NORMAL (healthy) grape spectral signatures. Anomalies
(cracks) are detected based on high reconstruction error.

WHY ANOMALY DETECTION?
----------------------
Anomaly detection complements supervised classification by:
1. Not requiring labeled anomaly samples during training (one-class learning)
2. Being robust to novel anomaly types not seen during training
3. Providing interpretable anomaly scores (reconstruction error)
4. Working well when anomalies are rare (class imbalance)

WHY balanced=False?
-------------------
For anomaly detection, we use the FULL unbalanced dataset because:
1. The autoencoder only trains on NORMAL samples (ignores anomaly count)
2. Test evaluation should reflect real-world class distribution
3. Balancing would artificially alter the detection task

WHY threshold = Nth percentile?
--------------------------------
The threshold is set as a high percentile (default: 97.5th) of TRAINING NORMAL errors because:
1. It's determined WITHOUT using anomaly labels (true one-class approach)
2. High percentile (e.g., 97.5%) means we expect ~2.5% false positive rate on normal samples
3. It's a conservative threshold that balances precision/recall
4. It's reproducible and doesn't require validation set tuning
5. CRITICAL: Calculated AFTER 5% outlier removal for cleaner baseline
6. The exact percentile is configurable via THRESHOLD_PERCENTILE constant

ROBUST TRAINING (NEW):
----------------------
Before training the autoencoder, we clean the Normal training data:
1. Use IsolationForest with contamination=0.05 to remove the 5% most anomalous samples
2. This ensures the autoencoder learns from a cleaner, more representative baseline
3. The threshold is then calculated from this cleaned training set
4. This improves model robustness and reduces false positives

METHODOLOGY
-----------
For each LOGO fold:
1. Fit StandardScaler on training data only (prevent data leakage)
2. Extract training NORMAL samples (y=0)
3. **NEW**: Remove 5% outliers from training normal using IsolationForest
4. Train autoencoder ONLY on cleaned training NORMAL samples
5. Compute reconstruction errors on test set (normal + anomaly)
6. Set threshold = THRESHOLD_PERCENTILE (default 97.5th) of cleaned training normal errors
7. Classify: error > threshold → ANOMALY

Labels:
- y == 0: NORMAL (Healthy/Regular)
- y == 1: ANOMALY (Crack)

Author: Refactored for thesis reproducibility
Date: 2026-01-17
"""
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]

import os
import random
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve, roc_curve,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

# Import the reusable preprocessing function
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from src.preprocessing.spectral_preprocessing import preprocess_pixel_level_dataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# REPRODUCIBILITY: Set all random seeds
# =============================================================================
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_SEED)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Random seed: {RANDOM_SEED}")

# =============================================================================
# CONFIGURATION (FIXED for anomaly detection)
# =============================================================================
THRESHOLD_PERCENTILE = 99  # Percentile of training normal errors for threshold
WL_MIN = 450
WL_MAX = 925
APPLY_SNV = True
BALANCED = False  # CRITICAL: Use full unbalanced dataset


# =============================================================================
# Autoencoder Model
# =============================================================================
class SpectralAutoencoder(nn.Module):
    """
    3-layer Autoencoder for spectral data anomaly detection.

    Architecture:
        Encoder: input -> h1 -> h2 -> h3 (bottleneck)
        Decoder: h3 -> h2 -> h1 -> input (reconstruction)

    The model learns to reconstruct normal spectral signatures.
    Anomalies (cracks) produce higher reconstruction errors because
    the model has never learned to reconstruct their patterns.
    """
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int, int] = (64, 32, 16)):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.hidden_dims = hidden_dims

        # Encoder: progressively compress the input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.2),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.2),

            nn.Linear(h2, h3),
            nn.ReLU(),
        )

        # Decoder: reconstruct from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(h3, h2),
            nn.ReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.2),

            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.2),

            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE reconstruction error per sample.
        Higher error indicates the sample is less like the training data (anomaly).
        """
        self.eval()
        with torch.no_grad():
            x = x.to(next(self.parameters()).device)  # Ensure same device
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=1)
        return mse


# =============================================================================
# Training Function
# =============================================================================
def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    verbose: bool = False
) -> Tuple[float, List[float]]:
    """
    Train autoencoder with early stopping on reconstruction loss.

    Args:
        model: SpectralAutoencoder instance
        train_loader: DataLoader with NORMAL samples only
        epochs: Maximum training epochs
        lr: Learning rate
        patience: Early stopping patience
        verbose: Print training progress

    Returns:
        Tuple of (best_loss, loss_history)
    """
    model.to(DEVICE)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for batch_x, in train_loader:
            batch_x = batch_x.to(DEVICE)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}, best loss: {best_loss:.6f}")
                break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    return best_loss, loss_history


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pr_auc: float,
    fold_idx: int,
    group_name: str,
    save_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot and save Precision-Recall curve for a fold.

    Returns:
        Tuple of (precision, recall) arrays for aggregation
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Fold {fold_idx} (Group: {group_name})', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return precision, recall


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    roc_auc: float,
    fold_idx: int,
    group_name: str,
    save_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot and save ROC curve for a fold.

    Returns:
        Tuple of (fpr, tpr) arrays for aggregation
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Fold {fold_idx} (Group: {group_name})', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fpr, tpr


def plot_reconstruction_error_distribution(
    errors_normal: np.ndarray,
    errors_anomaly: np.ndarray,
    threshold: float,
    fold_idx: int,
    group_name: str,
    save_path: Path
) -> None:
    """
    Plot reconstruction error distributions for normal vs anomaly samples.
    Generates TWO versions: one with log scale Y-axis, one with linear scale.
    """
    # Determine bin range
    all_errors = np.concatenate([errors_normal, errors_anomaly]) if len(errors_anomaly) > 0 else errors_normal
    bins = np.linspace(all_errors.min(), min(all_errors.max(), np.percentile(all_errors, 99)), 50)

    # --- Version 1: Log Scale Y-axis ---
    plt.figure(figsize=(10, 6))
    plt.hist(errors_anomaly, bins=bins, alpha=0.6, label=f'Crack (n={len(errors_anomaly)})',
             color='red', density=True)
    if len(errors_normal) > 0:
        plt.hist(errors_normal, bins=bins, alpha=0.6, label=f'Regular (n={len(errors_normal)})',
                 color='green', density=True)

    # Plot threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold ({THRESHOLD_PERCENTILE}th pctl) = {threshold:.4f}')

    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Density (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.title(f'Reconstruction Error Distribution - Fold {fold_idx} (Group: {group_name}) - Log Scale', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # --- Version 2: Linear Scale Y-axis ---
    linear_path = save_path.parent / f"{save_path.stem}_linear{save_path.suffix}"
    plt.figure(figsize=(10, 6))
    plt.hist(errors_anomaly, bins=bins, alpha=0.6, label=f'Crack (n={len(errors_anomaly)})',
             color='red', density=True)
    if len(errors_normal) > 0:
        plt.hist(errors_normal, bins=bins, alpha=0.6, label=f'Regular (n={len(errors_normal)})',
                 color='green', density=True)

    # Plot threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold ({THRESHOLD_PERCENTILE}th pctl) = {threshold:.4f}')

    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Reconstruction Error Distribution - Fold {fold_idx} (Group: {group_name}) - Linear Scale', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(linear_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregated_pr_curve(
    all_precisions: List[np.ndarray],
    all_recalls: List[np.ndarray],
    mean_pr_auc: float,
    std_pr_auc: float,
    save_path: Path
) -> None:
    """
    Plot aggregated mean PR curve from all folds with std band.

    Aggregation method:
    - Interpolate each fold's PR curve to a common recall axis (0 to 1, 100 points)
    - Compute mean and std of precision values at each recall point across all LOGO folds
    - Plot mean curve with ±1 std shaded band
    """
    plt.figure(figsize=(10, 7))

    # Plot individual fold curves in light color
    for precision, recall in zip(all_precisions, all_recalls):
        plt.plot(recall, precision, alpha=0.2, color='blue', linewidth=1)

    # Interpolate to common recall values for mean curve
    # Each fold's curve is interpolated to 100 evenly-spaced recall points
    mean_recall = np.linspace(0, 1, 100)
    interpolated_precisions = []

    for precision, recall in zip(all_precisions, all_recalls):
        sorted_idx = np.argsort(recall)
        recall_sorted = recall[sorted_idx]
        precision_sorted = precision[sorted_idx]
        interp_precision = np.interp(mean_recall, recall_sorted, precision_sorted)
        interpolated_precisions.append(interp_precision)

    mean_precision = np.mean(interpolated_precisions, axis=0)
    std_precision = np.std(interpolated_precisions, axis=0)

    plt.plot(mean_recall, mean_precision, 'b-', linewidth=2.5,
             label=f'Mean PR curve (AUC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})')
    plt.fill_between(mean_recall,
                     np.maximum(mean_precision - std_precision, 0),
                     np.minimum(mean_precision + std_precision, 1),
                     alpha=0.2, color='blue')

    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Aggregated Precision-Recall Curve (All Folds)', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregated_roc_curve(
    all_fprs: List[np.ndarray],
    all_tprs: List[np.ndarray],
    mean_roc_auc: float,
    std_roc_auc: float,
    save_path: Path
) -> None:
    """
    Plot aggregated mean ROC curve from all folds with std band.

    Aggregation method:
    - Interpolate each fold's ROC curve to a common FPR axis (0 to 1, 100 points)
    - Compute mean and std of TPR values at each FPR point across all LOGO folds
    - Plot mean curve with ±1 std shaded band
    """
    plt.figure(figsize=(10, 7))

    # Plot individual fold curves in light color
    for fpr, tpr in zip(all_fprs, all_tprs):
        plt.plot(fpr, tpr, alpha=0.2, color='blue', linewidth=1)

    # Interpolate to common FPR values for mean curve
    # Each fold's curve is interpolated to 100 evenly-spaced FPR points
    mean_fpr = np.linspace(0, 1, 100)
    interpolated_tprs = []

    for fpr, tpr in zip(all_fprs, all_tprs):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    std_tpr = np.std(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0

    plt.plot(mean_fpr, mean_tpr, 'b-', linewidth=2.5,
             label=f'Mean ROC curve (AUC = {mean_roc_auc:.4f} ± {std_roc_auc:.4f})')
    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     alpha=0.2, color='blue')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Aggregated ROC Curve (All Folds)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_boxplot_and_violin(
    all_errors_normal: List[np.ndarray],
    all_errors_anomaly: List[np.ndarray],
    boxplot_path: Path,
    violin_path: Path
) -> None:
    """
    Plot boxplot and violin plot of reconstruction errors across all folds.
    Saves FOUR separate figures: boxplot (log), boxplot (linear), violin (log), violin (linear).
    """
    # Concatenate all errors
    errors_normal_all = np.concatenate(all_errors_normal)
    errors_anomaly_all = np.concatenate(all_errors_anomaly) if all_errors_anomaly else np.array([])

    # Prepare data and labels - Crack (anomaly) first, Regular (normal) second
    data_box = []
    labels_box = []
    colors = ['red', 'green']  # Crack=red, Regular=green

    if len(errors_anomaly_all) > 0:
        data_box.append(errors_anomaly_all)
        labels_box.append(f'Crack\n(n={len(errors_anomaly_all)})')

    data_box.append(errors_normal_all)
    labels_box.append(f'Regular\n(n={len(errors_normal_all)})')

    # --- Figure 1: Boxplot (Log Scale) ---
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data_box, labels=labels_box, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors[:len(data_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.yscale('log')
    plt.ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    plt.title('Reconstruction Error Distribution (Boxplot - Log Scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Boxplot (Linear Scale) ---
    boxplot_linear_path = boxplot_path.parent / f"{boxplot_path.stem}_linear{boxplot_path.suffix}"
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data_box, labels=labels_box, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors[:len(data_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
    plt.title('Reconstruction Error Distribution (Boxplot - Linear Scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(boxplot_linear_path, dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 3: Violin plot (Log Scale) ---
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot(data_box, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i] if i < len(colors) else 'blue')
        pc.set_alpha(0.6)
    plt.xticks(range(1, len(data_box) + 1), labels_box)
    plt.yscale('log')
    plt.ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    plt.title('Reconstruction Error Distribution (Violin Plot - Log Scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(violin_path, dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 4: Violin plot (Linear Scale) ---
    violin_linear_path = violin_path.parent / f"{violin_path.stem}_linear{violin_path.suffix}"
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot(data_box, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i] if i < len(colors) else 'blue')
        pc.set_alpha(0.6)
    plt.xticks(range(1, len(data_box) + 1), labels_box)
    plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
    plt.title('Reconstruction Error Distribution (Violin Plot - Linear Scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(violin_linear_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_autoencoder_logo(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    hidden_dims: Tuple[int, int, int] = (64, 32, 16),
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate autoencoder using Leave-One-Group-Out cross-validation.

    Methodology per fold:
    1. Fit StandardScaler on training data only (no data leakage)
    2. Extract training NORMAL samples (y=0)
    3. **Remove 5% outliers from training normal using IsolationForest (contamination=0.05)**
    4. Train autoencoder ONLY on cleaned training NORMAL samples
    5. Compute reconstruction errors on test set (normal + anomaly)
    6. Set threshold = THRESHOLD_PERCENTILE (default 97.5th) of CLEANED TRAINING NORMAL errors
    7. Classify: error > threshold → ANOMALY (crack)

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (0=normal, 1=anomaly)
        groups: Group labels for LOGO CV
        hidden_dims: Autoencoder hidden layer dimensions
        epochs: Training epochs
        batch_size: Training batch size
        lr: Learning rate
        output_dir: Directory to save plots
        save_plots: Whether to save plots
        verbose: Print detailed per-fold logging

    Returns:
        Tuple of (fold_results_df, summary_dict)
    """
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    input_dim = X.shape[1]

    # Storage for fold-level results
    fold_results = []

    # Storage for aggregated plotting
    all_y_true = []
    all_y_scores = []
    all_precisions = []
    all_recalls = []
    all_fprs = []
    all_tprs = []
    all_errors_test_normal = []
    all_errors_test_anomaly = []

    print(f"\n{'='*70}")
    print(f"[INFO] Starting LOGO CV with {n_splits} folds...")
    print(f"[INFO] Threshold method: {THRESHOLD_PERCENTILE}th percentile of training normal errors")
    print(f"[INFO] Model architecture: {hidden_dims}")
    print(f"{'='*70}")

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(
        logo.split(X, y, groups), total=n_splits, desc="Autoencoder LOGO CV"
    )):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        group_test = groups[test_idx][0]  # Test group name

        # =================================================================
        # Step 1: Fit scaler on training data only (no data leakage)
        # =================================================================
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # =================================================================
        # Step 2: Train only on NORMAL (y=0) samples
        # =================================================================
        normal_mask_train = y_train == 0
        X_train_normal = X_train_scaled[normal_mask_train]
        n_train_normal = len(X_train_normal)
        n_train_anomaly = int(np.sum(y_train == 1))

        if n_train_normal < 10:
            print(f"  [WARN] Fold {fold_idx} (Group: {group_test}): Only {n_train_normal} normal samples, skipping")
            continue

        # =================================================================
        # Step 2.5: Robust Training - Remove outliers from Normal training data
        # Use IsolationForest to remove the 5% most anomalous 'normal' samples
        # This ensures the autoencoder learns a cleaner baseline
        # =================================================================
        iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED, n_jobs=-1)
        outlier_predictions = iso_forest.fit_predict(X_train_normal)
        # Keep only inliers (prediction == 1)
        inlier_mask = outlier_predictions == 1
        X_train_normal_cleaned = X_train_normal[inlier_mask]
        n_outliers_removed = n_train_normal - len(X_train_normal_cleaned)

        if verbose:
            print(f"  [INFO] Fold {fold_idx}: Removed {n_outliers_removed}/{n_train_normal} outliers from training normal data ({n_outliers_removed/n_train_normal*100:.1f}%)")

        train_dataset = TensorDataset(torch.from_numpy(X_train_normal_cleaned))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=0)

        # Initialize and train model
        set_seed(RANDOM_SEED + fold_idx)  # Different seed per fold for variety
        model = SpectralAutoencoder(input_dim, hidden_dims)

        start_train = time.time()
        best_loss, _ = train_autoencoder(model, train_loader, epochs=epochs, lr=lr, patience=15, verbose=False)
        train_time = time.time() - start_train

        # =================================================================
        # Step 3: Compute reconstruction errors
        # =================================================================
        # Explicitly ensure model is on correct device before evaluation
        model.to(DEVICE)
        model.eval()
        start_infer = time.time()

        # Errors on training normal (CLEANED - for threshold calculation)
        # CRITICAL: Use the cleaned training data for threshold, not the original
        X_train_normal_cleaned_tensor = torch.from_numpy(X_train_normal_cleaned).to(DEVICE)
        errors_train_normal = model.get_reconstruction_error(X_train_normal_cleaned_tensor).cpu().numpy()

        # Errors on test set
        X_test_tensor = torch.from_numpy(X_test_scaled).to(DEVICE)
        errors_test = model.get_reconstruction_error(X_test_tensor).cpu().numpy()

        infer_time = time.time() - start_infer

        # Separate test errors by class
        errors_test_normal = errors_test[y_test == 0]
        errors_test_anomaly = errors_test[y_test == 1]

        # Store for aggregated plots
        all_errors_test_normal.append(errors_test_normal)
        if len(errors_test_anomaly) > 0:
            all_errors_test_anomaly.append(errors_test_anomaly)

        # =================================================================
        # Step 4: Set threshold (THRESHOLD_PERCENTILE of CLEANED training normal errors)
        # CRITICAL: No labeled anomalies used for threshold selection!
        # Threshold calculated from cleaned training data (after outlier removal)
        # Uses configurable THRESHOLD_PERCENTILE constant (default: 97.5)
        # =================================================================
        threshold = float(np.percentile(errors_train_normal, THRESHOLD_PERCENTILE))

        # =================================================================
        # Step 5: Classify test samples
        # =================================================================
        y_scores = errors_test  # Anomaly score = reconstruction error
        y_pred = (errors_test > threshold).astype(int)

        # Store for aggregated analysis
        all_y_true.append(y_test)
        all_y_scores.append(y_scores)

        # =================================================================
        # Step 6: Compute all metrics
        # =================================================================
        n_test_normal = int(np.sum(y_test == 0))
        n_test_anomaly = int(np.sum(y_test == 1))

        # Safely extract confusion matrix with labels=[0,1] to ensure 2x2 shape
        # This handles cases where test fold has only one class or predictions are all same
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        # cm is always 2x2: [[TN, FP], [FN, TP]]
        # Safe extraction that works even if one class is missing
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])

        # Classification metrics (anomaly/crack as positive class)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        # Ranking metrics (require both classes in test set)
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_scores)
            pr_auc = average_precision_score(y_test, y_scores)
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        # Error statistics
        error_stats = {
            'train_normal_error_mean': float(np.mean(errors_train_normal)),
            'train_normal_error_std': float(np.std(errors_train_normal)),
            'test_normal_error_mean': float(np.mean(errors_test_normal)) if len(errors_test_normal) > 0 else np.nan,
            'test_normal_error_std': float(np.std(errors_test_normal)) if len(errors_test_normal) > 0 else np.nan,
            'test_anomaly_error_mean': float(np.mean(errors_test_anomaly)) if len(errors_test_anomaly) > 0 else np.nan,
            'test_anomaly_error_std': float(np.std(errors_test_anomaly)) if len(errors_test_anomaly) > 0 else np.nan,
        }

        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'test_group': group_test,
            'n_train_normal': n_train_normal,
            'n_train_anomaly': n_train_anomaly,
            'n_test_normal': n_test_normal,
            'n_test_anomaly': n_test_anomaly,
            'threshold': threshold,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            **error_stats,
            'train_loss': best_loss,
            'train_time_s': train_time,
            'infer_time_s': infer_time,
        }
        fold_results.append(fold_result)

        # =================================================================
        # Step 7: Verbose logging (per-fold summary)
        # =================================================================
        if verbose:
            print(f"\n  ┌─────────────────────────────────────────────────────────────")
            print(f"  │ [Fold {fold_idx}] Test Group: {group_test}")
            print(f"  ├─────────────────────────────────────────────────────────────")
            print(f"  │ Test samples:  {n_test_normal} NORMAL, {n_test_anomaly} ANOMALY")
            print(f"  │ Threshold ({THRESHOLD_PERCENTILE}th pctl of train normal): {threshold:.6f}")
            print(f"  ├─────────────────────────────────────────────────────────────")
            if not np.isnan(roc_auc):
                print(f"  │ ROC-AUC: {roc_auc:.4f}    PR-AUC: {pr_auc:.4f}")
            else:
                print(f"  │ ROC-AUC: N/A (single class)    PR-AUC: N/A")
            print(f"  │ Precision: {prec:.4f}    Recall: {rec:.4f}    F1: {f1:.4f}")
            print(f"  │ Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  └─────────────────────────────────────────────────────────────")

        # =================================================================
        # Step 8: Save per-fold plots
        # =================================================================
        if save_plots and output_dir is not None:
            fold_plot_dir = output_dir / "per_fold_plots"
            fold_plot_dir.mkdir(parents=True, exist_ok=True)

            # Reconstruction error distribution plot
            plot_reconstruction_error_distribution(
                errors_test_normal, errors_test_anomaly, threshold,
                fold_idx, str(group_test),
                fold_plot_dir / f"error_distribution_fold_{fold_idx:02d}.png"
            )

            # PR curve and ROC curve (only if both classes present)
            if len(np.unique(y_test)) > 1:
                precision_arr, recall_arr = plot_precision_recall_curve(
                    y_test, y_scores, pr_auc,
                    fold_idx, str(group_test),
                    fold_plot_dir / f"pr_curve_fold_{fold_idx:02d}.png"
                )
                all_precisions.append(precision_arr)
                all_recalls.append(recall_arr)

                fpr_arr, tpr_arr = plot_roc_curve(
                    y_test, y_scores, roc_auc,
                    fold_idx, str(group_test),
                    fold_plot_dir / f"roc_curve_fold_{fold_idx:02d}.png"
                )
                all_fprs.append(fpr_arr)
                all_tprs.append(tpr_arr)

    # =================================================================
    # Aggregate results
    # =================================================================
    df_folds = pd.DataFrame(fold_results)

    # Compute summary statistics (mean ± std)
    metrics_to_aggregate = [
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc',
        'threshold', 'train_normal_error_mean', 'test_normal_error_mean', 'test_anomaly_error_mean',
        'train_time_s', 'infer_time_s'
    ]

    summary = {
        'model_name': f"Autoencoder ({hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]})",
        'n_folds': len(fold_results),
        'threshold_percentile': THRESHOLD_PERCENTILE,
        'total_TP': int(df_folds['TP'].sum()),
        'total_FP': int(df_folds['FP'].sum()),
        'total_TN': int(df_folds['TN'].sum()),
        'total_FN': int(df_folds['FN'].sum()),
    }

    for metric in metrics_to_aggregate:
        if metric in df_folds.columns:
            values = df_folds[metric].dropna()
            summary[f'mean_{metric}'] = float(values.mean()) if len(values) > 0 else np.nan
            summary[f'std_{metric}'] = float(values.std()) if len(values) > 0 else np.nan

    # =================================================================
    # Save aggregated plots
    # =================================================================
    if save_plots and output_dir is not None:
        # Aggregated PR curve
        if len(all_precisions) > 0:
            plot_aggregated_pr_curve(
                all_precisions, all_recalls,
                summary.get('mean_pr_auc', np.nan),
                summary.get('std_pr_auc', np.nan),
                output_dir / "aggregated_pr_curve.png"
            )

        # Aggregated ROC curve
        if len(all_fprs) > 0:
            plot_aggregated_roc_curve(
                all_fprs, all_tprs,
                summary.get('mean_roc_auc', np.nan),
                summary.get('std_roc_auc', np.nan),
                output_dir / "aggregated_roc_curve.png"
            )

        # Boxplot and violin plot of errors (two separate figures)
        if len(all_errors_test_normal) > 0:
            plot_error_boxplot_and_violin(
                all_errors_test_normal,
                all_errors_test_anomaly,
                output_dir / "reconstruction_error_boxplot.png",
                output_dir / "reconstruction_error_violin.png"
            )

    return df_folds, summary


# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Main execution function for autoencoder anomaly detection experiment.

    This function orchestrates the complete experiment pipeline:
    1. Load and preprocess data (balanced=False for anomaly detection)
    2. Run LOGO cross-validation for each autoencoder configuration
    3. Compute and save comprehensive metrics
    4. Generate all required plots
    5. Output thesis-ready results tables
    """
    # =========================================================================
    # Configuration
    # =========================================================================
    CSV_PATH = Path(str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_2026-01-13.csv"))
    # Save experiments under the centralized experiments directory with a clear experiment name
    # User-requested base experiments directory:
    EXPERIMENTS_BASE = Path(str(_PROJECT_ROOT / r"experiments"))
    # Create a descriptive experiment name (readable, includes date/time)
    # Allow overriding via environment variable EXPERIMENT_NAME for reproducibility
    env_name = os.environ.get('EXPERIMENT_NAME')
    if env_name and len(env_name.strip()) > 0:
        # sanitize: replace spaces with underscores and remove problematic chars
        safe_name = env_name.strip().replace(' ', '_')
        EXPERIMENT_NAME = ''.join(c for c in safe_name if c.isalnum() or c in ('-', '_'))
    else:
        EXPERIMENT_NAME = f"autoencoder_one_class_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    OUTPUT_DIR = EXPERIMENTS_BASE / EXPERIMENT_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Experiment name: {EXPERIMENT_NAME}")
    print(f"[INFO] Writing outputs to: {OUTPUT_DIR}")

    # Autoencoder configurations to evaluate
    CONFIGS = [
        {"hidden_dims": (64, 32, 16), "epochs": 100, "batch_size": 256, "lr": 1e-3},
        {"hidden_dims": (128, 64, 32), "epochs": 100, "batch_size": 256, "lr": 1e-3},
        {"hidden_dims": (64, 32, 8), "epochs": 100, "batch_size": 256, "lr": 1e-3},
    ]

    # =========================================================================
    # Header
    # =========================================================================
    print(f"\n{'='*80}")
    print("   AUTOENCODER ANOMALY DETECTION FOR CRACK DETECTION")
    print("   One-Class Classification with LOGO Cross-Validation")
    print("   THESIS-GRADE IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"\n[CONFIGURATION]")
    print(f"  Random seed:        {RANDOM_SEED}")
    print(f"  Threshold:          {THRESHOLD_PERCENTILE}th percentile of training normal errors")
    print(f"  Wavelength range:   {WL_MIN}-{WL_MAX} nm")
    print(f"  SNV normalization:  {APPLY_SNV}")
    print(f"  Balanced dataset:   {BALANCED} (using FULL unbalanced data)")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print(f"  Device:             {DEVICE}")

    # =========================================================================
    # Load and preprocess data
    # IMPORTANT: balanced=False for anomaly detection (use full dataset)
    # =========================================================================
    print(f"\n{'='*80}")
    print("[DATA LOADING]")
    print(f"{'='*80}")
    print(f"  Source: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    X, y, groups, feature_names = preprocess_pixel_level_dataset(
        df,
        wl_min=WL_MIN,
        wl_max=WL_MAX,
        apply_snv=APPLY_SNV,
        remove_outliers=False,
        balanced=BALANCED,  # CRITICAL: Use full unbalanced dataset for anomaly detection
    )

    print(f"\n[DATA SUMMARY]")
    print(f"  Total samples:      {X.shape[0]:,}")
    print(f"  Features:           {X.shape[1]} (wavelengths)")
    print(f"  Groups:             {len(np.unique(groups))}")
    print(f"  Class distribution:")
    print(f"    - Normal (y=0):   {np.sum(y==0):,} ({100*np.sum(y==0)/len(y):.1f}%)")
    print(f"    - Anomaly (y=1):  {np.sum(y==1):,} ({100*np.sum(y==1)/len(y):.1f}%)")
    print(f"  Imbalance ratio:    {np.sum(y==0)/max(np.sum(y==1), 1):.1f}:1 (normal:anomaly)")

    # =========================================================================
    # Run experiments for each configuration
    # =========================================================================
    all_fold_results = []
    all_summaries = []

    for config_idx, config in enumerate(CONFIGS):
        config_name = f"{config['hidden_dims'][0]}-{config['hidden_dims'][1]}-{config['hidden_dims'][2]}"
        config_dir = OUTPUT_DIR / f"config_{config_name}"
        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"[EXPERIMENT {config_idx+1}/{len(CONFIGS)}] Autoencoder: {config_name}")
        print(f"{'='*80}")
        print(f"  Hidden dimensions: {config['hidden_dims']}")
        print(f"  Epochs:            {config['epochs']}")
        print(f"  Batch size:        {config['batch_size']}")
        print(f"  Learning rate:     {config['lr']}")

        try:
            df_folds, summary = evaluate_autoencoder_logo(
                X, y, groups,
                hidden_dims=config["hidden_dims"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                lr=config["lr"],
                output_dir=config_dir,
                save_plots=True,
                verbose=True,
            )

            # Add config identifier to fold results
            df_folds['config'] = config_name
            all_fold_results.append(df_folds)
            all_summaries.append(summary)

            # Print summary for this config
            print(f"\n{'='*60}")
            print(f"[RESULTS SUMMARY] Config: {config_name}")
            print(f"{'='*60}")
            print(f"  Accuracy:     {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            print(f"  Precision:    {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
            print(f"  Recall:       {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
            print(f"  F1 Score:     {summary['mean_f1_score']:.4f} ± {summary['std_f1_score']:.4f}")
            print(f"  ROC-AUC:      {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}")
            print(f"  PR-AUC:       {summary['mean_pr_auc']:.4f} ± {summary['std_pr_auc']:.4f}")
            print(f"  Threshold:    {summary['mean_threshold']:.6f} ± {summary['std_threshold']:.6f}")
            print(f"  Confusion (total): TP={summary['total_TP']}, FP={summary['total_FP']}, TN={summary['total_TN']}, FN={summary['total_FN']}")

            # Save fold-level results for this config
            df_folds.to_csv(config_dir / "fold_results.csv", index=False)
            df_folds.to_excel(config_dir / "fold_results.xlsx", index=False)

        except Exception as e:
            print(f"[ERROR] Config {config_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Save combined results
    # =========================================================================
    if all_fold_results:
        # Combine all fold results
        df_all_folds = pd.concat(all_fold_results, ignore_index=True)
        df_all_folds.to_csv(OUTPUT_DIR / "all_fold_results.csv", index=False)
        df_all_folds.to_excel(OUTPUT_DIR / "all_fold_results.xlsx", index=False)

        # Create summary dataframe
        df_summary = pd.DataFrame(all_summaries)

        # Round numeric columns for readability
        numeric_cols = df_summary.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_summary[col] = df_summary[col].round(4)

        df_summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)
        df_summary.to_excel(OUTPUT_DIR / "summary_results.xlsx", index=False)

        # =====================================================================
        # Print final summary
        # =====================================================================
        print(f"\n{'='*80}")
        print("   FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"\nAll configurations (mean ± std across folds):\n")

        # Create a clean display table
        display_cols = ['model_name', 'n_folds', 'mean_accuracy', 'std_accuracy',
                        'mean_precision', 'std_precision', 'mean_recall', 'std_recall',
                        'mean_f1_score', 'std_f1_score', 'mean_roc_auc', 'std_roc_auc',
                        'mean_pr_auc', 'std_pr_auc']
        display_df = df_summary[[c for c in display_cols if c in df_summary.columns]]
        print(display_df.to_string())

        print(f"\n{'='*80}")
        print("   OUTPUT FILES")
        print(f"{'='*80}")
        print(f"  [DATA]")
        print(f"    - Per-fold metrics:     {OUTPUT_DIR / 'all_fold_results.xlsx'}")
        print(f"    - Summary metrics:      {OUTPUT_DIR / 'summary_results.xlsx'}")
        print(f"  [PLOTS]")
        print(f"    - Aggregated PR curve:  {OUTPUT_DIR / 'config_*/aggregated_pr_curve.png'}")
        print(f"    - Aggregated ROC curve: {OUTPUT_DIR / 'config_*/aggregated_roc_curve.png'}")
        print(f"    - Error boxplot:        {OUTPUT_DIR / 'config_*/reconstruction_error_boxplot.png'}")
        print(f"    - Error violin:         {OUTPUT_DIR / 'config_*/reconstruction_error_violin.png'}")
        print(f"    - Per-fold plots:       {OUTPUT_DIR / 'config_*/per_fold_plots/'}")

        # Print best configuration
        best_idx = df_summary['mean_f1_score'].idxmax()
        best_config = df_summary.loc[best_idx]
        print(f"\n{'='*80}")
        print(f"[BEST CONFIGURATION by F1 Score]")
        print(f"{'='*80}")
        print(f"  Model:      {best_config['model_name']}")
        print(f"  F1 Score:   {best_config['mean_f1_score']:.4f} ± {best_config.get('std_f1_score', 0):.4f}")
        print(f"  PR-AUC:     {best_config['mean_pr_auc']:.4f} ± {best_config.get('std_pr_auc', 0):.4f}")
        print(f"  ROC-AUC:    {best_config['mean_roc_auc']:.4f} ± {best_config.get('std_roc_auc', 0):.4f}")
        print(f"  Precision:  {best_config['mean_precision']:.4f} ± {best_config.get('std_precision', 0):.4f}")
        print(f"  Recall:     {best_config['mean_recall']:.4f} ± {best_config.get('std_recall', 0):.4f}")

    else:
        print("[ERROR] No results to save!")

    print(f"\n{'='*80}")
    print("[DONE] Experiment completed successfully.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

