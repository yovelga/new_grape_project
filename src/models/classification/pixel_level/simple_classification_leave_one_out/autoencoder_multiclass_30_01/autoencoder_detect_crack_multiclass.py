"""
Autoencoder-based Anomaly Detection for CRACK Detection (Multiclass)
=====================================================================

THESIS-GRADE IMPLEMENTATION
---------------------------

This script implements a one-class classification approach using an autoencoder
trained on ALL classes EXCEPT CRACK. The autoencoder learns the "normal"
spectral patterns of all non-crack classes, and CRACK samples are detected
as anomalies based on high reconstruction error.

EXPERIMENT DESIGN:
------------------
- Training: All classes EXCEPT CRACK (9 classes: BACKGROUND, BRANCH, BURNT_PIXEL,
            IRON, LEAF, PLASTIC, REGULAR, TRIPOD, WHITE_REFERENCE)
- Test: All classes including CRACK
- Anomaly: CRACK class (high reconstruction error expected)
- Normal: All other classes (low reconstruction error expected)

WHY TRAIN/TEST SPLIT INSTEAD OF LOGO?
--------------------------------------
LOGO (Leave-One-Group-Out) cross-validation is not suitable here because:
1. Not all images contain all 10 classes
2. When leaving out one image, some classes may be entirely missing from train/test
3. This leads to inconsistent training and unreliable evaluation

Instead, we use a SINGLE MODEL trained on a stratified train/test split:
1. Split data 80/20 with stratification by groups (images)
2. Train ONE autoencoder on ALL non-CRACK samples from training set
3. Evaluate on the test set which contains all classes

METHODOLOGY:
------------
1. Split data into train/test (80/20, stratified by groups)
2. Fit StandardScaler on training data only (prevent data leakage)
3. Extract training NORMAL samples (all classes except CRACK)
4. Remove 5% outliers from training normal using IsolationForest
5. Train autoencoder ONLY on cleaned training NORMAL samples
6. Compute reconstruction errors on test set (all classes)
7. Set threshold = THRESHOLD_PERCENTILE of cleaned training normal errors
8. Classify: error > threshold → ANOMALY (predicted CRACK)

Labels:
- y == 0: NORMAL (all classes except CRACK)
- y == 1: ANOMALY (CRACK)

Author: Thesis experiment - Multiclass Autoencoder for Crack Detection
Date: 2026-01-30
"""

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

from sklearn.model_selection import train_test_split
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
# CONFIGURATION
# =============================================================================
THRESHOLD_PERCENTILE = 99  # Percentile of training normal errors for threshold
OUTLIER_CONTAMINATION = 0.0  # Set to 0 to disable outlier removal, or e.g. 0.05 for 5%
WL_MIN = 450
WL_MAX = 925
APPLY_SNV = True
BALANCED = True  # Use balanced dataset

# Target anomaly class
ANOMALY_CLASS = "CRACK"

# All classes in the multiclass dataset
ALL_CLASSES = [
    "BACKGROUND", "BRANCH", "BURNT_PIXEL", "CRACK", "IRON",
    "LEAF", "PLASTIC", "REGULAR", "TRIPOD", "WHITE_REFERENCE"
]

# Normal classes (all except CRACK)
NORMAL_CLASSES = [c for c in ALL_CLASSES if c != ANOMALY_CLASS]


# =============================================================================
# Autoencoder Model
# =============================================================================
class SpectralAutoencoder(nn.Module):
    """
    3-layer Autoencoder for spectral data anomaly detection.

    Architecture:
        Encoder: input -> h1 -> h2 -> h3 (bottleneck)
        Decoder: h3 -> h2 -> h1 -> input (reconstruction)

    The model learns to reconstruct normal spectral signatures from all
    non-crack classes. CRACK samples produce higher reconstruction errors.
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
            x = x.to(next(self.parameters()).device)
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
        train_loader: DataLoader with NORMAL samples only (all except CRACK)
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
    """Plot and save Precision-Recall curve for a fold."""
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
    """Plot and save ROC curve for a fold."""
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
    all_errors = np.concatenate([errors_normal, errors_anomaly]) if len(errors_anomaly) > 0 else errors_normal
    bins = np.linspace(all_errors.min(), min(all_errors.max(), np.percentile(all_errors, 99)), 50)

    # --- Version 1: Log Scale Y-axis ---
    plt.figure(figsize=(10, 6))
    plt.hist(errors_anomaly, bins=bins, alpha=0.6, label=f'CRACK (n={len(errors_anomaly)})',
             color='red', density=True)
    if len(errors_normal) > 0:
        plt.hist(errors_normal, bins=bins, alpha=0.6, label=f'Non-CRACK (n={len(errors_normal)})',
                 color='green', density=True)

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
    plt.hist(errors_anomaly, bins=bins, alpha=0.6, label=f'CRACK (n={len(errors_anomaly)})',
             color='red', density=True)
    if len(errors_normal) > 0:
        plt.hist(errors_normal, bins=bins, alpha=0.6, label=f'Non-CRACK (n={len(errors_normal)})',
                 color='green', density=True)

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
    """Plot aggregated mean PR curve from all folds with std band."""
    plt.figure(figsize=(10, 7))

    for precision, recall in zip(all_precisions, all_recalls):
        plt.plot(recall, precision, alpha=0.2, color='blue', linewidth=1)

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
    plt.title('Aggregated Precision-Recall Curve (All Folds) - CRACK Detection', fontsize=14)
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
    """Plot aggregated mean ROC curve from all folds with std band."""
    plt.figure(figsize=(10, 7))

    for fpr, tpr in zip(all_fprs, all_tprs):
        plt.plot(fpr, tpr, alpha=0.2, color='blue', linewidth=1)

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
    plt.title('Aggregated ROC Curve (All Folds) - CRACK Detection', fontsize=14)
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
    """
    errors_normal_all = np.concatenate(all_errors_normal)
    errors_anomaly_all = np.concatenate(all_errors_anomaly) if all_errors_anomaly else np.array([])

    data_box = []
    labels_box = []
    colors = ['red', 'green']

    if len(errors_anomaly_all) > 0:
        data_box.append(errors_anomaly_all)
        labels_box.append(f'CRACK\n(n={len(errors_anomaly_all)})')

    data_box.append(errors_normal_all)
    labels_box.append(f'Non-CRACK\n(n={len(errors_normal_all)})')

    # --- Boxplot (Log Scale) ---
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

    # --- Boxplot (Linear Scale) ---
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

    # --- Violin plot (Log Scale) ---
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

    # --- Violin plot (Linear Scale) ---
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


def plot_per_class_error_distribution(
    class_errors: Dict[str, np.ndarray],
    save_path: Path
) -> None:
    """
    Plot reconstruction error distribution for each original class.
    This helps understand how well the autoencoder reconstructs each class.
    """
    plt.figure(figsize=(14, 8))
    
    # Sort classes by median error
    sorted_classes = sorted(class_errors.keys(), 
                           key=lambda c: np.median(class_errors[c]) if len(class_errors[c]) > 0 else 0)
    
    data = [class_errors[c] for c in sorted_classes]
    labels = [f'{c}\n(n={len(class_errors[c])})' for c in sorted_classes]
    
    # Color CRACK differently
    colors = ['red' if c == ANOMALY_CLASS else 'steelblue' for c in sorted_classes]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.yscale('log')
    plt.ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title('Reconstruction Error by Original Class\n(Red = CRACK, Blue = Non-CRACK)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Data Loading Function for Multiclass
# =============================================================================
def load_multiclass_data(
    csv_path: Path,
    wl_min: float = 450,
    wl_max: float = 925,
    apply_snv: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load multiclass dataset and prepare for anomaly detection.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y_binary: Binary labels (0=Normal/non-crack, 1=Anomaly/crack)
        y_original: Original string labels for per-class analysis
        groups: Group labels for LOGO CV
        feature_names: List of wavelength feature names
    """
    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[INFO] Original dataset shape: {df.shape}")
    print(f"[INFO] Class distribution:")
    print(df['label'].value_counts().to_string())
    
    # Store original labels
    y_original = df['label'].values.copy()
    
    # Create binary labels: CRACK = 1 (anomaly), everything else = 0 (normal)
    df['binary_label'] = (df['label'] == ANOMALY_CLASS).astype(int)
    
    # Use the preprocessing function with binary labels
    X, y_binary, groups, feature_names = preprocess_pixel_level_dataset(
        df,
        wl_min=wl_min,
        wl_max=wl_max,
        apply_snv=apply_snv,
        remove_outliers=False,
        balanced=False,
        label_col='binary_label',
        label_map={0: 0, 1: 1},  # Already binary
    )
    
    return X, y_binary, y_original, groups, feature_names


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_autoencoder_single_model(
    X: np.ndarray,
    y: np.ndarray,
    y_original: np.ndarray,
    groups: np.ndarray,
    hidden_dims: Tuple[int, int, int] = (64, 32, 16),
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    test_size: float = 0.2,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate autoencoder using a single train/test split.
    
    WHY NOT LOGO CV?
    - Not all images contain all 10 classes
    - LOGO would leave some classes missing from train or test
    - Single model trained on pooled data is more robust
    
    Training: All classes EXCEPT CRACK
    Test: All classes including CRACK
    Anomaly detection: CRACK (y=1) vs Non-CRACK (y=0)
    """
    input_dim = X.shape[1]
    
    # =================================================================
    # Split data: stratify by groups to ensure good distribution
    # =================================================================
    # Get unique groups and split them
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(
        unique_groups, test_size=test_size, random_state=RANDOM_SEED
    )
    
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    y_orig_train, y_orig_test = y_original[train_mask], y_original[test_mask]
    
    print(f"\n{'='*70}")
    print(f"[INFO] Training SINGLE MODEL (no LOGO CV)")
    print(f"[INFO] Reason: Not all images contain all classes")
    print(f"[INFO] Train/Test split: {100*(1-test_size):.0f}%/{100*test_size:.0f}%")
    print(f"[INFO] Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
    print(f"[INFO] Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    print(f"[INFO] Training on: {NORMAL_CLASSES}")
    print(f"[INFO] Detecting anomaly: {ANOMALY_CLASS}")
    print(f"[INFO] Threshold method: {THRESHOLD_PERCENTILE}th percentile of training normal errors")
    print(f"[INFO] Model architecture: {hidden_dims}")
    print(f"{'='*70}")
    
    # Print class distribution in train/test
    print(f"\n[CLASS DISTRIBUTION]")
    print(f"  Training set:")
    for cls in ALL_CLASSES:
        n = np.sum(y_orig_train == cls)
        print(f"    - {cls}: {n:,}")
    print(f"  Test set:")
    for cls in ALL_CLASSES:
        n = np.sum(y_orig_test == cls)
        print(f"    - {cls}: {n:,}")

    # =================================================================
    # Step 1: Fit scaler on training data only
    # =================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # =================================================================
    # Step 2: Train only on NORMAL (y=0) samples (all except CRACK)
    # =================================================================
    normal_mask_train = y_train == 0
    X_train_normal = X_train_scaled[normal_mask_train]
    n_train_normal = len(X_train_normal)
    n_train_anomaly = int(np.sum(y_train == 1))

    print(f"\n[TRAINING DATA]")
    print(f"  Normal samples (non-CRACK): {n_train_normal:,}")
    print(f"  Anomaly samples (CRACK): {n_train_anomaly:,}")

    # =================================================================
    # Step 2.5: Remove outliers from Normal training data (if enabled)
    # =================================================================
    if OUTLIER_CONTAMINATION > 0:
        iso_forest = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=RANDOM_SEED, n_jobs=-1)
        outlier_predictions = iso_forest.fit_predict(X_train_normal)
        inlier_mask = outlier_predictions == 1
        X_train_normal_cleaned = X_train_normal[inlier_mask]
        n_outliers_removed = n_train_normal - len(X_train_normal_cleaned)

        if verbose:
            print(f"  Removed {n_outliers_removed}/{n_train_normal} outliers ({n_outliers_removed/n_train_normal*100:.1f}%)")
            print(f"  Final training samples: {len(X_train_normal_cleaned):,}")
    else:
        X_train_normal_cleaned = X_train_normal
        if verbose:
            print(f"  Outlier removal: DISABLED")
            print(f"  Final training samples: {len(X_train_normal_cleaned):,}")

    train_dataset = TensorDataset(torch.from_numpy(X_train_normal_cleaned))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)

    # Initialize and train model
    set_seed(RANDOM_SEED)
    model = SpectralAutoencoder(input_dim, hidden_dims)

    print(f"\n[TRAINING]")
    start_train = time.time()
    best_loss, _ = train_autoencoder(model, train_loader, epochs=epochs, lr=lr, patience=15, verbose=verbose)
    train_time = time.time() - start_train
    print(f"  Training completed in {train_time:.1f}s, best loss: {best_loss:.6f}")

    # =================================================================
    # Step 3: Compute reconstruction errors
    # =================================================================
    model.to(DEVICE)
    model.eval()
    start_infer = time.time()

    X_train_normal_cleaned_tensor = torch.from_numpy(X_train_normal_cleaned).to(DEVICE)
    errors_train_normal = model.get_reconstruction_error(X_train_normal_cleaned_tensor).cpu().numpy()

    X_test_tensor = torch.from_numpy(X_test_scaled).to(DEVICE)
    errors_test = model.get_reconstruction_error(X_test_tensor).cpu().numpy()

    infer_time = time.time() - start_infer

    # Separate test errors by binary class
    errors_test_normal = errors_test[y_test == 0]
    errors_test_anomaly = errors_test[y_test == 1]

    # Store per-class errors for analysis
    class_errors = {}
    for cls in ALL_CLASSES:
        cls_mask = y_orig_test == cls
        if np.sum(cls_mask) > 0:
            class_errors[cls] = errors_test[cls_mask]
        else:
            class_errors[cls] = np.array([])

    # =================================================================
    # Step 4: Set threshold
    # =================================================================
    threshold = float(np.percentile(errors_train_normal, THRESHOLD_PERCENTILE))

    # =================================================================
    # Step 5: Classify test samples
    # =================================================================
    y_scores = errors_test
    y_pred = (errors_test > threshold).astype(int)

    # =================================================================
    # Step 6: Compute all metrics
    # =================================================================
    n_test_normal = int(np.sum(y_test == 0))
    n_test_anomaly = int(np.sum(y_test == 1))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    error_stats = {
        'train_normal_error_mean': float(np.mean(errors_train_normal)),
        'train_normal_error_std': float(np.std(errors_train_normal)),
        'test_normal_error_mean': float(np.mean(errors_test_normal)) if len(errors_test_normal) > 0 else np.nan,
        'test_normal_error_std': float(np.std(errors_test_normal)) if len(errors_test_normal) > 0 else np.nan,
        'test_anomaly_error_mean': float(np.mean(errors_test_anomaly)) if len(errors_test_anomaly) > 0 else np.nan,
        'test_anomaly_error_std': float(np.std(errors_test_anomaly)) if len(errors_test_anomaly) > 0 else np.nan,
    }

    # Build results
    result = {
        'n_train_groups': len(train_groups),
        'n_test_groups': len(test_groups),
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

    if verbose:
        print(f"\n{'='*70}")
        print(f"[RESULTS]")
        print(f"{'='*70}")
        print(f"  Test samples:  {n_test_normal:,} Non-CRACK, {n_test_anomaly:,} CRACK")
        print(f"  Threshold ({THRESHOLD_PERCENTILE}th pctl of train normal): {threshold:.6f}")
        print(f"  ROC-AUC: {roc_auc:.4f}    PR-AUC: {pr_auc:.4f}")
        print(f"  Precision: {prec:.4f}    Recall: {rec:.4f}    F1: {f1:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP={tp:,} (CRACK correctly detected)")
        print(f"    FP={fp:,} (Non-CRACK incorrectly flagged as CRACK)")
        print(f"    TN={tn:,} (Non-CRACK correctly classified)")
        print(f"    FN={fn:,} (CRACK missed)")

    # =================================================================
    # Save plots
    # =================================================================
    if save_plots and output_dir is not None:
        # Error distribution plot
        plot_reconstruction_error_distribution(
            errors_test_normal, errors_test_anomaly, threshold,
            0, "all_groups",
            output_dir / "error_distribution.png"
        )

        # PR curve
        if len(np.unique(y_test)) > 1:
            plot_precision_recall_curve(
                y_test, y_scores, pr_auc,
                0, "single_model",
                output_dir / "pr_curve.png"
            )

            plot_roc_curve(
                y_test, y_scores, roc_auc,
                0, "single_model",
                output_dir / "roc_curve.png"
            )

        # Boxplot and violin
        plot_error_boxplot_and_violin(
            [errors_test_normal],
            [errors_test_anomaly] if len(errors_test_anomaly) > 0 else [],
            output_dir / "reconstruction_error_boxplot.png",
            output_dir / "reconstruction_error_violin.png"
        )

        # Per-class error distribution
        plot_per_class_error_distribution(
            class_errors,
            output_dir / "per_class_error_distribution.png"
        )

    # Build summary
    summary = {
        'model_name': f"Autoencoder ({hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]})",
        'experiment_type': 'detect_crack_multiclass_single_model',
        'training_classes': ', '.join(NORMAL_CLASSES),
        'anomaly_class': ANOMALY_CLASS,
        'test_size': test_size,
        'threshold_percentile': THRESHOLD_PERCENTILE,
        **result,
    }

    df_result = pd.DataFrame([result])

    return df_result, summary


# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Main execution function for autoencoder anomaly detection experiment.
    
    Experiment: Train on ALL classes EXCEPT CRACK → Detect CRACK as anomaly
    """
    # =========================================================================
    # Configuration
    # =========================================================================
    CSV_PATH = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_multiclass_2026-01-16.csv")
    
    EXPERIMENTS_BASE = Path(r"C:\Users\yovel\Desktop\Grape_Project\experiments")
    env_name = os.environ.get('EXPERIMENT_NAME')
    if env_name and len(env_name.strip()) > 0:
        safe_name = env_name.strip().replace(' ', '_')
        EXPERIMENT_NAME = ''.join(c for c in safe_name if c.isalnum() or c in ('-', '_'))
    else:
        EXPERIMENT_NAME = f"autoencoder_detect_crack_multiclass_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    OUTPUT_DIR = EXPERIMENTS_BASE / EXPERIMENT_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    print("   AUTOENCODER ANOMALY DETECTION FOR CRACK DETECTION (MULTICLASS)")
    print("   Single Model with Train/Test Split (NOT LOGO CV)")
    print("   TRAINING: All classes EXCEPT CRACK")
    print("   DETECTING: CRACK as anomaly")
    print(f"{'='*80}")
    print(f"\n[CONFIGURATION]")
    print(f"  Experiment name:    {EXPERIMENT_NAME}")
    print(f"  Random seed:        {RANDOM_SEED}")
    print(f"  Threshold:          {THRESHOLD_PERCENTILE}th percentile of training normal errors")
    print(f"  Wavelength range:   {WL_MIN}-{WL_MAX} nm")
    print(f"  SNV normalization:  {APPLY_SNV}")
    print(f"  Balanced dataset:   {BALANCED}")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print(f"  Device:             {DEVICE}")
    print(f"  Training classes:   {NORMAL_CLASSES}")
    print(f"  Anomaly class:      {ANOMALY_CLASS}")

    # =========================================================================
    # Load and preprocess data
    # =========================================================================
    print(f"\n{'='*80}")
    print("[DATA LOADING]")
    print(f"{'='*80}")
    print(f"  Source: {CSV_PATH}")

    X, y, y_original, groups, feature_names = load_multiclass_data(
        CSV_PATH,
        wl_min=WL_MIN,
        wl_max=WL_MAX,
        apply_snv=APPLY_SNV,
    )

    print(f"\n[DATA SUMMARY]")
    print(f"  Total samples:      {X.shape[0]:,}")
    print(f"  Features:           {X.shape[1]} (wavelengths)")
    print(f"  Groups:             {len(np.unique(groups))}")
    print(f"  Binary class distribution:")
    print(f"    - Normal (y=0):   {np.sum(y==0):,} ({100*np.sum(y==0)/len(y):.1f}%) [Non-CRACK]")
    print(f"    - Anomaly (y=1):  {np.sum(y==1):,} ({100*np.sum(y==1)/len(y):.1f}%) [CRACK]")
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

        try:
            df_result, summary = evaluate_autoencoder_single_model(
                X, y, y_original, groups,
                hidden_dims=config["hidden_dims"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                lr=config["lr"],
                test_size=0.2,
                output_dir=config_dir,
                save_plots=True,
                verbose=True,
            )

            df_result['config'] = config_name
            all_fold_results.append(df_result)
            all_summaries.append(summary)

            df_result.to_csv(config_dir / "results.csv", index=False)
            df_result.to_excel(config_dir / "results.xlsx", index=False)

        except Exception as e:
            print(f"[ERROR] Config {config_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Save combined results
    # =========================================================================
    if all_fold_results:
        df_all_results = pd.concat(all_fold_results, ignore_index=True)
        df_all_results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
        df_all_results.to_excel(OUTPUT_DIR / "all_results.xlsx", index=False)

        df_summary = pd.DataFrame(all_summaries)
        numeric_cols = df_summary.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_summary[col] = df_summary[col].round(4)

        df_summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)
        df_summary.to_excel(OUTPUT_DIR / "summary_results.xlsx", index=False)

        print(f"\n{'='*80}")
        print("   FINAL RESULTS SUMMARY")
        print(f"{'='*80}")

        display_cols = ['model_name', 'accuracy', 'precision', 'recall', 
                        'f1_score', 'roc_auc', 'pr_auc']
        display_df = df_summary[[c for c in display_cols if c in df_summary.columns]]
        print(display_df.to_string())

        best_idx = df_summary['f1_score'].idxmax()
        best_config = df_summary.loc[best_idx]
        print(f"\n{'='*80}")
        print(f"[BEST CONFIGURATION by F1 Score]")
        print(f"{'='*80}")
        print(f"  Model:      {best_config['model_name']}")
        print(f"  F1 Score:   {best_config['f1_score']:.4f}")
        print(f"  PR-AUC:     {best_config['pr_auc']:.4f}")
        print(f"  ROC-AUC:    {best_config['roc_auc']:.4f}")

    print(f"\n{'='*80}")
    print("[DONE] Experiment completed successfully.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
