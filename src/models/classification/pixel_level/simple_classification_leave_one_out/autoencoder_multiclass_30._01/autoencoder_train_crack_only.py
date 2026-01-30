"""
Autoencoder-based Anomaly Detection - Train ONLY on CRACK (Multiclass)
=======================================================================

THESIS-GRADE IMPLEMENTATION
---------------------------

This script implements a one-class classification approach using an autoencoder
trained ONLY on CRACK samples. The autoencoder learns the spectral patterns
of cracks, and all other classes (non-crack) are detected as anomalies based
on high reconstruction error.

EXPERIMENT DESIGN:
------------------
- Training: ONLY CRACK class
- Test: All 10 classes
- Normal (for autoencoder): CRACK (low reconstruction error expected)
- Anomaly: All other classes (high reconstruction error expected)

LOGO CROSS-VALIDATION OVER CRACK-CONTAINING GROUPS:
----------------------------------------------------
Since only some images contain CRACK samples, we use a modified LOGO approach:
1. Find all groups (images) that contain CRACK samples
2. For each CRACK-containing group (fold):
   - Train autoencoder on CRACK samples from ALL OTHER CRACK groups
   - Test on ALL samples from the held-out group
3. This ensures no data leakage while properly evaluating on each CRACK group

METHODOLOGY:
------------
1. Identify groups that contain CRACK samples (e.g., 19 groups)
2. For each fold (CRACK-containing group):
   a. Fit StandardScaler on training CRACK data only
   b. Remove 5% outliers from training CRACK using IsolationForest
   c. Train autoencoder ONLY on cleaned training CRACK samples
   d. Compute reconstruction errors on held-out test group (all classes)
   e. Set threshold = THRESHOLD_PERCENTILE of training CRACK errors
   f. Classify: error > threshold → ANOMALY (predicted non-CRACK)
3. Aggregate results across all folds

Labels (INVERTED for this experiment):
- y == 0: NORMAL for autoencoder (CRACK) - what we train on
- y == 1: ANOMALY (all other classes)

Author: Thesis experiment - Autoencoder trained on CRACK only
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

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve, roc_curve,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

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
THRESHOLD_PERCENTILE = 99  # Percentile of training CRACK errors for threshold
OUTLIER_CONTAMINATION = 0.0  # Set to 0 to disable outlier removal, or e.g. 0.05 for 5%
WL_MIN = 450
WL_MAX = 925
APPLY_SNV = True
BALANCED = False  # Use full unbalanced dataset for anomaly detection

# Training class (what autoencoder learns to reconstruct)
TRAINING_CLASS = "CRACK"

# All classes in the multiclass dataset
ALL_CLASSES = [
    "BACKGROUND", "BRANCH", "BURNT_PIXEL", "CRACK", "IRON",
    "LEAF", "PLASTIC", "REGULAR", "TRIPOD", "WHITE_REFERENCE"
]

# Anomaly classes (all except CRACK)
ANOMALY_CLASSES = [c for c in ALL_CLASSES if c != TRAINING_CLASS]


# =============================================================================
# Autoencoder Model
# =============================================================================
class SpectralAutoencoder(nn.Module):
    """
    3-layer Autoencoder for spectral data anomaly detection.

    Architecture:
        Encoder: input -> h1 -> h2 -> h3 (bottleneck)
        Decoder: h3 -> h2 -> h1 -> input (reconstruction)

    The model learns to reconstruct CRACK spectral signatures.
    Non-CRACK samples produce higher reconstruction errors.
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
        Higher error indicates the sample is less like CRACK (the training data).
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
        train_loader: DataLoader with CRACK samples only
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
    errors_crack: np.ndarray,
    errors_non_crack: np.ndarray,
    threshold: float,
    fold_idx: int,
    group_name: str,
    save_path: Path
) -> None:
    """
    Plot reconstruction error distributions for CRACK vs non-CRACK samples.
    Note: In this experiment, CRACK = training class (low error expected)
                              non-CRACK = anomaly (high error expected)
    """
    all_errors = np.concatenate([errors_crack, errors_non_crack]) if len(errors_non_crack) > 0 else errors_crack
    bins = np.linspace(all_errors.min(), min(all_errors.max(), np.percentile(all_errors, 99)), 50)

    # --- Version 1: Log Scale Y-axis ---
    plt.figure(figsize=(10, 6))
    if len(errors_non_crack) > 0:
        plt.hist(errors_non_crack, bins=bins, alpha=0.6, label=f'Non-CRACK (n={len(errors_non_crack)})',
                 color='blue', density=True)
    plt.hist(errors_crack, bins=bins, alpha=0.6, label=f'CRACK (n={len(errors_crack)})',
             color='red', density=True)

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
    if len(errors_non_crack) > 0:
        plt.hist(errors_non_crack, bins=bins, alpha=0.6, label=f'Non-CRACK (n={len(errors_non_crack)})',
                 color='blue', density=True)
    plt.hist(errors_crack, bins=bins, alpha=0.6, label=f'CRACK (n={len(errors_crack)})',
             color='red', density=True)

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
    plt.title('Aggregated Precision-Recall Curve (All Folds) - Non-CRACK Detection', fontsize=14)
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
    plt.title('Aggregated ROC Curve (All Folds) - Non-CRACK Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_boxplot_and_violin(
    all_errors_crack: List[np.ndarray],
    all_errors_non_crack: List[np.ndarray],
    boxplot_path: Path,
    violin_path: Path
) -> None:
    """
    Plot boxplot and violin plot of reconstruction errors across all folds.
    Note: CRACK = training class, Non-CRACK = anomaly
    """
    errors_crack_all = np.concatenate(all_errors_crack)
    errors_non_crack_all = np.concatenate(all_errors_non_crack) if all_errors_non_crack else np.array([])

    data_box = []
    labels_box = []
    colors = ['blue', 'red']  # Non-CRACK=blue, CRACK=red

    if len(errors_non_crack_all) > 0:
        data_box.append(errors_non_crack_all)
        labels_box.append(f'Non-CRACK\n(n={len(errors_non_crack_all)})')

    data_box.append(errors_crack_all)
    labels_box.append(f'CRACK\n(n={len(errors_crack_all)})')

    # --- Boxplot (Log Scale) ---
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data_box, labels=labels_box, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors[:len(data_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.yscale('log')
    plt.ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    plt.title('Reconstruction Error Distribution (Boxplot - Log Scale)\nTrained on CRACK only', fontsize=14)
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
    plt.title('Reconstruction Error Distribution (Boxplot - Linear Scale)\nTrained on CRACK only', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(boxplot_linear_path, dpi=150, bbox_inches='tight')
    plt.close()

    # --- Violin plot (Log Scale) ---
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot(data_box, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
        pc.set_alpha(0.6)
    plt.xticks(range(1, len(data_box) + 1), labels_box)
    plt.yscale('log')
    plt.ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    plt.title('Reconstruction Error Distribution (Violin Plot - Log Scale)\nTrained on CRACK only', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(violin_path, dpi=150, bbox_inches='tight')
    plt.close()

    # --- Violin plot (Linear Scale) ---
    violin_linear_path = violin_path.parent / f"{violin_path.stem}_linear{violin_path.suffix}"
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot(data_box, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
        pc.set_alpha(0.6)
    plt.xticks(range(1, len(data_box) + 1), labels_box)
    plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
    plt.title('Reconstruction Error Distribution (Violin Plot - Linear Scale)\nTrained on CRACK only', fontsize=14)
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
    This helps understand how well the CRACK-trained autoencoder reconstructs each class.
    """
    plt.figure(figsize=(14, 8))
    
    # Sort classes by median error
    sorted_classes = sorted(class_errors.keys(), 
                           key=lambda c: np.median(class_errors[c]) if len(class_errors[c]) > 0 else 0)
    
    data = [class_errors[c] for c in sorted_classes]
    labels = [f'{c}\n(n={len(class_errors[c])})' for c in sorted_classes]
    
    # Color CRACK differently (it's the training class, should have lowest error)
    colors = ['red' if c == TRAINING_CLASS else 'steelblue' for c in sorted_classes]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.yscale('log')
    plt.ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title('Reconstruction Error by Original Class (Trained on CRACK only)\n(Red = CRACK [training class], Blue = Non-CRACK [anomaly])', fontsize=14)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load multiclass dataset and prepare for anomaly detection.
    
    In this experiment:
    - y_binary = 0 means CRACK (normal for autoencoder training)
    - y_binary = 1 means non-CRACK (anomaly)
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y_binary: Binary labels (0=CRACK/training class, 1=non-CRACK/anomaly)
        y_original: Original string labels for per-class analysis
        image_ids: Image IDs (hs_dir) for LOGO CV
        segment_ids: Segment IDs (mask_path) for preventing leakage
        feature_names: List of wavelength feature names
    """
    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[INFO] Original dataset shape: {df.shape}")
    print(f"[INFO] Class distribution:")
    print(df['label'].value_counts().to_string())
    
    # Store original labels
    y_original = df['label'].values.copy()
    
    # Extract image IDs (hs_dir) and segment IDs (mask_path)
    image_ids = df['hs_dir'].values.copy()
    segment_ids = df['mask_path'].values.copy() if 'mask_path' in df.columns else df['hs_dir'].values.copy()
    
    # Create binary labels: CRACK = 0 (training class), everything else = 1 (anomaly)
    # NOTE: This is INVERTED compared to the other script!
    df['binary_label'] = (df['label'] != TRAINING_CLASS).astype(int)
    
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
    
    return X, y_binary, y_original, image_ids, segment_ids, feature_names


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_autoencoder_domain_aware_cv(
    X: np.ndarray,
    y: np.ndarray,
    y_original: np.ndarray,
    image_ids: np.ndarray,
    segment_ids: np.ndarray,
    hidden_dims: Tuple[int, int, int] = (64, 32, 16),
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
    non_crack_test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate autoencoder using domain-aware CV matching the unified pipeline.
    
    SPLIT STRATEGY (same as unified_experiment_pipeline_acc.py):
    1. LOGO on CRACK samples - each fold holds out one CRACK image
    2. Fixed 80/20 split on non-CRACK samples (same across all folds)
    
    For each fold:
    - TRAIN: ONLY CRACK samples from other CRACK images
    - TEST: CRACK from held-out image + non-CRACK test (20% fixed)
    
    This ensures:
    - No segment leakage between train/test
    - Consistent test set of non-CRACK samples across folds
    - Proper LOGO evaluation on CRACK samples
    
    Note: y=0 is CRACK (training class), y=1 is non-CRACK (anomaly)
    """
    input_dim = X.shape[1]
    n_samples = len(y)
    all_indices = np.arange(n_samples)

    # =================================================================
    # Step 1: Separate CRACK and non-CRACK samples
    # =================================================================
    crack_mask = y == 0  # CRACK (training class)
    non_crack_mask = y == 1  # Non-CRACK (anomaly)
    
    crack_indices = all_indices[crack_mask]
    non_crack_indices = all_indices[non_crack_mask]
    
    crack_image_ids = image_ids[crack_indices]
    non_crack_image_ids = image_ids[non_crack_indices]
    non_crack_segment_ids = segment_ids[non_crack_indices]
    
    unique_crack_images = np.unique(crack_image_ids)
    n_crack_groups = len(unique_crack_images)
    
    print(f"\n{'='*70}")
    print(f"[SPLIT] CRACK samples: {len(crack_indices):,} from {n_crack_groups} images")
    print(f"[SPLIT] Non-CRACK samples: {len(non_crack_indices):,}")
    
    # =================================================================
    # Step 2: Fixed 80/20 split for non-CRACK samples (by segment to prevent leakage)
    # =================================================================
    if len(non_crack_indices) > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=non_crack_test_fraction, random_state=RANDOM_SEED)
        dummy_y = np.zeros(len(non_crack_indices))
        
        try:
            nc_train_local, nc_test_local = next(gss.split(
                non_crack_indices, dummy_y, groups=non_crack_segment_ids
            ))
        except ValueError:
            # Fallback to image-level split if segment split fails
            nc_train_local, nc_test_local = next(gss.split(
                non_crack_indices, dummy_y, groups=non_crack_image_ids
            ))
        
        non_crack_train_global = non_crack_indices[nc_train_local]
        non_crack_test_global = non_crack_indices[nc_test_local]
        
        # Verify no segment leakage
        nc_train_segments = set(segment_ids[non_crack_train_global])
        nc_test_segments = set(segment_ids[non_crack_test_global])
        if nc_train_segments & nc_test_segments:
            raise ValueError("LEAKAGE DETECTED: Non-CRACK segments overlap train/test!")
        
        print(f"[SPLIT] Non-CRACK 80/20 split: {len(non_crack_train_global):,} train, {len(non_crack_test_global):,} test")
    else:
        non_crack_train_global = np.array([], dtype=int)
        non_crack_test_global = np.array([], dtype=int)
    
    # =================================================================
    # Step 3: LOGO on CRACK samples
    # =================================================================
    logo = LeaveOneGroupOut()
    
    print(f"[INFO] Will train {n_crack_groups} models (LOGO over CRACK images)")
    print(f"[INFO] Training on: {TRAINING_CLASS} ONLY")
    print(f"[INFO] Detecting anomaly: {ANOMALY_CLASSES}")
    print(f"[INFO] Threshold method: {THRESHOLD_PERCENTILE}th percentile of training CRACK errors")
    print(f"[INFO] Model architecture: {hidden_dims}")
    print(f"{'='*70}")

    fold_results = []
    all_y_true = []
    all_y_scores = []
    all_precisions = []
    all_recalls = []
    all_fprs = []
    all_tprs = []
    all_errors_test_crack = []
    all_errors_test_non_crack = []
    all_class_errors = {c: [] for c in ALL_CLASSES}
    
    # Track best model by F1 score for saving
    best_fold_f1 = -1.0
    best_model_state = None
    best_scaler = None
    best_threshold = None
    best_fold_idx = -1

    for fold_idx, (crack_train_local, crack_test_local) in enumerate(tqdm(
        logo.split(crack_indices, y[crack_indices], groups=crack_image_ids),
        total=n_crack_groups, desc="LOGO CV (CRACK images)"
    )):
        crack_train_global = crack_indices[crack_train_local]
        crack_test_global = crack_indices[crack_test_local]
        
        # TRAIN: Only CRACK samples from other images
        train_idx = crack_train_global
        
        # TEST: CRACK from held-out image + fixed non-CRACK test set
        test_idx = np.concatenate([crack_test_global, non_crack_test_global])
        
        # Verify no segment leakage
        train_segments = set(segment_ids[train_idx])
        test_segments = set(segment_ids[test_idx])
        segment_leak = train_segments & test_segments
        if segment_leak:
            print(f"  [WARN] Fold {fold_idx}: {len(segment_leak)} segment leaks detected, skipping")
            continue
        
        held_out_image = str(np.unique(crack_image_ids[crack_test_local])[0])
        
        X_train_crack = X[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        y_orig_test = y_original[test_idx]
        
        n_train_crack = len(X_train_crack)
        n_test_total = len(X_test)
        n_test_crack = int(np.sum(y_test == 0))
        n_test_non_crack = int(np.sum(y_test == 1))
        
        if n_train_crack < 10:
            print(f"  [WARN] Fold {fold_idx} (Image: {held_out_image}): Only {n_train_crack} CRACK training samples, skipping")
            continue

        # =================================================================
        # Step 4: Fit scaler on training CRACK data
        # =================================================================
        scaler = StandardScaler()
        X_train_crack_scaled = scaler.fit_transform(X_train_crack).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # =================================================================
        # Step 5: Remove outliers from CRACK training data (if enabled)
        # =================================================================
        if OUTLIER_CONTAMINATION > 0:
            iso_forest = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=RANDOM_SEED, n_jobs=-1)
            outlier_predictions = iso_forest.fit_predict(X_train_crack_scaled)
            inlier_mask = outlier_predictions == 1
            X_train_crack_cleaned = X_train_crack_scaled[inlier_mask]
            n_outliers_removed = n_train_crack - len(X_train_crack_cleaned)

            if verbose:
                print(f"  [INFO] Fold {fold_idx}: Removed {n_outliers_removed}/{n_train_crack} outliers ({n_outliers_removed/n_train_crack*100:.1f}%)")
        else:
            X_train_crack_cleaned = X_train_crack_scaled

        # Check if we have enough samples after cleaning (BatchNorm requires > 1 per batch)
        min_samples_required = max(batch_size + 1, 10)  # Need at least batch_size + 1 to have valid batches
        if len(X_train_crack_cleaned) < min_samples_required:
            print(f"  [WARN] Fold {fold_idx} (Image: {held_out_image}): Only {len(X_train_crack_cleaned)} samples after cleaning, skipping")
            continue

        train_dataset = TensorDataset(torch.from_numpy(X_train_crack_cleaned))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)  # drop_last=True to avoid single-sample batches

        # Initialize and train model
        set_seed(RANDOM_SEED + fold_idx)
        model = SpectralAutoencoder(input_dim, hidden_dims)

        start_train = time.time()
        best_loss, _ = train_autoencoder(model, train_loader, epochs=epochs, lr=lr, patience=15, verbose=False)
        train_time = time.time() - start_train

        # =================================================================
        # Step 6: Compute reconstruction errors
        # =================================================================
        model.to(DEVICE)
        model.eval()
        start_infer = time.time()

        X_train_crack_cleaned_tensor = torch.from_numpy(X_train_crack_cleaned).to(DEVICE)
        errors_train_crack = model.get_reconstruction_error(X_train_crack_cleaned_tensor).cpu().numpy()

        X_test_tensor = torch.from_numpy(X_test_scaled).to(DEVICE)
        errors_test = model.get_reconstruction_error(X_test_tensor).cpu().numpy()

        infer_time = time.time() - start_infer

        # Separate test errors by binary class
        errors_test_crack = errors_test[y_test == 0]  # CRACK (training class)
        errors_test_non_crack = errors_test[y_test == 1]  # Non-CRACK (anomaly)

        # Store per-class errors for analysis
        for cls in ALL_CLASSES:
            cls_mask = y_orig_test == cls
            if np.sum(cls_mask) > 0:
                all_class_errors[cls].append(errors_test[cls_mask])

        all_errors_test_crack.append(errors_test_crack)
        if len(errors_test_non_crack) > 0:
            all_errors_test_non_crack.append(errors_test_non_crack)

        # =================================================================
        # Step 7: Set threshold (based on CRACK training errors)
        # =================================================================
        threshold = float(np.percentile(errors_train_crack, THRESHOLD_PERCENTILE))

        # =================================================================
        # Step 8: Classify test samples
        # Anomaly = error > threshold (predicting non-CRACK)
        # =================================================================
        y_scores = errors_test
        y_pred = (errors_test > threshold).astype(int)

        all_y_true.append(y_test)
        all_y_scores.append(y_scores)

        # =================================================================
        # Step 9: Compute metrics
        # =================================================================
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
            'train_crack_error_mean': float(np.mean(errors_train_crack)),
            'train_crack_error_std': float(np.std(errors_train_crack)),
            'test_crack_error_mean': float(np.mean(errors_test_crack)) if len(errors_test_crack) > 0 else np.nan,
            'test_crack_error_std': float(np.std(errors_test_crack)) if len(errors_test_crack) > 0 else np.nan,
            'test_non_crack_error_mean': float(np.mean(errors_test_non_crack)) if len(errors_test_non_crack) > 0 else np.nan,
            'test_non_crack_error_std': float(np.std(errors_test_non_crack)) if len(errors_test_non_crack) > 0 else np.nan,
        }

        fold_result = {
            'fold': fold_idx,
            'test_crack_image': held_out_image,
            'n_train_crack': n_train_crack,
            'n_train_crack_cleaned': len(X_train_crack_cleaned),
            'n_test_total': n_test_total,
            'n_test_crack': n_test_crack,
            'n_test_non_crack': n_test_non_crack,
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
        
        # Track best model by F1 score
        if f1 > best_fold_f1:
            best_fold_f1 = f1
            best_model_state = model.state_dict().copy()
            best_scaler = scaler
            best_threshold = threshold
            best_fold_idx = fold_idx

        if verbose:
            print(f"\n  ┌─────────────────────────────────────────────────────────────")
            print(f"  │ [Fold {fold_idx}] Held-out CRACK image: {held_out_image}")
            print(f"  ├─────────────────────────────────────────────────────────────")
            print(f"  │ Training: {len(X_train_crack_cleaned):,} CRACK samples")
            print(f"  │ Test: {n_test_crack:,} CRACK + {n_test_non_crack:,} non-CRACK = {n_test_total:,} total")
            print(f"  │ Threshold ({THRESHOLD_PERCENTILE}th pctl): {threshold:.6f}")
            print(f"  ├─────────────────────────────────────────────────────────────")
            if not np.isnan(roc_auc):
                print(f"  │ ROC-AUC: {roc_auc:.4f}    PR-AUC: {pr_auc:.4f}")
            else:
                print(f"  │ ROC-AUC: N/A    PR-AUC: N/A")
            print(f"  │ Precision: {prec:.4f}    Recall: {rec:.4f}    F1: {f1:.4f}")
            print(f"  │ Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  └─────────────────────────────────────────────────────────────")

        # Save per-fold plots
        if save_plots and output_dir is not None:
            fold_plot_dir = output_dir / "per_fold_plots"
            fold_plot_dir.mkdir(parents=True, exist_ok=True)

            plot_reconstruction_error_distribution(
                errors_test_crack, errors_test_non_crack, threshold,
                fold_idx, held_out_image,
                fold_plot_dir / f"error_distribution_fold_{fold_idx:02d}.png"
            )

            if len(np.unique(y_test)) > 1:
                precision_arr, recall_arr = plot_precision_recall_curve(
                    y_test, y_scores, pr_auc,
                    fold_idx, held_out_image,
                    fold_plot_dir / f"pr_curve_fold_{fold_idx:02d}.png"
                )
                all_precisions.append(precision_arr)
                all_recalls.append(recall_arr)

                fpr_arr, tpr_arr = plot_roc_curve(
                    y_test, y_scores, roc_auc,
                    fold_idx, held_out_image,
                    fold_plot_dir / f"roc_curve_fold_{fold_idx:02d}.png"
                )
                all_fprs.append(fpr_arr)
                all_tprs.append(tpr_arr)

    # =================================================================
    # Aggregate results
    # =================================================================
    df_folds = pd.DataFrame(fold_results)

    metrics_to_aggregate = [
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc',
        'threshold', 'train_crack_error_mean', 'test_crack_error_mean', 'test_non_crack_error_mean',
        'train_time_s', 'infer_time_s'
    ]

    summary = {
        'model_name': f"Autoencoder ({hidden_dims[0]}-{hidden_dims[1]}-{hidden_dims[2]})",
        'experiment_type': 'train_on_crack_only_logo',
        'training_class': TRAINING_CLASS,
        'anomaly_classes': ', '.join(ANOMALY_CLASSES),
        'n_crack_groups': n_crack_groups,
        'n_folds': len(fold_results),
        'threshold_percentile': THRESHOLD_PERCENTILE,
        'total_TP': int(df_folds['TP'].sum()) if len(df_folds) > 0 else 0,
        'total_FP': int(df_folds['FP'].sum()) if len(df_folds) > 0 else 0,
        'total_TN': int(df_folds['TN'].sum()) if len(df_folds) > 0 else 0,
        'total_FN': int(df_folds['FN'].sum()) if len(df_folds) > 0 else 0,
    }

    for metric in metrics_to_aggregate:
        if metric in df_folds.columns:
            values = df_folds[metric].dropna()
            summary[f'mean_{metric}'] = float(values.mean()) if len(values) > 0 else np.nan
            summary[f'std_{metric}'] = float(values.std()) if len(values) > 0 else np.nan

    # Save aggregated plots
    if save_plots and output_dir is not None:
        if len(all_precisions) > 0:
            plot_aggregated_pr_curve(
                all_precisions, all_recalls,
                summary.get('mean_pr_auc', np.nan),
                summary.get('std_pr_auc', np.nan),
                output_dir / "aggregated_pr_curve.png"
            )

        if len(all_fprs) > 0:
            plot_aggregated_roc_curve(
                all_fprs, all_tprs,
                summary.get('mean_roc_auc', np.nan),
                summary.get('std_roc_auc', np.nan),
                output_dir / "aggregated_roc_curve.png"
            )

        if len(all_errors_test_crack) > 0:
            plot_error_boxplot_and_violin(
                all_errors_test_crack,
                all_errors_test_non_crack,
                output_dir / "reconstruction_error_boxplot.png",
                output_dir / "reconstruction_error_violin.png"
            )

        # Plot per-class error distribution
        aggregated_class_errors = {}
        for cls, error_lists in all_class_errors.items():
            if error_lists:
                aggregated_class_errors[cls] = np.concatenate(error_lists)
            else:
                aggregated_class_errors[cls] = np.array([])
        
        plot_per_class_error_distribution(
            aggregated_class_errors,
            output_dir / "per_class_error_distribution.png"
        )

    # =================================================================
    # Save best model
    # =================================================================
    if output_dir is not None and best_model_state is not None:
        model_dir = output_dir / "saved_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Recreate model with best weights
        best_model = SpectralAutoencoder(input_dim, hidden_dims)
        best_model.load_state_dict(best_model_state)
        
        # Save model state dict
        torch.save({
            'model_state_dict': best_model_state,
            'hidden_dims': hidden_dims,
            'input_dim': input_dim,
            'threshold': best_threshold,
            'threshold_percentile': THRESHOLD_PERCENTILE,
            'best_fold_idx': best_fold_idx,
            'best_fold_f1': best_fold_f1,
            'training_class': TRAINING_CLASS,
            'anomaly_classes': ANOMALY_CLASSES,
        }, model_dir / "autoencoder_best_model.pt")
        
        # Save scaler
        joblib.dump(best_scaler, model_dir / "scaler.joblib")
        
        # Save config as JSON
        import json
        config_info = {
            'hidden_dims': list(hidden_dims),
            'input_dim': input_dim,
            'threshold': best_threshold,
            'threshold_percentile': THRESHOLD_PERCENTILE,
            'best_fold_idx': best_fold_idx,
            'best_fold_f1': best_fold_f1,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'training_class': TRAINING_CLASS,
            'anomaly_classes': ANOMALY_CLASSES,
        }
        with open(model_dir / "model_config.json", 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print(f"\n[SAVED] Best model (Fold {best_fold_idx}, F1={best_fold_f1:.4f}) saved to: {model_dir}")

    return df_folds, summary


# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Main execution function for autoencoder anomaly detection experiment.
    
    Experiment: Train ONLY on CRACK → Detect non-CRACK as anomaly
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
        EXPERIMENT_NAME = f"autoencoder_train_crack_only_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
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
    print("   AUTOENCODER ANOMALY DETECTION - TRAINED ON CRACK ONLY (MULTICLASS)")
    print("   Domain-Aware Cross-Validation (matching unified pipeline)")
    print("   LOGO on CRACK images + Fixed 80/20 split on non-CRACK")
    print("   TRAINING: CRACK class ONLY (from other CRACK images)")
    print("   DETECTING: Non-CRACK as anomaly")
    print(f"{'='*80}")
    print(f"\n[CONFIGURATION]")
    print(f"  Experiment name:    {EXPERIMENT_NAME}")
    print(f"  Random seed:        {RANDOM_SEED}")
    print(f"  Threshold:          {THRESHOLD_PERCENTILE}th percentile of training CRACK errors")
    print(f"  Wavelength range:   {WL_MIN}-{WL_MAX} nm")
    print(f"  SNV normalization:  {APPLY_SNV}")
    print(f"  Balanced dataset:   {BALANCED} (using FULL unbalanced data)")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print(f"  Device:             {DEVICE}")
    print(f"  Training class:     {TRAINING_CLASS}")
    print(f"  Anomaly classes:    {ANOMALY_CLASSES}")

    # =========================================================================
    # Load and preprocess data
    # =========================================================================
    print(f"\n{'='*80}")
    print("[DATA LOADING]")
    print(f"{'='*80}")
    print(f"  Source: {CSV_PATH}")

    X, y, y_original, image_ids, segment_ids, feature_names = load_multiclass_data(
        CSV_PATH,
        wl_min=WL_MIN,
        wl_max=WL_MAX,
        apply_snv=APPLY_SNV,
    )

    print(f"\n[DATA SUMMARY]")
    print(f"  Total samples:      {X.shape[0]:,}")
    print(f"  Features:           {X.shape[1]} (wavelengths)")
    print(f"  Unique images:      {len(np.unique(image_ids))}")
    print(f"  Unique segments:    {len(np.unique(segment_ids))}")
    print(f"  Binary class distribution (for this experiment):")
    print(f"    - CRACK (y=0):     {np.sum(y==0):,} ({100*np.sum(y==0)/len(y):.1f}%) [Training class]")
    print(f"    - Non-CRACK (y=1): {np.sum(y==1):,} ({100*np.sum(y==1)/len(y):.1f}%) [Anomaly]")
    print(f"  Imbalance ratio:    {np.sum(y==1)/max(np.sum(y==0), 1):.1f}:1 (anomaly:training)")

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
            df_folds, summary = evaluate_autoencoder_domain_aware_cv(
                X, y, y_original, image_ids, segment_ids,
                hidden_dims=config["hidden_dims"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                lr=config["lr"],
                output_dir=config_dir,
                save_plots=True,
                verbose=True,
            )

            df_folds['config'] = config_name
            all_fold_results.append(df_folds)
            all_summaries.append(summary)

            print(f"\n{'='*60}")
            print(f"[RESULTS SUMMARY] Config: {config_name}")
            print(f"{'='*60}")
            print(f"  Accuracy:     {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            print(f"  Precision:    {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
            print(f"  Recall:       {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
            print(f"  F1 Score:     {summary['mean_f1_score']:.4f} ± {summary['std_f1_score']:.4f}")
            print(f"  ROC-AUC:      {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}")
            print(f"  PR-AUC:       {summary['mean_pr_auc']:.4f} ± {summary['std_pr_auc']:.4f}")

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
        df_all_folds = pd.concat(all_fold_results, ignore_index=True)
        df_all_folds.to_csv(OUTPUT_DIR / "all_fold_results.csv", index=False)
        df_all_folds.to_excel(OUTPUT_DIR / "all_fold_results.xlsx", index=False)

        df_summary = pd.DataFrame(all_summaries)
        numeric_cols = df_summary.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_summary[col] = df_summary[col].round(4)

        df_summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)
        df_summary.to_excel(OUTPUT_DIR / "summary_results.xlsx", index=False)

        print(f"\n{'='*80}")
        print("   FINAL RESULTS SUMMARY")
        print(f"{'='*80}")

        display_cols = ['model_name', 'n_crack_groups', 'n_folds', 'mean_accuracy', 'std_accuracy',
                        'mean_precision', 'std_precision', 'mean_recall', 'std_recall',
                        'mean_f1_score', 'std_f1_score', 'mean_roc_auc', 'std_roc_auc',
                        'mean_pr_auc', 'std_pr_auc']
        display_df = df_summary[[c for c in display_cols if c in df_summary.columns]]
        print(display_df.to_string())

        best_idx = df_summary['mean_f1_score'].idxmax()
        best_config = df_summary.loc[best_idx]
        print(f"\n{'='*80}")
        print(f"[BEST CONFIGURATION by F1 Score]")
        print(f"{'='*80}")
        print(f"  Model:      {best_config['model_name']}")
        print(f"  F1 Score:   {best_config['mean_f1_score']:.4f} ± {best_config.get('std_f1_score', 0):.4f}")
        print(f"  PR-AUC:     {best_config['mean_pr_auc']:.4f} ± {best_config.get('std_pr_auc', 0):.4f}")

    print(f"\n{'='*80}")
    print("[DONE] Experiment completed successfully.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
