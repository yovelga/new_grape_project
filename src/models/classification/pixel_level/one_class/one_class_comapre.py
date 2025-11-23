"""
One-Class Anomaly Detection with Leave-One-Group-Out Cross-Validation for Spectral Data

This script provides a complete benchmarking workflow for evaluating one-class (anomaly/outlier) detection models
on spectral data, with an emphasis on industrial crack/defect detection scenarios
(common in materials science and manufacturing).

Key Features:
- Loads data and extracts groups for cross-validation (e.g., clusters of samples).
- Implements a robust PyTorch-based autoencoder with scikit-learn compatibility.
- Supports early stopping and adaptive learning rate for neural network training.
- Performs leave-one-group-out (LOGO) cross-validation for unbiased metrics.
- Benchmarks models separately for "crack" and "regular" classes in a proper one-class-only training scheme.
- Aggregates model results and metrics; saves ROC curves and a final summary Excel file.
- Designed to be easy to extend with other one-class models.

Author: [Your Name]
Created: [YYYY-MM-DD]
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    average_precision_score, f1_score  # add f1_score import
)

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Suppress convergence warnings for cleaner output
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from sklearn.base import clone as sk_clone
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

# Experiment configuration: maximum allowed epochs for cross-validation and final model training.
# Early stopping and adaptive LR will usually prevent reaching max unless learning is very slow.
EPOCHS_CV = 10      # Number of epochs per fold in cross-validation (set high)
EPOCHS_FINAL = 10   # Number of epochs for final model training (set high)




def fresh_clone(est):
    # sklearn estimators
    if hasattr(est, "get_params") and est.__class__.__module__.startswith("sklearn"):
        return sk_clone(est)
    if hasattr(est, "get_params"):
        params = est.get_params(deep=True) if "deep" in est.get_params.__code__.co_varnames else est.get_params()
        return est.__class__(**params)
    # fallback
    return copy.deepcopy(est)


class TorchAutoencoder:
    """
    PyTorch Autoencoder with sklearn-like API, GPU support, LR decay, early stopping.
    """
    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (128, 64, 32, 64, 128),
        lr: float = 1e-3,
        max_epochs: int = 500,
        batch_size: int | str = "auto",
        patience_lr: int = 4,
        patience_es: int = 15,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        weight_decay: float = 0.0,
        dropout: float = 0.0,  # NEW: dropout in encoder
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the TorchAutoencoder.
        """
        self.input_dim = input_dim
        self.hidden = hidden
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience_lr = patience_lr
        self.patience_es = patience_es
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self.verbose = verbose

        self._set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()
        self._best_state = None
        self._best_val_loss = float("inf")
        self._current_lr = lr

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self) -> None:
        layers = []
        prev_dim = self.input_dim
        encoder_layers = self.hidden[:len(self.hidden)//2+1]
        decoder_layers = self.hidden[len(self.hidden)//2+1:]
        # Encoder
        for h in encoder_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = h
        # Decoder
        for h in decoder_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        # Output
        layers.append(nn.Linear(prev_dim, self.input_dim))
        self.model = nn.Sequential(*layers).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
        *,
        y_val: np.ndarray | None = None,
    ) -> "TorchAutoencoder":
        """
        Train AE to reconstruct X. If X_val is None, split 90/10 internally (without leakage).
        Optionally, if y_val is provided, logs ROC-AUC and AP on reconstruction error.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        # Always do internal split for validation (never use test set)
        n = len(X_train)
        idx = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        split = int(n * 0.9)
        train_idx, val_idx = idx[:split], idx[split:]
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]

        batch_size = min(200, len(X_tr)) if self.batch_size == "auto" else self.batch_size

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr)),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0 if os.name == "nt" else 2
        )

        val_tensor = torch.from_numpy(X_vl).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.patience_lr, factor=self.lr_factor,
            min_lr=self.min_lr, verbose=self.verbose
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_since_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)

            self.model.eval()
            with torch.no_grad():
                val_out = self.model(val_tensor)
                val_loss = criterion(val_out, val_tensor).item()

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Use delta=1e-5 for improvement threshold
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if self.verbose:
                msg = (f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                       f"lr={current_lr:.2e} | best_val={best_val_loss:.6f} | "
                       f"no_improve={epochs_since_improve}")
                print(msg)

            if epochs_since_improve >= self.patience_es:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}. Restoring best weights.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._best_state = best_state
        self._best_val_loss = best_val_loss
        self._current_lr = optimizer.param_groups[0]["lr"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return reconstructions (not error).
        """
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device)
            out = self.model(X_tensor).cpu().numpy()
        return out

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-sample MSE reconstruction error (anomaly score).
        """
        X = np.asarray(X, dtype=np.float32)
        recon = self.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    def save(self, path: str) -> None:
        """
        Save model to path.
        """
        torch.save({
            "state_dict": self.model.state_dict(),
            "init_params": {
                "input_dim": self.input_dim,
                "hidden": self.hidden,
                "lr": self.lr,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "patience_lr": self.patience_lr,
                "patience_es": self.patience_es,
                "lr_factor": self.lr_factor,
                "min_lr": self.min_lr,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
                "verbose": self.verbose,
            }
        }, path)

    @classmethod
    def load(cls, path: str) -> "TorchAutoencoder":
        """
        Load model from path, securely loading only weights (recommended by PyTorch security).
        """
        # Use weights_only=True to follow PyTorch's security best practices
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(**checkpoint["init_params"])
        obj.model.load_state_dict(checkpoint["state_dict"])
        return obj

    def set_params(self, **kwargs):
        """
        Set parameters (for sklearn compatibility).
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def get_params(self, deep=True):
        """
        Get parameters (for sklearn compatibility).
        """
        return {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience_lr": self.patience_lr,
            "patience_es": self.patience_es,
            "lr_factor": self.lr_factor,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def current_lr(self) -> float:
        """
        Return current learning rate.
        """
        return self._current_lr


print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =================================================================================
# 1. SETUP AND DATA LOADING (WITH GROUP EXTRACTION)
# =================================================================================
print("--- 1. Setting up environment ---")
load_dotenv(dotenv_path=os.getenv('ENV_PATH', r'/.env'))
BASE_PATH = Path(os.getenv('BASE_PATH', ''))
DATA_PATH = os.getenv('DATASET_FOR_TRAIN_PATH')
CSV_PATH = BASE_PATH / DATA_PATH

# Save all results/plots in a 'results' folder next to this script
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {RESULT_DIR}")


def extract_cluster_id(hs_dir: str) -> str:
    """Extracts the cluster folder name."""
    try:
        return Path(hs_dir).parts[-3]
    except IndexError:
        return "unknown"

def find_valid_clusters_with_both_classes(groups: np.ndarray, y: np.ndarray) -> list:
    """
    Return a list of cluster IDs that contain both crack (1) and regular (0) samples.
    """
    valid_clusters = []
    for cluster in np.unique(groups):
        labels_in_cluster = y[groups == cluster]
        if np.any(labels_in_cluster == 0) and np.any(labels_in_cluster == 1):
            valid_clusters.append(cluster)
    return valid_clusters

print("\n--- 2. Loading and preparing data with groups ---")
df = pd.read_csv(CSV_PATH)
df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

spectral_cols = df.columns[6:-3]
X = df[spectral_cols].values.astype(np.float32)
y = df["label"].values.astype(int)  # 0 = regular, 1 = crack
groups = df["cluster_id"].values

# --- Filter clusters to only those containing BOTH crack and regular ---
valid_clusters = find_valid_clusters_with_both_classes(groups, y)
mask = np.isin(groups, valid_clusters)
X = X[mask]
y = y[mask]
groups = groups[mask]

print(f"Dataset details: {len(X)} samples, {len(np.unique(groups))} unique clusters (groups) with both classes.")


# ================= Confusion Matrix (clean) =================


def save_confusion_matrix(y_true, y_pred, out_path, title="Confusion Matrix",
                          class_names=("Normal", "Anomaly"), dpi=200):
    """
    Saves a 2x2 confusion matrix figure to ``out_path``.
    y_true, y_pred: 1D arrays of 0/1 labels.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
    cm = np.asarray(cm)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(2), labels=class_names)
    ax.set_yticks(np.arange(2), labels=class_names)

    thresh = (cm.max() + cm.min()) / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color=("white" if cm[i, j] > thresh else "black"),
                fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
# ============================================================

# =================================================================================
# 3. MODEL AND EVALUATION LOGIC
# =================================================================================

def fold_is_valid(y_train, y_test, nominal_class=0):
    """
    Returns True if the fold is valid:
    - At least one nominal sample in train
    - Both classes present in test
    """
    return (np.sum(y_train == nominal_class) > 0) and (len(np.unique(y_test)) == 2)

def threshold_at_fpr(y_true, scores, target_fpr=0.05):

    fpr, tpr, thr = roc_curve(y_true, scores)
    target_fpr = np.clip(target_fpr, 0.0, 1.0)
    return float(np.interp(target_fpr, fpr, thr))

def evaluate_model_with_logo(
    model_name, model_instance, X, y, groups, train_on_class,
    scaler_type="standard", target_fpr=0.05
):
    """
    Performs Leave-One-Group-Out cross-validation for a given one-class model.
    train_on_class: The label (0 or 1) of the class to train on.
    scaler_type: "standard" or "robust"
    """
    print(f"\n--- Evaluating {model_name} with LOGO (Training on class: {train_on_class}) ---")
    logo = LeaveOneGroupOut()

    all_true_labels = []
    all_anomaly_scores = []
    per_fold_metrics = []

    for i, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X, y, groups), total=logo.get_n_splits(groups=groups),
             desc=f"Cross-validating {model_name}")
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        group_id = np.unique(groups[test_idx])[0]

        # Fold validity checks
        if not fold_is_valid(y_train, y_test, nominal_class=train_on_class):
            print(f"Skipping fold: invalid (group={group_id}) - "
                  f"{'no nominal in train' if np.sum(y_train == train_on_class) == 0 else 'single-class test'}")
            continue

        # Fit scaler ONLY on the training data of the current fold (nominal class)
        scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        X_train_one_class = X_train[y_train == train_on_class]
        scaler.fit(X_train_one_class)
        X_train_scaled = scaler.transform(X_train_one_class)
        X_test_scaled = scaler.transform(X_test)

        # Train the model (fresh clone per fold)
        model_this_fold = fresh_clone(model_instance)
        if isinstance(model_this_fold, TorchAutoencoder):
            model_this_fold._build_model()  # Reset weights
        model_this_fold.set_params(max_epochs=EPOCHS_CV)
        model_this_fold.fit(X_train_scaled)  # AE handles its own validation split

        # Get anomaly scores (MSE per sample)
        if isinstance(model_this_fold, TorchAutoencoder):
            scores = model_this_fold.reconstruction_error(X_test_scaled)
        else:
            scores = -model_this_fold.decision_function(X_test_scaled)  # larger = more anomalous

        # Anomaly is the class we DID NOT train on.
        y_true_anomaly = (y_test != train_on_class).astype(int)

        all_true_labels.extend(y_true_anomaly.tolist())
        all_anomaly_scores.extend(scores.tolist())

        # Only compute metrics if both classes present in test (already checked)
        fold_auc = roc_auc_score(y_true_anomaly, scores)
        fold_ap = average_precision_score(y_true_anomaly, scores)
        per_fold_metrics.append({
            "fold": i,
            "group_id": group_id,
            "n_test": len(y_test),
            "roc_auc": fold_auc,
            "ap": fold_ap
        })
        print(f"Group={group_id} | n_test={len(y_test)} | AUC={fold_auc:.3f} | AP={fold_ap:.3f}")

    # --- After CV: aggregate and compute metrics ---
    all_true_labels = np.array(all_true_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)
    if len(per_fold_metrics) == 0:
        print("No valid folds found. Exiting evaluation.")
        return {}, (np.array([]), np.array([]), float("nan"))

    micro_auc = roc_auc_score(all_true_labels, all_anomaly_scores)
    micro_ap = average_precision_score(all_true_labels, all_anomaly_scores)
    mean_auc = np.mean([m["roc_auc"] for m in per_fold_metrics])
    std_auc = np.std([m["roc_auc"] for m in per_fold_metrics])
    mean_ap = np.mean([m["ap"] for m in per_fold_metrics])
    std_ap = np.std([m["ap"] for m in per_fold_metrics])

    thr_star = threshold_at_fpr(all_true_labels, all_anomaly_scores, target_fpr=target_fpr)
    y_pred = (all_anomaly_scores > thr_star).astype(int)
    cm = confusion_matrix(all_true_labels, y_pred)

    acc = accuracy_score(all_true_labels, y_pred)
    prec = precision_score(all_true_labels, y_pred, zero_division=0)
    rec = recall_score(all_true_labels, y_pred, zero_division=0)
    f1 = f1_score(all_true_labels, y_pred, zero_division=0)

    print(f"\nMean ROC-AUC={mean_auc:.3f} ± {std_auc:.3f} | Mean AP={mean_ap:.3f} ± {std_ap:.3f}")
    print(f"Micro ROC-AUC={micro_auc:.3f} | Micro AP={micro_ap:.3f}")
    print(f"Thr@FPR={target_fpr*100:.1f}% = {thr_star:.6f} | Acc={acc:.3f} | Prec={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")

    # Save per-fold metrics
    pd.DataFrame(per_fold_metrics).to_csv(RESULT_DIR / f"{model_name}_oneclass_cv_metrics.csv", index=False)

    # --- Plot ROC curve (micro) ---
    fpr, tpr, roc_auc_overall = np.array([]), np.array([]), float("nan")
    try:
        fpr, tpr, _ = roc_curve(all_true_labels, all_anomaly_scores)
        roc_auc_overall = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc_overall:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'Overall ROC Curve - {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(RESULT_DIR / f'{model_name}_roc_curve.png')
        plt.close()
    except Exception as e:
        print(f"ROC curve plot failed: {e}")

    # --- Plot Confusion Matrix using thr_star (clean util) ---
    try:
        save_confusion_matrix(
            y_true=all_true_labels,
            y_pred=y_pred,
            out_path=RESULT_DIR / f"{model_name}_confusion_matrix.png",
            title=f"Confusion Matrix (FPR={target_fpr*100:.1f}%) - {model_name}",
            class_names=("Normal", "Anomaly"),
            dpi=200
        )
    except Exception as e:
        print(f"Confusion matrix plot failed: {e}")

    # --- F1-max threshold scan (OOF) and sweep CSV ---
    try:


        y_true = all_true_labels
        scores = all_anomaly_scores

        precs, recs, thrs = precision_recall_curve(y_true, scores)
        if len(thrs) > 0:
            f1_vals = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-9)
            best_idx = int(np.nanargmax(f1_vals))
            thr_f1max = float(thrs[best_idx])

            y_pred_f1 = (scores > thr_f1max).astype(int)
            acc_f1  = accuracy_score(y_true, y_pred_f1)
            prec_f1 = precision_score(y_true, y_pred_f1, zero_division=0)
            rec_f1  = recall_score(y_true, y_pred_f1, zero_division=0)
            f1_best = f1_score(y_true, y_pred_f1)

            print(f"Thr (F1-max) = {thr_f1max:.6f} | Acc={acc_f1:.3f} | Prec={prec_f1:.3f} | Recall={rec_f1:.3f} | F1={f1_best:.3f}")
            print(f"ROC-AUC={roc_auc_score(y_true, scores):.3f} | AP={average_precision_score(y_true, scores):.3f}")

            save_confusion_matrix(
                y_true, y_pred_f1,
                out_path=RESULT_DIR / f"{model_name}_confusion_matrix_F1max.png",
                title=f"Confusion Matrix (F1-max) - {model_name}"
            )
            pd.DataFrame({
                "threshold": thrs,
                "precision": precs[:-1],
                "recall": recs[:-1],
                "f1": f1_vals
            }).to_csv(RESULT_DIR / f"{model_name}_threshold_sweep.csv", index=False)

            with open(RESULT_DIR / f"{model_name}_best_threshold_F1max.txt", "w") as f:
                f.write(
                    f"best_thr={thr_f1max}\nAcc={acc_f1}\nPrec={prec_f1}\n"
                    f"Recall={rec_f1}\nF1={f1_best}\n"
                )
        else:
            print("F1-max scan skipped: precision_recall_curve produced no thresholds.")
    except Exception as e:
        print("F1-max threshold scan or CM plot failed:", e)

    # Return all relevant metrics and arrays for summary/plotting
    return {
        "Model": model_name,
        "Mean ROC_AUC": mean_auc,
        "Std ROC_AUC": std_auc,
        "Mean AP": mean_ap,
        "Std AP": std_ap,
        "Micro ROC_AUC": micro_auc,
        "Micro AP": micro_ap,
        "Threshold@FPR": thr_star,
        "Accuracy@Thr": acc,
        "Precision@Thr": prec,
        "Recall@Thr": rec,
        "F1@Thr": f1,
    }, (fpr, tpr, roc_auc_overall)


# =================================================================================
# 4. RUNNING THE BENCHMARK
# =================================================================================
all_results = []
input_dim = X.shape[1]

# --- Define Models ---
models_to_run = [
    ("Autoencoder_on_Regulars",
     TorchAutoencoder(input_dim=input_dim, lr=1e-3, max_epochs=EPOCHS_CV,
                      patience_lr=4, patience_es=15, lr_factor=0.5, min_lr=1e-6,
                      batch_size="auto", verbose=True, weight_decay=1e-5, dropout=0.0), 0),
]


# --- Main Loop ---
roc_curves = {}
all_results = []

for name, model, train_class in models_to_run:
    try:
        metrics, roc_pack = evaluate_model_with_logo(
            name, model, X, y, groups, train_on_class=train_class,
            scaler_type="standard", target_fpr=0.05
        )
        all_results.append(metrics)
        roc_curves[name] = roc_pack
    except Exception as e:
        print(f"!!! ERROR evaluating {name}: {e} !!!")
        all_results.append({"Model": name, "Mean ROC_AUC": "FAILED"})

# --- Plot all ROC curves together (micro) ---
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc_v) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_v:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title("ROC Curves Comparison (All Models)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "all_models_roc_comparison.png")
plt.close()


# =================================================================================
# 5. FINAL SUMMARY
# =================================================================================
print("\n--- 5. Generating final summary ---")
rows = []
for r in all_results:
    if isinstance(r, dict) and "Model" in r:
        rows.append(r)
    else:
        # Edge case: convert to basic record to avoid crashing
        rows.append({"Model": str(r), "Mean ROC_AUC": np.nan})

# If for any reason there are no results, generate "NO_RESULTS" records
if not rows:
    rows = [{"Model": name, "Mean ROC_AUC": "NO_RESULTS"}
            for name, _, _ in models_to_run]

summary_df = pd.DataFrame(rows).set_index("Model")
summary_path = RESULT_DIR / "logo_comparison_summary.xlsx"
summary_df.to_excel(summary_path)

print("\nComparison Summary (Leave-One-Group-Out):")
print(summary_df)
print(f"\n✅ All done. Models, plots, and summary saved to '{RESULT_DIR}'.")
