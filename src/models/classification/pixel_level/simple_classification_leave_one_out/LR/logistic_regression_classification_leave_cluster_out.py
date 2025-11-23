"""
logistic_cluster_classification.py
==================================

Cluster-aware **Logistic Regression** for hyperspectral grape-pixel
classification (0 = regular, 1 = crack).

Evaluation workflow
-------------------
1. **GroupKFold** cross-validation (group = cluster ID).
2. **GroupShuffleSplit** train/hold-out split that keeps clusters intact.
3. Diagnostic plots (confusion matrix, ROC, PR, calibration).

All outputs are saved to `results/LogReg/<time-stamp>/`.
"""

# ------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------ #
import os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

from src.utils.project_helps import project_path

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
P_LOSS = 0.01  # which cleaned dataset to load
BALANCED = True  # down-sample majority class?
N_SPLITS = 5  # GroupKFold splits
TEST_FRACT = 0.20  # hold-out fraction in GroupShuffleSplit

# -----  Paths ------------------------------------------------------ #
BASE_DIR = project_path()  # project root
DATASET_DIR = BASE_DIR / "dataset_builder_grapes" / "dataset"
CSV_PATH = DATASET_DIR / f"dataset_outlier_{P_LOSS}.csv"

THIS_DIR = Path(__file__).parent
RESULTS_DIR = (
    THIS_DIR.parent
    / "results"
    / "LogReg"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def extract_cluster_id(hs_dir: str) -> str:
    """Return the cluster folder name: .../<cluster>/<date>/HS/..."""
    parts = Path(hs_dir).parts
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_data(path: Path, balanced: bool = False):
    """Load X, y, groups (cluster_id) and feature names."""
    df = pd.read_csv(path)
    if {"hs_dir", "label"}.difference(df.columns):
        raise KeyError("CSV must contain 'hs_dir' and 'label' columns")

    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

    # optional class balancing
    if balanced:
        cracked = df[df.label == 1]
        regular = df[df.label == 0].sample(len(cracked), random_state=42)
        df = pd.concat([cracked, regular]).sample(frac=1, random_state=42)

    features = [c for c in df.columns if c.endswith("nm")]
    X = df[features].to_numpy(np.float32)
    y = df["label"].to_numpy(int)
    groups = df["cluster_id"].to_numpy(str)
    return X, y, groups, features


def save_fig(name: str):
    """Save current Matplotlib figure inside RESULTS_DIR and close."""
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}.png", dpi=300)
    plt.close()


# ------------------------------------------------------------------ #
# Plotting utilities
# ------------------------------------------------------------------ #
def plot_confusion(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Regular", "Crack"],
        yticklabels=["Regular", "Crack"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{tag} – Confusion Matrix")
    save_fig(f"{tag}_confusion")


def plot_roc_pr(y_true, probs, tag):
    if len(np.unique(y_true)) == 1:
        return  # single-class → skip ROC/PR
    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{tag} – ROC")
    plt.legend()
    plt.grid(True)
    save_fig(f"{tag}_roc")
    # PR
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{tag} – Precision-Recall")
    plt.legend()
    plt.grid(True)
    save_fig(f"{tag}_pr")


def plot_calibration(y_true, probs, tag, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins)
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{tag} – Calibration")
    plt.grid(True)
    save_fig(f"{tag}_calibration")


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #
def run_logistic(balanced: bool = False):
    # 1) Load data
    X, y, groups, feats = load_data(CSV_PATH, balanced)
    print(f"Pixels: {len(y):,} | Clusters: {len(np.unique(groups))}")

    # 2) GroupKFold cross-validation
    gkf = GroupKFold(n_splits=N_SPLITS)
    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")

    acc = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="accuracy")
    roc = cross_val_score(
        model, X, y, cv=gkf, groups=groups, scoring="roc_auc", error_score=np.nan
    )
    print(
        f"GroupKFold {N_SPLITS}× : "
        f"ACC={np.nanmean(acc):.3f}±{np.nanstd(acc):.3f} | "
        f"ROC={np.nanmean(roc):.3f}±{np.nanstd(roc):.3f}"
    )

    # 3) Group-aware train/test split
    gss = GroupShuffleSplit(test_size=TEST_FRACT, n_splits=1, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # 4) Fit + evaluate
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    report_txt = classification_report(
        y_te, y_pred, target_names=["Regular", "Crack"], digits=3
    )
    print("\nGroup-aware hold-out report\n", report_txt)
    (RESULTS_DIR / "classification_report.txt").write_text(report_txt)

    # 5) Plots
    plot_confusion(y_te, y_pred, "holdout")
    plot_roc_pr(y_te, y_prob, "holdout")
    plot_calibration(y_te, y_prob, "holdout")

    print("\nAll outputs →", RESULTS_DIR)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    run_logistic(balanced=BALANCED)
