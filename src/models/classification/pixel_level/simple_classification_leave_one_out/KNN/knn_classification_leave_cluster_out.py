"""
knn_classification_leave_cluster_out.py
======================================

Cluster-aware **K-Nearest-Neighbours** (KNN) classifier for hyperspectral
grape-pixel signatures (0 = regular, 1 = crack).

Evaluation schemes
------------------
1. GroupKFold cross-validation   (group = cluster ID)
2. GroupShuffleSplit hold-out    (keeps clusters intact)
3. Leave-One-Cluster-Out (LOCO)  cross-validation

All reports and plots are exported to a time-stamped directory:
<project_root>/results/KNN/<YYYY-MM-DD_HH-MM-SS>/
"""

# ------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------ #
import os, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve

from src.utils.project_helps import project_path

warnings.filterwarnings("ignore")  # keep console clean

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
P_LOSS = 0.01  # which cleaned dataset to load
BALANCED = True  # down-sample majority class?
N_SPLITS = 5  # GroupKFold splits
TEST_FRACT = 0.20  # hold-out fraction
N_NEIGHBORS = 7  # K for KNN

# ---------------- Paths ----------------
BASE_DIR = project_path()
CSV_PATH = (
    BASE_DIR / "dataset_builder_grapes" / "dataset" / f"dataset_outlier_{P_LOSS}.csv"
)

THIS_DIR = Path(__file__).parent
RESULTS_DIR = (
    THIS_DIR.parent / "results" / "KNN" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Utility helpers
# ------------------------------------------------------------------ #
def extract_cluster_id(hs_dir: str) -> str:
    """Return the cluster folder name (three levels up)."""
    parts = Path(hs_dir).parts
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_data(path: Path, balanced: bool):
    """Return X, y, groups, feature names."""
    df = pd.read_csv(path)
    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

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
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}.png", dpi=300)
    plt.close()


# ---------------- Plotting ----------------
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
        return
    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{tag} – ROC Curve")
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
# Build KNN pipeline
# ------------------------------------------------------------------ #
def build_knn(k: int = N_NEIGHBORS) -> Pipeline:
    """Standard scaling ➜ KNN classifier."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )


# ------------------------------------------------------------------ #
# Evaluation helpers
# ------------------------------------------------------------------ #
def group_kfold_cv(X, y, groups, pipe):
    gkf = GroupKFold(n_splits=N_SPLITS)
    # --- inside group_kfold_cv ---
    accs = cross_val_score(
        pipe, X, y, cv=gkf, groups=groups, scoring="accuracy", n_jobs=5
    )
    rocs = cross_val_score(
        pipe,
        X,
        y,
        cv=gkf,
        groups=groups,
        scoring="roc_auc",
        n_jobs=5,
        error_score=np.nan,
    )
    print(
        f"GroupKFold {N_SPLITS}× : "
        f"ACC={np.nanmean(accs):.3f}±{np.nanstd(accs):.3f} | "
        f"ROC={np.nanmean(rocs):.3f}±{np.nanstd(rocs):.3f}"
    )


def group_shuffle_split(X, y, groups, pipe):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACT, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    pipe.fit(X[tr_idx], y[tr_idx])

    y_pred = pipe.predict(X[te_idx])
    y_prob = pipe.predict_proba(X[te_idx])[:, 1]

    report = classification_report(
        y[te_idx], y_pred, target_names=["Regular", "Crack"], digits=3
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report)
    print("\nGroup-aware hold-out report saved.\n")

    plot_confusion(y[te_idx], y_pred, "holdout")
    plot_roc_pr(y[te_idx], y_prob, "holdout")
    plot_calibration(y[te_idx], y_prob, "holdout")


def leave_one_cluster_out(X, y, groups, pipe):
    logo = LeaveOneGroupOut()
    fold_acc, fold_roc = [], []

    for tr, te in logo.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        y_pred = pipe.predict(X[te])
        acc = accuracy_score(y[te], y_pred)
        fold_acc.append(acc)

        if len(np.unique(y[te])) > 1:
            y_prob = pipe.predict_proba(X[te])[:, 1]
            roc = roc_auc_score(y[te], y_prob)
        else:
            roc = np.nan
        fold_roc.append(roc)

    df = pd.DataFrame(
        {"cluster": np.unique(groups), "accuracy": fold_acc, "roc_auc": fold_roc}
    )
    df.to_csv(RESULTS_DIR / "loco_metrics.csv", index=False)

    print(
        f"LOCO ({len(df)} clusters) : "
        f"ACC={np.nanmean(fold_acc):.3f}±{np.nanstd(fold_acc):.3f} | "
        f"ROC={np.nanmean(fold_roc):.3f}±{np.nanstd(fold_roc):.3f}"
    )


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Loading", CSV_PATH)
    X, y, groups, feats = load_data(CSV_PATH, BALANCED)
    print(f"Pixels: {len(y):,} | Clusters: {len(np.unique(groups))}\n")

    knn_pipe = build_knn()

    # 1) GroupKFold CV
    group_kfold_cv(X, y, groups, knn_pipe)

    # 2) Group-aware hold-out
    group_shuffle_split(X, y, groups, knn_pipe)

    # 3) Leave-One-Cluster-Out CV
    leave_one_cluster_out(X, y, groups, knn_pipe)

    print("\nAll outputs in →", RESULTS_DIR)
