"""
lda_classification_leave_cluster_out.py
======================================

Group-aware Quadratic Discriminant Analysis (QDA) for hyperspectral
grape pixels.  Two evaluation schemes are implemented:

1.  **GroupKFold + GroupShuffleSplit** (original workflow)
2.  **Leave-One-Cluster-Out (LOCO)** cross-validation

All metrics and plots are exported to a timestamped directory
under `results/`.
"""

# --------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------- #
import os, sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imageio.testing import THIS_DIR

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
    LeaveOneGroupOut,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Silence the "priors not set; will be inferred" warning
warnings.filterwarnings("ignore", category=UserWarning, message="priors not set.*")
from src.utils.project_helps import project_path

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
P_LOSS = 0.01  # which cleaned dataset to load  (0.01 ⇒ 1 % outliers)
REG_PARAM = 0.10  # QDA shrinkage
BALANCED = True  # down-sample class 0 to match class 1
N_SPLITS = 5  # GroupKFold splits

# Base data directory – override with env var if you like
BASE_DIR = project_path()
DATASETS_DIR = BASE_DIR / "dataset_builder_grapes" / "dataset"
CSV_PATH = DATASETS_DIR / f"dataset_outlier_{P_LOSS}.csv"

# Results folder
THIS_DIR = Path(__file__).parent
RESULTS_DIR = (
    THIS_DIR.parent / "results" / "QDA" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
print(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def extract_cluster_id(hs_dir: str) -> str:
    """
    Given a path like .../1_05/01.08.24/HS/..., return the cluster folder
    (three levels up from the file).
    """
    parts = Path(hs_dir).parts
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_data(csv_path: Path, balanced: bool = False):
    """Return X, y, groups, feature-names."""
    df = pd.read_csv(csv_path)
    if not {"hs_dir", "label"}.issubset(df.columns):
        raise KeyError("CSV must contain 'hs_dir' and 'label' columns")

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
    """Save current Matplotlib figure into RESULTS_DIR and close it."""
    path = RESULTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"   ↳ saved {path.name}")


# --------------------------------------------------------------------- #
# Plotting utilities
# --------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Regular", "Cracked"],
        yticklabels=["Regular", "Cracked"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    save_fig("confusion_matrix")


def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    save_fig("roc_curve")


def plot_pr(y_true, probs):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    save_fig("pr_curve")


# --------------------------------------------------------------------- #
# QDA pipeline wrappers
# --------------------------------------------------------------------- #
def build_pipeline(reg_param: float = REG_PARAM) -> Pipeline:
    """Standard-scale then QDA with shrinkage."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=reg_param)),
        ]
    )


def group_kfold_eval(X, y, groups, pipe):
    gkf = GroupKFold(n_splits=N_SPLITS)
    acc = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="accuracy")
    roc = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    print(
        f"\nGroupKFold ({N_SPLITS} splits):  "
        f"ACC={acc.mean():.3f}±{acc.std():.3f}  "
        f"ROC-AUC={roc.mean():.3f}±{roc.std():.3f}"
    )


def train_test_group_split(X, y, groups, pipe):
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    pipe.fit(X[tr_idx], y[tr_idx])

    y_pred = pipe.predict(X[te_idx])
    y_prob = pipe.predict_proba(X[te_idx])[:, 1]

    # Text report
    report = classification_report(
        y[te_idx], y_pred, target_names=["Regular", "Cracked"], digits=3
    )
    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    print("\nSaved detailed classification report to", report_path.name)

    # Plots
    plot_confusion(y[te_idx], y_pred)
    plot_roc(y[te_idx], y_prob)
    plot_pr(y[te_idx], y_prob)


def leave_one_cluster_out(X, y, groups, pipe):
    logo = LeaveOneGroupOut()
    acc = cross_val_score(
        pipe, X, y, cv=logo, groups=groups, scoring="accuracy", n_jobs=-1
    )
    roc = cross_val_score(
        pipe,
        X,
        y,
        cv=logo,
        groups=groups,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise",
    )

    print(
        f"\nLOCO ({len(acc)} clusters):  "
        f"ACC={acc.mean():.3f}±{acc.std():.3f}  "
        f"ROC-AUC={roc.mean():.3f}±{roc.std():.3f}"
    )

    # Per-cluster CSV
    per_cluster = pd.DataFrame(
        {"cluster": np.unique(groups), "accuracy": acc, "roc_auc": roc}
    )
    csv_path = RESULTS_DIR / "loco_metrics.csv"
    per_cluster.to_csv(csv_path, index=False)
    print("Per-cluster scores saved to", csv_path.name)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Loading data from:", CSV_PATH)
    X, y, groups, feature_names = load_data(CSV_PATH, BALANCED)
    print(f"Total pixels: {len(y):,}  |  Clusters: {len(np.unique(groups))}")

    pipeline = build_pipeline(REG_PARAM)

    # ----- Evaluation #1 : GroupKFold CV -----
    group_kfold_eval(X, y, groups, pipeline)

    # ----- Evaluation #2 : Group-aware train/test split -----
    train_test_group_split(X, y, groups, pipeline)

    # ----- Evaluation #3 : Leave-One-Cluster-Out CV -----
    leave_one_cluster_out(X, y, groups, pipeline)

    print("\nAll results →", RESULTS_DIR)
