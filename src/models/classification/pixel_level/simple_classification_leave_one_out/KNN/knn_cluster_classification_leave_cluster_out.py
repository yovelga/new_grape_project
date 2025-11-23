"""
KNN with Group‑aware splits
──────────────────────────
• Each spectrum (pixel) has hs_dir like:
      D:\data\raw\2_42\25.09.24\HS
  → cluster_id = 2_42
• GroupShuffleSplit (80 % train / 20 % test) and GroupKFold CV
  guarantee that no pixels from the same cluster appear
  in both train and validation/test at once.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve  # ← 修正点


CSV_PATH = r"D:\Grape_Project\dataset_builder_grapes\dataset\Last_combined_cleaned_signatures_15_15_cleand_5pct_new.csv"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def extract_cluster_id(path: str) -> str:
    """
    Given hs_dir =  D:\data\raw\2_42\25.09.24\HS
    returns        '2_42'
    i.e. the folder *two* levels above 'HS'.
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if len(parts) < 3:
        return "unknown"
    return parts[-3]  # -1=HS , -2=date , -3=cluster


def load_data(path, balanced=False):
    df = pd.read_csv(path)

    if "label" not in df.columns or "hs_dir" not in df.columns:
        raise KeyError("CSV must contain 'label' and 'hs_dir' columns.")

    # derive cluster_id column
    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

    feats = [c for c in df.columns if c.endswith("nm")]
    if balanced:
        cracked = df[df.label == 1]
        regular = df[df.label == 0].sample(n=len(cracked), random_state=42)
        df = pd.concat([cracked, regular]).sample(frac=1, random_state=42)

    X = df[feats].values.astype(np.float32)
    y = df["label"].values
    groups = df["cluster_id"].values
    return X, y, groups, feats


def plot_confusion(y_t, y_p):
    cm = confusion_matrix(y_t, y_p)
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
    plt.show()


def plot_roc(y_t, prob):
    fpr, tpr, _ = roc_curve(y_t, prob)
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend()
    plt.show()


def plot_pr(y_t, prob):
    pr, rc, _ = precision_recall_curve(y_t, prob)
    ap = average_precision_score(y_t, prob)
    plt.plot(rc, pr, label=f"AP={ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.show()


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def run_knn(balanced=False):
    X, y, groups, feat_names = load_data(CSV_PATH, balanced)

    print(f"Total pixels: {len(y)}  | Clusters: {len(np.unique(groups))}")

    # ---------- Group K‑Fold CV ----------
    gkf = GroupKFold(n_splits=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

    acc = cross_val_score(knn, X, y, cv=gkf, groups=groups, scoring="accuracy")
    roc = cross_val_score(knn, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    print(
        f"GroupKFold CV  |  Acc {acc.mean():.3f}±{acc.std():.3f}  "
        f"ROC AUC {roc.mean():.3f}±{roc.std():.3f}"
    )

    # ---------- Group train/test split ----------
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    knn.fit(Xtr, ytr)
    y_pred = knn.predict(Xte)
    y_prob = knn.predict_proba(Xte)[:, 1]

    print("\nClassification Report (group split):")
    print(classification_report(yte, y_pred, target_names=["Regular", "Cracked"]))

    # ---------- Plots ----------
    plot_confusion(yte, y_pred)
    plot_roc(yte, y_prob)
    plot_pr(yte, y_prob)


if __name__ == "__main__":
    run_knn(balanced=False)  # run on full data
    run_knn(balanced=True)  # optional: balanced (undersampled) set
