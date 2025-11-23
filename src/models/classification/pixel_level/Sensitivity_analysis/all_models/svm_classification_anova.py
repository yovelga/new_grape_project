# svm_classification_anova.py
# ------------------------------------------------------------
# SVM classification with ANOVA‑F feature selection
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ------------------------------------------------------------
# Path to CSV file (update as needed)
# ------------------------------------------------------------
CSV_PATH = r"/dataset_builder_grapes/detection/dataset/signatures_cleand_1pct.csv"


# ------------------------------------------------------------
# Helper: extract cluster_id from hs_dir path
# ------------------------------------------------------------
def extract_cluster_id(hs_dir: str) -> str:
    parts = os.path.normpath(hs_dir).split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
def load_data(path, balanced=False):
    df = pd.read_csv(path)
    if {"hs_dir", "label"}.difference(df.columns):
        raise KeyError("Columns 'hs_dir' and 'label' are required")

    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

    if balanced:
        cracked = df[df.label == 1]
        regular = df[df.label == 0].sample(n=len(cracked), random_state=42)
        df = (
            pd.concat([cracked, regular])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    feats = [c for c in df.columns if c.endswith("nm")]
    X = df[feats].values.astype(np.float32)
    y = df["label"].values
    groups = df["cluster_id"].values
    return X, y, groups, feats


# ------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------
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
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pr(y_true, probs):
    pr, rc, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(rc, pr, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision‑Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# SVM + ANOVA-F feature selection + sensitivity analysis
# ------------------------------------------------------------
def run_svm_anova_sensitivity(
    max_k=203, min_k=1, step=1, balanced=False, svc_C=1.0, svc_kernel="rbf"
):
    # 1. load
    X, y, groups, feats = load_data(CSV_PATH, balanced)
    print(f"Total samples: {len(y)} | Clusters: {len(np.unique(groups))}")

    # 2. compute F-scores
    selector_all = SelectKBest(score_func=f_classif, k=len(feats))
    selector_all.fit(X, y)
    scores = selector_all.scores_
    feat_series = pd.Series(scores, index=feats).sort_values(ascending=False)
    print("\nTop 10 features by ANOVA‑F:")
    print(feat_series.head(10).to_string(float_format="%.2f"))

    results = []
    cv = GroupKFold(n_splits=5)

    # 3. loop K=1→max_k
    for k in range(min_k, max_k + 1, step):
        top_feats = feat_series.index[:k].tolist()
        idxs = [feats.index(f) for f in top_feats]
        X_sel = X[:, idxs]

        # pipeline: scaler + SVM (probability=True for ROC/PR)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(C=svc_C, kernel=svc_kernel, probability=True)),
            ]
        )

        acc = cross_val_score(
            pipe, X_sel, y, cv=cv, groups=groups, scoring="accuracy"
        ).mean()
        roc = cross_val_score(
            pipe, X_sel, y, cv=cv, groups=groups, scoring="roc_auc"
        ).mean()
        print(f"K={k:>3} → Acc={acc:.3f} | ROC-AUC={roc:.3f}")

        results.append({"k": k, "accuracy": acc, "roc_auc": roc, "features": top_feats})

    # 4. to DataFrame
    df = pd.DataFrame(results)

    # 5. plot sensitivity
    plt.figure(figsize=(7, 4))
    plt.plot(df["k"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["k"], df["roc_auc"], marker="s", label="ROC AUC")
    plt.xlabel("Number of features (K)")
    plt.ylabel("Performance")
    plt.title(f"SVM ({svc_kernel}) Sensitivity to #Features")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. save
    df.to_csv("svm_anova_sensitivity.csv", index=False)
    print("\nResults saved to svm_anova_sensitivity.csv")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example: RBF‑kernel SVM, C=1.0
    run_svm_anova_sensitivity(
        max_k=204, min_k=1, step=1, balanced=True, svc_C=1.0, svc_kernel="rbf"
    )
