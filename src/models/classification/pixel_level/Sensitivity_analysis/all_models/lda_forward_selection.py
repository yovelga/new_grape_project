# lda_manual_forward_selection.py
# ------------------------------------------------------------
# Manual forward feature selection for LDA maximizing ROC-AUC
# ------------------------------------------------------------
# This script loads hyperspectral data, then:
# 1. Performs manual forward selection: at each step it adds the feature
#    that yields the highest CV ROC‑AUC under GroupKFold.
# 2. Records the order and ROC‑AUC for each added feature.
# 3. Evaluates final selected set for CV accuracy.
# 4. Plots ROC‑AUC progression vs. number of features.
# 5. Splits off a held‑out test by cluster and plots:
#      • Confusion matrix
#      • ROC curve
#      • Precision‑Recall curve
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from src.utils.project_helps import project_path, datasets_path

# ------------------------------------------------------------
# Path to CSV file — update to your own location
# ------------------------------------------------------------
PROJECT_PATH = project_path()
CSV_PATH = datasets_path() / "dataset_outlier_0.01.csv"
# CSV_PATH = r"D:\Grape_Project\dataset_builder_grapes\dataset\signatures_cleand_1pct.csv"


# ------------------------------------------------------------
# Helper: extract cluster_id from hs_dir path
# ------------------------------------------------------------
def extract_cluster_id(hs_dir: str) -> str:
    parts = os.path.normpath(hs_dir).split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


# ------------------------------------------------------------
# Load data into X, y, groups, feats
# ------------------------------------------------------------
def load_data(path, balanced=False):
    df = pd.read_csv(path)
    if {"hs_dir", "label"}.difference(df.columns):
        raise KeyError("Columns 'hs_dir' and 'label' are required")
    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

    if balanced:
        # Undersample the larger class
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
# Main: manual forward selection + final evaluation and plotting
# ------------------------------------------------------------
def run_lda_manual_forward(
    n_features=20, min_Wavelength=400, max_Wavelength=1200, balanced=False
):
    # 1. Load and balance data
    X, y, groups, feats = load_data(CSV_PATH, balanced)
    # parse each feat name (“750.5nm” → 750.5)
    numeric = np.array([float(f.rstrip("nm")) for f in feats])
    mask_vis = (numeric >= min_Wavelength) & (numeric <= max_Wavelength)
    # apply mask to feats and X
    feats = [f for f, keep in zip(feats, mask_vis) if keep]
    X = X[:, mask_vis]
    print(
        f" bands: {len(feats)} features from {min_Wavelength}nm to {max_Wavelength}nm"
    )
    print(f"Total samples: {len(y)} | Clusters: {len(np.unique(groups))}")

    # 2. Prepare LDA pipeline and CV splitter
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    cv = GroupKFold(n_splits=5)

    # 3. Manual forward selection
    selected, remaining, history = [], feats.copy(), []
    for step in range(1, n_features + 1):
        best_feat, best_score = None, -np.inf
        for feat in remaining:
            trial = selected + [feat]
            X_sel = X[:, [feats.index(f) for f in trial]]
            score = cross_val_score(
                pipe, X_sel, y, cv=cv, groups=groups, scoring="roc_auc", n_jobs=-1
            ).mean()
            if score > best_score:
                best_score, best_feat = score, feat
        selected.append(best_feat)
        remaining.remove(best_feat)
        history.append((best_feat, best_score))
        print(f"Step {step:>2}: add {best_feat:<8} → ROC‑AUC={best_score:.4f}")

    # 4. Save selection history
    df = pd.DataFrame(history, columns=["feature", "roc_auc"])
    df["selection_rank"] = range(1, len(df) + 1)
    df.to_excel(f"lda_forward_history_{min_WL}_{max_WL}.xlsx", index=False)

    # 5. Final CV accuracy on selected set
    idxs = [feats.index(f) for f in selected]
    acc = cross_val_score(
        pipe, X[:, idxs], y, cv=cv, groups=groups, scoring="accuracy", n_jobs=-1
    ).mean()
    print(f"\nFinal {n_features}-feature CV Accuracy: {acc:.3f}")

    # 6. Plot ROC‑AUC vs. number of features
    plt.figure(figsize=(7, 4))
    plt.plot(df["selection_rank"], df["roc_auc"], marker="o")
    plt.xlabel("Number of features selected")
    plt.ylabel("CV ROC‑AUC")
    plt.title("Manual Forward Selection Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7. Compute CV accuracy progression as well (for combined plot)
    accs = []
    for k in df["selection_rank"]:
        feats_k = selected[:k]
        idxs_k = [feats.index(f) for f in feats_k]
        accs.append(
            cross_val_score(
                pipe,
                X[:, idxs_k],
                y,
                cv=cv,
                groups=groups,
                scoring="accuracy",
                n_jobs=-1,
            ).mean()
        )
    df["accuracy"] = accs

    plt.figure(figsize=(8, 5))
    plt.plot(df["selection_rank"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["selection_rank"], df["roc_auc"], marker="s", label="ROC AUC")
    plt.xlabel("Number of features selected (K)")
    plt.ylabel("Performance")
    plt.title(
        f"LDA Forward Selection Sensitivity Analysis Wavelength {min_WL} to {max_WL}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 8. Final train/test split by cluster
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_final = X[:, idxs]
    pipe.fit(X_final[tr_idx], y[tr_idx])
    y_pred = pipe.predict(X_final[te_idx])
    y_prob = pipe.predict_proba(X_final[te_idx])[:, 1]

    # 9. Compute final metrics
    cm = confusion_matrix(y[te_idx], y_pred)
    fpr, tpr, _ = roc_curve(y[te_idx], y_prob)
    pr, rc, _ = precision_recall_curve(y[te_idx], y_prob)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y[te_idx], y_prob)

    # 10. Plot confusion / ROC / PR
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Regular", "Cracked"],
        yticklabels=["Regular", "Cracked"],
    )
    plt.title("Confusion Matrix")

    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(rc, pr, label=f"AP={ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision‑Recall")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Selected features:", selected)


if __name__ == "__main__":
    min_WL = 400
    max_WL = 1200
    run_lda_manual_forward(
        n_features=20, balanced=True, min_Wavelength=min_WL, max_Wavelength=max_WL
    )
