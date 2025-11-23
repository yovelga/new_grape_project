# lda_classification_mrmr.py
# ------------------------------------------------------------
# LDA classification with mRMR‐style feature selection
# ------------------------------------------------------------
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, cross_val_score
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

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file

load_dotenv(dotenv_path=r'../.env')

BASE_PATH = os.getenv('BASE_PATH')
if BASE_PATH is None:
    raise ValueError("BASE_PATH is not set in .env or failed to load.")

BASE_PATH = Path(BASE_PATH)

# Access variables
BASE_PATH = os.getenv('BASE_PATH')
DATASET_LDA_PATH = os.getenv('dataset_lda_path')
LDA_MODEL_PATH = os.getenv('lda_model_path')



# ------------------------------------------------------------
# Path to CSV file – update as needed
# ------------------------------------------------------------
P_LOSS = 0.01
BASE_PATH = Path(BASE_PATH)
DATASETS_DIR = BASE_PATH / "dataset_builder_grapes" / "dataset"
CSV_PATH = BASE_PATH / DATASET_LDA_PATH


print(BASE_PATH)
print(DATASETS_DIR)
print(CSV_PATH)
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
# Utility: mRMR-like spacing-based selection
# ------------------------------------------------------------
def select_diverse_feats_by_spacing(feat_series, delta_nm=10, k=50):
    """
    Greedy selection: pick top‐F feature, then skip any bands within ±delta_nm,
    until we have k features.
    feat_series: pandas.Series indexed by wavelength strings sorted desc by F-score
    """
    # parse numeric wavelengths
    wl_nums = {f: float(f.rstrip("nm")) for f in feat_series.index}
    selected = []
    for f in feat_series.index:
        if len(selected) >= k:
            break
        if all(abs(wl_nums[f] - wl_nums[s]) > delta_nm for s in selected):
            selected.append(f)
    return selected


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
# Main: LDA + mRMR-like sensitivity analysis
# ------------------------------------------------------------
def run_lda_mrmr_sensitivity(max_k=50, min_k=1, step=1, delta_nm=10, balanced=False):
    # 1. Load data
    X, y, groups, feats = load_data(CSV_PATH, balanced)
    print(f"Total samples: {len(y)} | Clusters: {len(np.unique(groups))}")

    # 2. Compute ANOVA-F scores and sort
    selector_all = SelectKBest(score_func=f_classif, k=len(feats))
    selector_all.fit(X, y)
    scores = selector_all.scores_
    feat_series = pd.Series(scores, index=feats).sort_values(ascending=False)
    print("\nTop 10 features by ANOVA‑F:")
    print(feat_series.head(10).to_string(float_format="%.2f"))

    results = []
    cv = GroupKFold(n_splits=5)

    # 3. Iterate K from min_k to max_k
    for k in range(min_k, max_k + 1, step):
        # select k diverse features
        top_feats = select_diverse_feats_by_spacing(feat_series, delta_nm=delta_nm, k=k)
        idxs = [feats.index(f) for f in top_feats]
        X_sel = X[:, idxs]

        # build pipeline: scaler + LDA (with shrinkage)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )

        # evaluate
        acc = cross_val_score(
            pipe, X_sel, y, cv=cv, groups=groups, scoring="accuracy"
        ).mean()
        roc = cross_val_score(
            pipe, X_sel, y, cv=cv, groups=groups, scoring="roc_auc"
        ).mean()
        print(f"K={k:>3} → Acc={acc:.3f} | ROC-AUC={roc:.3f}")

        results.append({"k": k, "accuracy": acc, "roc_auc": roc, "features": top_feats})

    # 4. Convert to DataFrame
    df = pd.DataFrame(results)

    # 5. Plot sensitivity
    plt.figure(figsize=(7, 4))
    plt.plot(df["k"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["k"], df["roc_auc"], marker="s", label="ROC AUC")
    plt.xlabel("Number of features (K)")
    plt.ylabel("Performance")
    plt.title("LDA + mRMR-like Sensitivity to #Features")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Save results
    df.to_csv("lda_mrmr_sensitivity.csv", index=False)
    print("\nResults saved to lda_mrmr_sensitivity.csv")

    # Choose best k (e.g., 30)
    k_best = 30
    top_feats = select_diverse_feats_by_spacing(feat_series, delta_nm=10, k=k_best)
    idxs = [feats.index(f) for f in top_feats]
    X_sel = X[:, idxs]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    pipe.fit(X_sel, y)

    joblib.dump({"model": pipe, "features": top_feats}, f"{BASE_PATH/LDA_MODEL_PATH}")
    print(f"Model saved to lda_grape_model.joblib to {BASE_PATH/LDA_MODEL_PATH} with features: {top_feats}")





# ------------------------------------------------------------
# Execute when run as script
# ------------------------------------------------------------
if __name__ == "__main__":
    run_lda_mrmr_sensitivity(max_k=50, min_k=1, step=1, delta_nm=10, balanced=True)
