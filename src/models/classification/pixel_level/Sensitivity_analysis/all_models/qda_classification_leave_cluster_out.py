# lda_classification_leave_cluster_out.py
# ------------------------------------------------------------
# QDA with Group-Aware Splits (Cluster-Leave-Out)
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# calibration_curve is not required in this file; you can add it if desired.

# ------------------------------------------------------------
# Path to CSV file (update as needed)
# ------------------------------------------------------------
CSV_PATH = r"D:\Grape_Project\dataset_builder_grapes\dataset\combined_cleaned_signatures_07_58_new.csv"


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
        raise KeyError("Columns 'hs_dir' and 'label' are required in the CSV")

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
# Plotting functions
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
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main QDA pipeline
# ------------------------------------------------------------
def run_qda(balanced=False, reg_param=0.1):
    X, y, groups, feats = load_data(CSV_PATH, balanced)
    print(f"Total samples: {len(y)} | Clusters: {len(np.unique(groups))}")

    # QDA with regularization (reg_param = λ). Higher values -> stronger shrinkage.
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                QuadraticDiscriminantAnalysis(
                    reg_param=reg_param, store_covariance=False
                ),
            ),
        ]
    )

    # ---------- Group-aware CV ----------
    gkf = GroupKFold(n_splits=5)
    acc = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="accuracy")
    roc = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    print(
        f"GroupKFold CV → Acc: {acc.mean():.3f}±{acc.std():.3f} | ROC AUC: {roc.mean():.3f}±{roc.std():.3f}"
    )

    # ---------- Group-aware Train/Test split ----------
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    pipe.fit(X[tr_idx], y[tr_idx])

    y_pred = pipe.predict(X[te_idx])
    y_prob = pipe.predict_proba(X[te_idx])[:, 1]

    print("\nClassification Report (group split):")
    print(classification_report(y[te_idx], y_pred, target_names=["Regular", "Cracked"]))

    # ---------- Plots ----------
    plot_confusion(y[te_idx], y_pred)
    plot_roc(y[te_idx], y_prob)
    plot_pr(y[te_idx], y_prob)


# ------------------------------------------------------------
# Leave-One-Cluster-Out (LOCO)  ----  New addition
# ------------------------------------------------------------
from sklearn.model_selection import LeaveOneGroupOut


def run_qda_loco(balanced=False, reg_param=0.1):
    """
    Trains QDA multiple times (once per cluster) – each iteration leaves one cluster out for testing.
    Prints mean accuracy and ROC-AUC, and shows per-cluster results.
    """
    X, y, groups, feats = load_data(CSV_PATH, balanced)
    logo = LeaveOneGroupOut()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", QuadraticDiscriminantAnalysis(reg_param=reg_param)),
        ]
    )

    # LOCO Cross-Validation
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
    )  # Raises error if a cluster has only one class

    print(
        f"\nLOCO CV → Acc: {acc.mean():.3f}±{acc.std():.3f} | ROC AUC: {roc.mean():.3f}±{roc.std():.3f}"
    )

    # Optional: detailed per-cluster results
    cluster_list = np.unique(groups)
    detailed = pd.DataFrame({"cluster": cluster_list, "accuracy": acc, "roc_auc": roc})
    print("\nPer-cluster performance:")
    print(
        detailed.to_string(
            index=False,
            formatters={"accuracy": "{:.3f}".format, "roc_auc": "{:.3f}".format},
        )
    )


# ------------------------------------------------------------
# Sensitivity analysis: QDA with descending K features from 50 to 1
# ------------------------------------------------------------
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd


def run_qda_sensitivity(max_k=203, min_k=1, step=1, balanced=False, reg_param=0.1):
    # 1. Load data
    X, y, groups, feats = load_data(CSV_PATH, balanced)

    # 2. Compute F-values and build a sorted Series
    selector_all = SelectKBest(score_func=f_classif, k=len(feats))
    selector_all.fit(X, y)
    scores = selector_all.scores_
    feat_series = pd.Series(scores, index=feats).sort_values(ascending=False)

    # ▶️ New: print top-10 channels
    print("\nTop 10 most influential channels (by ANOVA-F):")
    print(feat_series.head(10).to_string(float_format="%.2f"))

    # 3. … continue with your K loop …

    print(f"\n📊 Starting sensitivity analysis: K={min_k}→{max_k}, balanced={balanced}")

    results = []
    # 3. Iterate over K
    for k in range(min_k, max_k + 1, step):
        # take top-k features
        top_feats = feat_series.index[:k].tolist()
        top_scores = feat_series.iloc[:k]

        # print selected features + their F-values
        # print(f"\nK={k}: selected features and F-values")
        # print(top_scores.to_string(float_format='%.2f'))

        # build X_sel
        idxs = [feats.index(f) for f in top_feats]
        X_sel = X[:, idxs]

        # evaluate
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", QuadraticDiscriminantAnalysis(reg_param=reg_param)),
            ]
        )
        cv = GroupKFold(n_splits=5)
        acc = cross_val_score(
            pipe, X_sel, y, cv=cv, groups=groups, scoring="accuracy"
        ).mean()
        roc = cross_val_score(
            pipe, X_sel, y, cv=cv, groups=groups, scoring="roc_auc"
        ).mean()

        print(f" → K={k} Acc={acc:.3f} | ROC-AUC={roc:.3f}")

        # ← here we add the selected wavelengths to the results
        results.append(
            {"k": k, "accuracy": acc, "roc_auc": roc, "wavelengths": top_feats}
        )

    # 4. To DataFrame & Plot (unchanged) …
    df = pd.DataFrame(results)
    plt.figure(figsize=(7, 4))
    plt.plot(df["k"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["k"], df["roc_auc"], marker="s", label="ROC AUC")
    # plt.gca().invert_xaxis()
    plt.xlabel("Number of features (K)")
    plt.ylabel("Performance")
    plt.title("QDA Sensitivity to Number of Features")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    df.to_csv("qda_sensitivity_results.csv", index=False)
    print("\n✓ Results saved to qda_sensitivity_results.csv")


# ------------------------------------------------------------
# Execute desired functions
# ------------------------------------------------------------
if __name__ == "__main__":
    # Uncomment as needed:
    # run_qda(balanced=True, reg_param=0.1)
    # run_qda_loco(balanced=True, reg_param=0.1)

    # Sensitivity analysis: from 50 features down to 1
    run_qda_sensitivity(max_k=20, min_k=1, step=1, balanced=True, reg_param=0.1)
