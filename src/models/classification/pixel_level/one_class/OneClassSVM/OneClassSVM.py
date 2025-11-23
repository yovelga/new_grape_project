import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import os
from dotenv import load_dotenv
from pathlib import Path
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. Load environment and data
load_dotenv(dotenv_path=os.getenv('ENV_PATH', r'/.env'))
BASE_PATH = Path(os.getenv('BASE_PATH'))
DATA_PATH = os.getenv('DATASET_FOR_TRAIN_PATH')
CSV_DIR = BASE_PATH / DATA_PATH

print(f"Loading data from: {CSV_DIR}")
df = pd.read_csv(CSV_DIR)
print(f"Data shape: {df.shape}")

# 2. Feature selection (CRACK)
spectral_cols = df.columns[6:-3]
X = df[spectral_cols].values
y = df["label"].values  # 0 = regular, 1 = crack

# -------- LOGO SUPPORT --------
def extract_cluster_id(hs_dir: str) -> str:
    """Extracts folder/cluster ID for grouping (must match to your data!)."""
    try:
        return Path(hs_dir).parts[-3]
    except Exception:
        return "unknown"

if "hs_dir" in df.columns:
    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)
    groups = df["cluster_id"].values
else:
    # fallback: group by row index so every row is its own group
    groups = np.arange(len(df))

# LOGO-CV for OneClassSVM
logo = LeaveOneGroupOut()
all_true, all_scores, all_preds, test_idx_all, all_fold = [], [], [], [], []

print("Starting LOGO cross-validation for OneClassSVM...")
for fold, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X, y, groups), total=logo.get_n_splits(groups=groups))):
    # Use only "normal" class for training
    normal_class = 0    # Change to 1 if crack is "normal" in your experiment
    X_train, y_train = X[train_idx], y[train_idx]
    X_train_norm = X_train[y_train == normal_class]
    if X_train_norm.shape[0] < 10:
        print(f"Skipping fold {fold} (too few samples for training).")
        continue

    X_test, y_test = X[test_idx], y[test_idx]

    # Fit scaler only on train (normal class)
    scaler = StandardScaler().fit(X_train_norm)
    X_train_scaled = scaler.transform(X_train_norm)
    X_test_scaled = scaler.transform(X_test)

    # Optional: feature selector (VarianceThreshold as in original)
    sel = VarianceThreshold(threshold=1e-5)
    X_train_fs = sel.fit_transform(X_train_scaled)
    X_test_fs = sel.transform(X_test_scaled)

    # Tune OneClassSVM via grid (keep small for speed)
    param_grid = {
        "nu": [0.01, 0.05, 0.1],
        "gamma": ['scale', 0.01, 0.1, 0.5]
    }
    ocsvm = OneClassSVM(kernel="rbf")
    grid = GridSearchCV(
        ocsvm,
        param_grid,
        cv=3,
        verbose=0,
        n_jobs=-1,
        scoring="accuracy"
    )
    train_targets = np.ones(X_train_fs.shape[0])  # All normal class
    grid.fit(X_train_fs, train_targets)
    ocsvm_best = grid.best_estimator_

    # Predict/score on held-out group
    y_pred = ocsvm_best.predict(X_test_fs)
    # True anomaly (crack is 1, regular is 0): detect anything not 'normal' class as anomaly
    y_pred_bin = (y_pred == -1).astype(int)
    y_true_anomaly = (y_test != normal_class).astype(int)

    scores = - ocsvm_best.decision_function(X_test_fs) # Higher = more anomalous
    all_true.extend(y_true_anomaly)
    all_scores.extend(scores)
    all_preds.extend(y_pred_bin)
    test_idx_all.extend(test_idx)
    all_fold.extend([fold] * len(test_idx))

    # Save per-fold predictions if desired
    result_df = pd.DataFrame({
        "index": test_idx,
        "fold": fold,
        "true_label": y_true_anomaly,
        "score": scores,
        "pred_label": y_pred_bin
    })
    result_csv_fold = f"ocsvm_logo_pred_fold{fold}.csv"
    result_df.to_csv(result_csv_fold, index=False)

# --- Aggregate results and plots ---
all_true = np.array(all_true)
all_scores = np.array(all_scores)
all_preds = np.array(all_preds)
all_fold = np.array(all_fold)
test_idx_all = np.array(test_idx_all)

# Save all out-of-fold results
res_all_df = pd.DataFrame({
    "index": test_idx_all,
    "fold": all_fold,
    "true_label": all_true,
    "score": all_scores,
    "pred_label": all_preds
})
res_all_df = res_all_df.sort_values("index")

# Attach cluster_id for per-cluster evaluation
if "cluster_id" in df.columns:
    cluster_table = df[["cluster_id"]].reset_index()
    res_all_df = res_all_df.merge(cluster_table, left_on="index", right_on="index", how="left")
else:
    res_all_df["cluster_id"] = res_all_df["fold"] # fallback in case not present

res_all_df.to_csv("ocsvm_logo_all_predictions.csv", index=False)

# --- Compute and print metrics over all clusters ---
fpr, tpr, _ = roc_curve(all_true, all_scores)
roc_auc_val = auc(fpr, tpr)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(all_true, all_scores)
avg_prec_val = average_precision_score(all_true, all_scores)
acc_val = accuracy_score(all_true, all_preds)

print(f"LOGO SVM ROC AUC: {roc_auc_val:.3f} | AP: {avg_prec_val:.3f} | ACC: {acc_val:.3f}")

# --- Per-cluster results ---
clusters = res_all_df["cluster_id"].unique()
cluster_metrics = []
print("\nPer-cluster results (ROC AUC | AP | ACC):")
for cid in clusters:
    r = res_all_df[res_all_df["cluster_id"] == cid]
    if len(r["true_label"].unique()) < 2:
        # must have at least one pos and one neg class for ROC AUC
        continue
    roc_auc = auc(*roc_curve(r["true_label"], r["score"])[:2])
    ap = average_precision_score(r["true_label"], r["score"])
    acc = accuracy_score(r["true_label"], r["pred_label"])
    print(f"  Cluster {cid}: ROC AUC={roc_auc:.3f} | AP={ap:.3f} | ACC={acc:.3f}")
    cluster_metrics.append({"cluster_id": cid, "roc_auc": roc_auc, "ap": ap, "acc": acc})

# --- Show average results over clusters ---
if cluster_metrics:
    avg_roc_auc = np.mean([m["roc_auc"] for m in cluster_metrics])
    avg_ap = np.mean([m["ap"] for m in cluster_metrics])
    avg_acc = np.mean([m["acc"] for m in cluster_metrics])
    print(f"\nAverage across clusters: ROC AUC={avg_roc_auc:.3f} | AP={avg_ap:.3f} | ACC={avg_acc:.3f}")
    pd.DataFrame(cluster_metrics).to_excel("ocsvm_logo_per_cluster_metrics.xlsx", index=False)
else:
    print("\nNo valid clusters for per-cluster metrics (each cluster must have both classes in test samples).")

# ROC Curve plot
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc_val:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("One-Class SVM ROC Curve (LOGO)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("ocsvm_logo_roc_curve.png")
plt.close()

# PR Curve plot
plt.figure()
plt.plot(recall, precision, lw=2, label=f"AP = {avg_prec_val:.3f}")
plt.title("Precision–Recall Curve (LOGO OneClassSVM)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("ocsvm_logo_pr_curve.png")
plt.close()

# Anomaly Score Boxplot
plt.figure(figsize=(7,5))
sns.boxplot(x=all_true, y=all_scores)
plt.title("Anomaly Score Distribution (LOGO OneClassSVM)")
plt.xlabel("True Label (0=Regular, 1=Crack/Anomaly)")
plt.ylabel("OCSVM Anomaly Score (higher = more anomalous)")
plt.tight_layout()
plt.savefig("ocsvm_logo_anomaly_score_boxplot.png")
plt.close()

# Anomaly Score Histogram by Class
plt.figure(figsize=(8,4))
for lbl in np.unique(all_true):
    plt.hist(all_scores[all_true == lbl], bins=40, alpha=0.65,
             label=f"Label {lbl}", density=True)
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.title("Anomaly Score Histogram by True Class (LOGO OCSVM)")
plt.legend()
plt.tight_layout()
plt.savefig("ocsvm_logo_anomaly_score_hist.png")
plt.close()

# Save metrics
pd.DataFrame([{
    "roc_auc": roc_auc_val,
    "avg_precision": avg_prec_val,
    "accuracy": acc_val
}]).to_excel("ocsvm_logo_metrics.xlsx", index=False)
print("All done (LOGO SVM)! All results/plots saved.")
