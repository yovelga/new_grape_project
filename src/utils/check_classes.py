# Greedy forward-selection with LOGO CV, and per-cluster class-counts utility
from pathlib import Path
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# Suppress specific sklearn metric warnings for undefined metrics due to single-class test sets in CV folds
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module="sklearn.metrics._ranking")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module="sklearn.metrics._classification")

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select given columns by index from a numpy array."""
    def __init__(self, cols):
        self.cols = list(cols)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.cols]

def run_greedy_forward_logo(dataset_path: Path, out_dir: Path,
                            balance_per_fold: bool = True,
                            k_max: int = 12):
    """
    For each model, find ONE greedy forward order of k_max wavelengths under LOGO.
    At each step, try adding each remaining wavelength, evaluate mean ROC AUC via LOGO,
    pick the best, append; repeat until k_max. Then evaluate full metrics per k for that order.
    Save order, CSV metrics+Δ, and marginal-gain plots.
    """
    # --- Load ---
    df = pd.read_csv(dataset_path)
    feat_cols = [c for c in df.columns if str(c).endswith("nm")]
    def _extract_cluster_id(hs_dir: str) -> str:
        parts = os.path.normpath(hs_dir).split(os.sep)
        return parts[-3] if len(parts) >= 3 else "unknown"
    groups = df["hs_dir"].apply(_extract_cluster_id).values
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    MODELS = {
        "LDA": LinearDiscriminantAnalysis(solver="svd"),
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        # "SVC": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),  # optional
    }

    SCORING = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    logo = LeaveOneGroupOut()

    for model_name, clf in MODELS.items():
        # ---- GREEDY ORDER ----
        selected, remaining = [], list(range(len(feat_cols)))
        for step in range(1, k_max + 1):
            best_score, best_feat = -np.inf, None
            for f in remaining:
                cols = selected + [f]
                steps = []
                if balance_per_fold:
                    steps.append(("sampler", RandomUnderSampler(random_state=42)))
                steps += [
                    ("sel", ColumnSelector(cols)),
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ]
                pipe = (ImbPipeline if balance_per_fold else Pipeline)(steps=steps)
                # LOGO eval with ROC AUC
                scores = cross_validate(
                    pipe, X, y,
                    cv=logo, groups=groups,
                    scoring={"roc_auc": "roc_auc"},
                    n_jobs=1, return_estimator=False, return_train_score=False
                )
                mean_roc = float(np.mean(scores["test_roc_auc"]))
                if mean_roc > best_score:
                    best_score, best_feat = mean_roc, f
            selected.append(best_feat)
            remaining.remove(best_feat)
            # (optional) print progress
            print(f"[{model_name}] step {step}/{k_max}: +{feat_cols[best_feat]} (ROC AUC={best_score:.4f})")

        # Save order as wavelengths names
        order_path = out_dir / f"greedy_order_{model_name}.txt"
        with open(order_path, "w", encoding="utf-8") as f:
            for idx in selected:
                f.write(str(feat_cols[idx]) + "\n")

        # ---- EVALUATE METRICS PER k FOR THIS ORDER ----
        rows = []
        t0_all = time.time()
        for k in range(1, k_max + 1):
            cols = selected[:k]
            steps = []
            if balance_per_fold:
                steps.append(("sampler", RandomUnderSampler(random_state=42)))
            steps += [
                ("sel", ColumnSelector(cols)),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
            pipe = (ImbPipeline if balance_per_fold else Pipeline)(steps=steps)
            t0 = time.time()
            cvres = cross_validate(
                pipe, X, y,
                cv=logo, groups=groups,
                scoring=SCORING,
                n_jobs=1, return_estimator=False, return_train_score=False
            )
            fit_time = float(np.mean(cvres["fit_time"]))
            rows.append({
                "Model": model_name, "k": k,
                "Mean Acc.": float(np.mean(cvres["test_accuracy"])),
                "Mean ROC AUC": float(np.mean(cvres["test_roc_auc"])),
                "Mean PR AUC": float(np.mean(cvres["test_pr_auc"])),
                "Mean F1": float(np.mean(cvres["test_f1"])),
                "Prec. (Cracked)": float(np.mean(cvres["test_precision"])),
                "Recall (Cracked)": float(np.mean(cvres["test_recall"])),
                "Train Time (s)": fit_time,
            })

        res = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        # deltas
        res["delta_roc_auc"] = res["Mean ROC AUC"].diff().fillna(0.0)
        res["delta_pr_auc"]  = res["Mean PR AUC"].diff().fillna(0.0)
        res["delta_accuracy"] = res["Mean Acc."].diff().fillna(0.0)

        # Save CSV
        csv_path = out_dir / f"greedy_results_{model_name}.csv"
        res.to_csv(csv_path, index=False)

        # Plot marginal ROC AUC
        plt.figure(figsize=(7,5))
        plt.plot(res["k"], res["delta_roc_auc"], marker="o")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(f"Marginal Gain ΔROC AUC vs. k (LOGO) – {model_name}")
        plt.xlabel("k (number of wavelengths)")
        plt.ylabel("ΔROC AUC (k - k-1)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"greedy_marginal_gain_roc_auc_{model_name}.png")
        plt.close()

        # (optional) PR AUC / Accuracy marginal plots
        for col, fname in [("delta_pr_auc", "pr_auc"), ("delta_accuracy", "accuracy")]:
            plt.figure(figsize=(7,5))
            plt.plot(res["k"], res[col], marker="o")
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.title(f"Marginal Gain Δ{col.replace('delta_','').upper()} vs. k – {model_name}")
            plt.xlabel("k")
            plt.ylabel(col)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"greedy_marginal_gain_{fname}_{model_name}.png")
            plt.close()

def save_cluster_class_counts(dataset_path: Path, out_dir: Path):
    """
    Create a table of sample counts per cluster_id and class label (0/1),
    plus totals per cluster and overall, and save to CSV.
    """
    df = pd.read_csv(dataset_path)
    # extract cluster_id like above
    def _extract_cluster_id(hs_dir: str) -> str:
        parts = os.path.normpath(hs_dir).split(os.sep)
        return parts[-3] if len(parts) >= 3 else "unknown"
    df["cluster_id"] = df["hs_dir"].apply(_extract_cluster_id)
    # count table
    ct = (df.groupby(["cluster_id", "label"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "count_class_0", 1: "count_class_1"}))
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    ct.to_csv(out_dir / "cluster_class_counts.csv")
    print(ct.head())

def main():
    base_path = Path(os.getenv("BASE_PATH", "."))
    dataset_path = base_path / os.getenv("DATASET_LDA_PATH", "YOUR_DATA.csv")  # adjust if needed

    # 1) Greedy forward selection (LOGO) – order + metrics + plots
    greedy_dir = Path(__file__).parent / "results_logo_greedy_forward"
    run_greedy_forward_logo(dataset_path=dataset_path, out_dir=greedy_dir,
                            balance_per_fold=True, k_max=12)

    # 2) Cluster×Class counts
    counts_dir = Path(__file__).parent / "results_aux"
    save_cluster_class_counts(dataset_path=dataset_path, out_dir=counts_dir)

if __name__ == "__main__":
    main()
