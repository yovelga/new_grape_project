# run_logo_wavelength_sensitivity.py
# ------------------------------------------------------------------------------------
# LOGO wavelength-sensitivity analysis for HSI grape crack detection.
# - Validates LOGO folds (removes single-class train/test splits).
# - Two feature selection modes: "kbest" (ANOVA F) and "sfs" (forward SFS).
# - Models: LDA, Logistic Regression.
# - k = 1..20, metrics: Accuracy, ROC AUC, PR AUC (AP), Precision, Recall, F1, fit-time.
# - Saves per-k CSV + LaTeX, a master CSV, and plots:
#     * Marginal Gain (ΔROC AUC, ΔPR AUC, ΔAccuracy) vs k
#     * Performance vs k (absolute)
#     * Boxplots per k (ROC AUC)
#     * ROC & PR curves for a chosen k (best/target)
#     * Feature frequency heatmap (for SFS)
#     * Cumulative ΔMetric vs k
# - Zero leakage: everything in a Pipeline and evaluated only inside CV.
# - Deterministic: random_state=42 whenever applicable.
# ------------------------------------------------------------------------------------

SPRINT = False  # set to True for a quick SPRINT run!

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from collections import Counter
import csv

# ----- SPRINT/FULL config handling -----
class Config:
    pass

def apply_fast_subset(df, group_col, label_col, max_groups=3, rows_per_class=200, seed=42):
    # Keep up to max_groups, then for each group, up to rows_per_class per class.
    np.random.seed(seed)
    unique_groups = pd.unique(df[group_col])
    keep_groups = unique_groups[:max_groups]
    sub = df[df[group_col].isin(keep_groups)].copy()
    frames = []
    for group in keep_groups:
        group_df = sub[sub[group_col]==group]
        for clazz in sorted(group_df[label_col].unique()):
            sdf = group_df[group_df[label_col]==clazz]
            n = min(rows_per_class, len(sdf))
            frames.append(sdf.sample(n=n, random_state=seed) if n < len(sdf) else sdf)
    return pd.concat(frames, ignore_index=True)

def make_logo_splits(X, y, groups, max_folds=None):
    logo = LeaveOneGroupOut()
    all_splits = [(tr, te) for tr, te in logo.split(np.zeros(len(y)), y, groups)]
    valid = []
    for tr, te in all_splits:
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            valid.append((tr, te))
        if max_folds is not None and len(valid) >= max_folds:
            break
    if not valid:
        raise RuntimeError("No valid LOGO folds remain after filtering.")
    return valid

def get_config_from_sprint(sprint, main_models=None, main_selector_factory=None):
    config = Config()
    if sprint:
        config.K_LIST = range(1, 6)  # k = 1 to 5 in sprint mode
        config.MODELS = {"LDA": LinearDiscriminantAnalysis()}
        # Make the lambda signature match the "full" version for compatibility
        config.SELECTOR_FACTORY = lambda model, k, mode=None: SelectKBest(f_classif, k=k)
        config.MAX_FOLDS = 2
        config.DO_PLOTS = True  # show/save all plots in sprint mode
        config.FAST_SUBSET = {"max_groups":3, "rows_per_class":200}
        config.RESULTS_PATH = Path(__file__).parent / "results" / "fast"
    else:
        # Fallback to project defaults, if not specified show classical full experiment setup
        config.K_LIST = range(1, 21)
        config.MODELS = main_models or {
            "LDA": LinearDiscriminantAnalysis(solver="svd"),
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        }
        config.SELECTOR_FACTORY = main_selector_factory or (
            lambda model, k, mode: (
                SelectKBest(f_classif, k=k)
                if mode == "kbest"
                else SequentialFeatureSelector(
                    model, n_features_to_select=k, direction="forward",
                    scoring="roc_auc",
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    n_jobs=1
                )
            )
        )
        config.MAX_FOLDS = None
        config.DO_PLOTS = True
        config.FAST_SUBSET = None
        config.RESULTS_PATH = Path(__file__).parent / "results" / "full"
    return config



from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


# ------------------------------
# Logging & warnings
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120


# ------------------------------
# Data helpers
# ------------------------------
def _extract_group_from_hs_dir(hs_dir: str) -> str:
    """
    Extract a stable per-image cluster/group id from hs_dir path.
    Adjust if your structure differs. Here we use parent-of-parent folder name.
    """
    parts = os.path.normpath(str(hs_dir)).split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_xyg(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Expect columns:
      - spectral features: columns whose names end with "nm"
      - y label column: "label" (0/1)
      - group-source column: "hs_dir" (string path-like)
    """
    df = pd.read_csv(dataset_path)
    feat_cols = [c for c in df.columns if str(c).endswith("nm")]
    if "label" not in df.columns or "hs_dir" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'hs_dir' columns.")
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()
    groups = df["hs_dir"].apply(_extract_group_from_hs_dir).to_numpy()
    return X, y, groups, feat_cols


def get_valid_logo_splits(y: np.ndarray, groups: np.ndarray):
    """
    Build LOGO splits and filter any split where either train or test has single class.
    """
    logo = LeaveOneGroupOut()
    valid = []
    n_all = logo.get_n_splits(groups=groups)
    for tr, te in logo.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            valid.append((tr, te))
    if len(valid) == 0:
        raise RuntimeError("No valid LOGO folds remain after filtering.")
    if len(valid) < n_all:
        logging.warning(f"Removed {n_all - len(valid)} invalid LOGO folds (single-class train/test). Using {len(valid)}.")
    else:
        logging.info(f"All {n_all} LOGO folds are valid.")
    return valid


# ------------------------------
# Pipelines
# ------------------------------
def make_pipeline(
    model, *,
    k: int,
    mode: str,
    balance_per_fold: bool,
    feat_names: List[str] = None
):
    """
    mode: "kbest" or "sfs".
    - kbest uses SelectKBest(f_classif)
    - sfs uses SequentialFeatureSelector(forward, scoring=roc_auc, inner 3-fold CV)
    """
    steps = []
    if balance_per_fold:
        steps.append(("sampler", RandomUnderSampler(random_state=42)))

    steps.append(("scaler", StandardScaler()))

    if mode == "kbest":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif mode == "sfs":
        selector = SequentialFeatureSelector(
            model, n_features_to_select=k, direction="forward",
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=1  # avoid nested parallelism
        )
    else:
        raise ValueError("mode must be 'kbest' or 'sfs'")

    steps.append(("selector", selector))
    steps.append(("clf", model))

    pipe_cls = ImbPipeline if balance_per_fold else Pipeline
    return pipe_cls(steps=steps)


# ------------------------------
# Plot helpers
# ------------------------------


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance_vs_k(master_df: pd.DataFrame, out_dir: Path):
    metrics = [("Mean Acc.", "Accuracy"), ("Mean ROC AUC", "ROC AUC"), ("Mean PR AUC", "PR AUC")]
    for col, title in metrics:
        plt.figure(figsize=(8, 5))
        for m, sub in master_df.groupby("Model"):
            plt.plot(sub["k"], sub[col], marker="o", label=m)
        plt.xlabel("k (Number of selected wavelengths)")
        plt.ylabel(title)
        plt.title(f"{title} vs k")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        _savefig(out_dir / f"perf_vs_k__{col.replace(' ', '_').lower()}.png")


def plot_marginal_gain(master_df: pd.DataFrame, out_dir: Path):
    for metric in ["Mean ROC AUC", "Mean PR AUC", "Mean Acc."]:
        col = f"Δ {metric}"
        plt.figure(figsize=(8, 5))
        for m, sub in master_df.groupby("Model"):
            plt.plot(sub["k"], sub[col], marker="o", label=m)
        plt.axhline(0, ls="--", lw=1, c="grey")
        plt.xlabel("k")
        plt.ylabel(col)
        plt.title(f"Marginal Gain {metric} vs k")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        _savefig(out_dir / f"marginal_gain__{metric.replace(' ', '_').lower()}.png")


def plot_boxplot_per_k(cvres_store: Dict[int, Dict[str, List[float]]], metric_key: str, out_dir: Path):
    """
    cvres_store[k][metric_key] -> list of per-fold values.
    """
    ks = sorted(cvres_store.keys())
    data = [cvres_store[k][metric_key] for k in ks]

    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(
        data,
        tick_labels=ks,
        showmeans=True,
        meanprops=dict(marker='^', markersize=6),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black')
    )
    plt.xlabel("k")
    plt.ylabel(metric_key)
    plt.title(f"Boxplot across LOGO folds: {metric_key} by k")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Legend explaining the elements
    legend_handles = [
        Patch(facecolor='orange', edgecolor='black', label='IQR (Box)'),
        Line2D([0],[0], color='black', lw=1, label='Whiskers / Median'),
        Line2D([0],[0], marker='^', linestyle='None', label='Mean', markersize=6),
        Line2D([0],[0], marker='o', linestyle='None', label='Outliers', markersize=4)
    ]
    plt.legend(handles=legend_handles, loc='lower right', frameon=True, fontsize=10)

    _savefig(out_dir / f"boxplot__{metric_key.replace(' ', '_').lower()}.png")


def _mean_roc_pr_curves(y_true_list, y_score_list, out_dir: Path, tag: str):
    """
    Plot per-fold ROC/PR curves (faint) and a mean curve (thick) + clear legend.
    """
    fpr_grid = np.linspace(0, 1, 200)
    rec_grid = np.linspace(0, 1, 200)

    # --- ROC ---
    tprs = []
    aucs = []
    plt.figure(figsize=(6, 5))
    for yt, ys in zip(y_true_list, y_score_list):
        fpr, tpr, _ = roc_curve(yt, ys)
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, alpha=0.25, color='gray', lw=1)

    mean_tpr = np.mean(np.stack(tprs), axis=0)
    mean_auc = auc(fpr_grid, mean_tpr)
    mean_line, = plt.plot(fpr_grid, mean_tpr, lw=3, color='mediumpurple')
    chance_line, = plt.plot([0, 1], [0, 1], 'k--', lw=1)

    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (mean over folds) — {tag}")
    plt.grid(True, linestyle="--", alpha=0.6)

    roc_handles = [
        Line2D([0],[0], color='gray', lw=1, alpha=0.25, label='Per-fold ROC'),
        Line2D([0],[0], color='mediumpurple', lw=3, label=f'Mean ROC (AUC={mean_auc:.3f})'),
        Line2D([0],[0], color='black', lw=1, ls='--', label='Chance (AUC=0.5)'),
    ]
    plt.legend(handles=roc_handles, loc='lower right', frameon=True, fontsize=10)

    _savefig(out_dir / f"roc_mean__{tag}.png")

    # --- PR ---
    prs = []
    aps = []
    plt.figure(figsize=(6, 5))
    for yt, ys in zip(y_true_list, y_score_list):
        prec, rec, _ = precision_recall_curve(yt, ys)
        prs.append(np.interp(rec_grid, rec[::-1], prec[::-1]))
        aps.append(average_precision_score(yt, ys))
        plt.plot(rec, prec, alpha=0.25, color='gray', lw=1)

    mean_prec = np.mean(np.stack(prs), axis=0)
    mean_ap = np.mean(aps)
    plt.plot(rec_grid, mean_prec, lw=3, color='mediumpurple')

    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision–Recall (mean) — {tag}")
    plt.grid(True, linestyle="--", alpha=0.6)

    pr_handles = [
        Line2D([0],[0], color='gray', lw=1, alpha=0.25, label='Per-fold PR'),
        Line2D([0],[0], color='mediumpurple', lw=3, label=f'Mean PR (AP={mean_ap:.3f})'),
    ]
    plt.legend(handles=pr_handles, loc='lower left', frameon=True, fontsize=10)

    _savefig(out_dir / f"pr_mean__{tag}.png")



def _build_matrix(freq_df_by_k: Dict[int, pd.Series], feat_order: List[str] = None):
    """
    freq_df_by_k: {k: pd.Series(index=feature_name, value=fold_frequency)}
    Returns: (mat, all_feats, ks) where mat shape is [n_features x n_k].
    """
    all_feats = feat_order or sorted({f for s in freq_df_by_k.values() for f in s.index})
    ks = sorted(freq_df_by_k.keys())
    mat = np.zeros((len(all_feats), len(ks)), dtype=float)
    for j, k in enumerate(ks):
        s = freq_df_by_k[k].reindex(all_feats).fillna(0).astype(float)
        mat[:, j] = s.to_numpy()
    return mat, all_feats, ks


def plot_feature_frequency_heatmap_full(
    freq_df_by_k: Dict[int, pd.Series],
    out_dir: Path,
    feat_order: List[str] = None,
    y_fontsize: int = 6,
    pdf_name: str = "feature_frequency_heatmap_full.pdf",
):
    """
    Full heatmap (all wavelengths). Optimized for print-quality PDF in the appendix.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mat, all_feats, ks = _build_matrix(freq_df_by_k, feat_order)

    # Figure size scales with content so labels are readable
    H = max(8, len(all_feats) * 0.18 + 2)
    W = max(8, len(ks) * 0.35 + 2)

    fig, ax = plt.subplots(figsize=(W, H))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Fold frequency")

    ax.set_yticks(np.arange(len(all_feats)))
    ax.set_yticklabels(all_feats, fontsize=y_fontsize)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Wavelengths")
    ax.set_title("Feature frequency across k (SFS) — Full")

    # Layout and subtle grid for readability
    fig.subplots_adjust(left=0.28, right=0.96, top=0.95, bottom=0.10)
    ax.set_xticks(np.arange(-0.5, len(ks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(all_feats), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.2, alpha=0.4)

    # Save as PDF to preserve crisp text at any zoom
    fig.savefig(out_dir / pdf_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_frequency_heatmap_topN(
    freq_df_by_k: Dict[int, pd.Series],
    out_dir: Path,
    feat_order: List[str] = None,
    top_n: int = 40,
    sort_by: str = "total",   # "total" (sum across k) or "kmax" (peak across k)
    y_fontsize: int = 9,
    png_name: str = "feature_frequency_heatmap_topN.png",
):
    """
    Readable Top-N heatmap (default N=40). Sorted by overall importance.
    Ideal for the main chapter.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_full, all_feats, ks = _build_matrix(freq_df_by_k, feat_order)

    # Rank features
    if sort_by == "kmax":
        scores = mat_full.max(axis=1)
    else:
        scores = mat_full.sum(axis=1)
    idx_sorted = np.argsort(-scores)  # descending
    idx_keep = idx_sorted[:min(top_n, len(all_feats))]
    feats_keep = [all_feats[i] for i in idx_keep]
    mat = mat_full[idx_keep, :]

    # Scaled figure for readability
    h = max(6, len(feats_keep) * 0.35 + 1)
    w = max(6, len(ks) * 0.35 + 2)

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Fold frequency")

    ax.set_yticks(np.arange(len(feats_keep)))
    ax.set_yticklabels(feats_keep, fontsize=y_fontsize)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Wavelengths")
    ax.set_title(f"Feature frequency across k (SFS) — Top-{len(feats_keep)}")

    fig.subplots_adjust(left=0.30, right=0.96, top=0.95, bottom=0.10)
    ax.set_xticks(np.arange(-0.5, len(ks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feats_keep), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.2, alpha=0.5)

    fig.savefig(out_dir / png_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_delta(master_df: pd.DataFrame, out_dir: Path):
    for metric in ["Mean ROC AUC", "Mean PR AUC", "Mean Acc."]:
        plt.figure(figsize=(8, 5))
        for m, sub in master_df.groupby("Model"):
            base = sub.iloc[0][metric]
            plt.plot(sub["k"], sub[metric] - base, marker="o", label=m)
        plt.xlabel("k"); plt.ylabel(f"{metric} - {metric}(k=1)")
        plt.title(f"Cumulative Δ{metric} vs k")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        _savefig(out_dir / f"cumulative_delta__{metric.replace(' ', '_').lower()}.png")

# ------------------------------
# Feature selection tracking & consensus helpers
# ------------------------------


def _get_selected_feature_names(fitted_pipeline, feature_names):
    """
    Supports both SelectKBest and SFS (sklearn). If it's SFS from mlxtend – uses k_feature_idx_.
    """
    try:
        selector = fitted_pipeline.named_steps.get("selector", None)
    except Exception:
        selector = None

    if selector is None:
        return []

    # sklearn selectors (SelectKBest / SequentialFeatureSelector) – has get_support
    if hasattr(selector, "get_support"):
        mask = selector.get_support()
        return [f for f, keep in zip(feature_names, mask) if keep]

    # mlxtend SFS
    if hasattr(selector, "k_feature_idx_"):
        idxs = list(selector.k_feature_idx_)
        return [feature_names[i] for i in idxs]

    return []

def _save_fold_selections(out_dir, model_name, k, fold_selections):
    path = out_dir / f"selected_features_per_fold__{model_name}__k{k}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold_idx", "selected_features"])
        for i, feats in enumerate(fold_selections):
            w.writerow([i, "|".join(map(str, feats))])

def _save_frequency(out_dir, model_name, k, freq_counter: Counter):
    path = out_dir / f"selected_frequency__{model_name}__k{k}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "count"])
        for feat, cnt in freq_counter.most_common():
            w.writerow([feat, cnt])

def _consensus_top_k(freq_counter: Counter, k: int):
    # Tie-breaking by feature name (or convert to index if your names are numeric)
    ranked = sorted(freq_counter.items(), key=lambda x: (-x[1], str(x[0])))
    return [feat for feat, _ in ranked[:k]]


# ------------------------------
# Runner
# ------------------------------
class LogoRunner:
    def __init__(self, dataset_path: Path, run_name: str, method: str = "sfs", balance: bool = True):
        """
        method: 'kbest' or 'sfs'
        """
        self.dataset_path = dataset_path
        self.method = method.lower()
        self.balance = balance
        base_dir = Path(__file__).parent
        self.out_dir = base_dir / "results" / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results → {self.out_dir}")

        self.models = {
            "LDA": LinearDiscriminantAnalysis(solver="svd"),
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        }
        self.scoring = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }

        self.X, self.y, self.groups, self.feat_cols = load_xyg(self.dataset_path)
        self.logo_folds = get_valid_logo_splits(self.y, self.groups)

    def run_tables_and_master(self, k_from: int = 1, k_to: int = 20) -> pd.DataFrame:
        master_rows = []
        per_k_fold_metrics = {}  # for boxplots
        for k in tqdm(range(k_from, k_to + 1), desc=f"Processing k ({self.method})"):
            rows = []
            per_k_fold_metrics[k] = {"roc_auc": []}  # collect fold-level ROC AUCs across models

            for model_name, model in self.models.items():
                pipe = make_pipeline(
                    model, k=k, mode=self.method, balance_per_fold=self.balance, feat_names=self.feat_cols
                )
                cvres = cross_validate(
                    pipe, self.X, self.y,
                    cv=self.logo_folds,
                    scoring=self.scoring,
                    n_jobs=-1,
                    return_estimator=True,
                    return_train_score=False
                )

                # Feature tracking: extract per-fold selections
                feature_names = list(self.feat_cols)
                fold_selections = []
                for est in cvres["estimator"]:
                    feats = _get_selected_feature_names(est, feature_names)
                    fold_selections.append(feats)
                freq = Counter([f for feats in fold_selections for f in feats])
                consensus_feats = _consensus_top_k(freq, k)
                _save_fold_selections(self.out_dir, model_name, k, fold_selections)
                _save_frequency(self.out_dir, model_name, k, freq)

                row = {
                    "Method": self.method,
                    "Model": model_name,
                    "k": k,
                    "Mean Acc.": float(np.mean(cvres["test_accuracy"])),
                    "Mean ROC AUC": float(np.mean(cvres["test_roc_auc"])),
                    "Mean PR AUC": float(np.mean(cvres["test_pr_auc"])),
                    "Mean F1": float(np.mean(cvres["test_f1"])),
                    "Prec. (Cracked)": float(np.mean(cvres["test_precision"])),
                    "Recall (Cracked)": float(np.mean(cvres["test_recall"])),
                    "Train Time (s)": float(np.mean(cvres["fit_time"])),
                    # New: consensus features (by frequency/tie-break)
                    "Selected_Consensus_k": "|".join(map(str, consensus_feats)),
                }
                # Add: Selected_Fold_0, Selected_Fold_1, ... as columns
                for i, feats in enumerate(fold_selections):
                    row[f"Selected_Fold_{i}"] = "|".join(map(str, feats))
                # Add Top5_By_Frequency for this (model, k) combo
                top5_freq = "|".join(map(str, [f for f, _ in freq.most_common(5)]))
                row["Top5_By_Frequency"] = top5_freq

                rows.append(row)
                master_rows.append(row)

                # store fold ROC AUCs for boxplot
                per_k_fold_metrics[k]["roc_auc"].extend(list(cvres["test_roc_auc"]))

                # For SFS: store frequency per k (for heatmap/backward compat)
                if self.method == "sfs":
                    freq_df = pd.Series(freq, dtype=int).reindex(self.feat_cols, fill_value=0)
                    k_freq_dir = self.out_dir / "wavelengths_selected"
                    k_freq_dir.mkdir(exist_ok=True)
                    freq_df.sort_values(ascending=False).to_frame("fold_frequency").to_csv(
                        k_freq_dir / f"wavelengths_selected_{model_name}_k{k:02d}.csv"
                    )

            # Save per-k table (CSV + LaTeX)
            k_dir = self.out_dir / f"k_{k:02d}"
            k_dir.mkdir(exist_ok=True)
            kdf = pd.DataFrame(rows)
            kdf.to_csv(k_dir / "table_k.csv", index=False)
            try:
                kdf.to_latex(k_dir / "table_k.tex", index=False, float_format="%.4f",
                             caption=f"{self.method.upper()} Performance (k={k}, LOGO{' Balanced' if self.balance else ''})",
                             label=f"tab:{self.method}_k{k:02d}", position="ht!")
            except Exception as e:
                logging.warning(f"LaTeX export failed for k={k}: {e}")

        master_df = pd.DataFrame(master_rows).sort_values(["Method", "Model", "k"]).reset_index(drop=True)

        # deltas per model (marginal gain)
        for metric in ["Mean ROC AUC", "Mean PR AUC", "Mean Acc."]:
            master_df[f"Δ {metric}"] = master_df.groupby(["Method", "Model"])[metric].diff().fillna(0.0)

        master_df.to_csv(self.out_dir / "all_k_master.csv", index=False)
        return master_df, per_k_fold_metrics

    def plot_all(self, master_df: pd.DataFrame, per_k_fold_metrics: Dict[int, Dict[str, List[float]]], best_k: int = 10):
        # Absolute performance vs k
        plot_performance_vs_k(master_df, self.out_dir)

        # Marginal gains
        plot_marginal_gain(master_df, self.out_dir)

        # Boxplots per k (ROC AUC)
        plot_boxplot_per_k(per_k_fold_metrics, "roc_auc", self.out_dir)

        # Cumulative Δ
        plot_cumulative_delta(master_df, self.out_dir)

        # Mean ROC/PR for chosen k (re-run once at best_k to get fold-level scores)
        for model_name, model in self.models.items():
            tag = f"{self.method}_{model_name}_k{best_k:02d}"
            pipe = make_pipeline(model, k=best_k, mode=self.method, balance_per_fold=self.balance)
            # collect per-fold probabilities via cross_validate? -> use manual loop to also grab y_true per fold
            y_true_list, y_score_list = [], []
            for tr, te in self.logo_folds:
                est = pipe.fit(self.X[tr], self.y[tr])
                # Prob of positive class (assume class 1)
                if hasattr(est.named_steps["clf"], "predict_proba"):
                    ys = est.named_steps["clf"].predict_proba(
                        est.named_steps["selector"].transform(
                            est.named_steps["scaler"].transform(self.X[te])
                        )
                    )[:, 1]
                else:
                    # decision_function fallback -> map to [0,1]
                    dfc = est.named_steps["clf"].decision_function(
                        est.named_steps["selector"].transform(
                            est.named_steps["scaler"].transform(self.X[te])
                        )
                    )
                    ys = (dfc - dfc.min()) / (dfc.max() - dfc.min() + 1e-9)
                y_true_list.append(self.y[te])
                y_score_list.append(ys)
            _mean_roc_pr_curves(y_true_list, y_score_list, self.out_dir, tag)

        # Feature frequency heatmap (SFS only)
        if self.method == "sfs":
            # Load back all per-k frequency files and plot a combined heatmap
            freq_by_k = {}
            freq_dir = self.out_dir / "wavelengths_selected"
            if freq_dir.exists():
                # Merge models' frequencies per k (sum)
                files = list(freq_dir.glob("wavelengths_selected_*_k*.csv"))
                by_k_model = {}
                for fp in files:
                    k = int(str(fp.stem).split("_k")[-1])
                    s = pd.read_csv(fp, index_col=0)["fold_frequency"]
                    by_k_model.setdefault(k, []).append(s)
                for k, arr in by_k_model.items():
                    # sum frequencies from both models (if both exist)
                    ssum = pd.concat(arr, axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
                    freq_by_k[k] = ssum
                if len(freq_by_k) > 19:
                    # Clean legend labels for plotting (remove 'wl_' and 'nm')
                    cleaned = {k: s.rename(index=lambda x: str(x).replace("wl_", "").replace("nm","")) for k, s in freq_by_k.items()}
                    plot_feature_frequency_heatmap_full(cleaned, self.out_dir)
                    plot_feature_frequency_heatmap_topN(cleaned, self.out_dir, top_n=40)
# ------------------------------
# main
# ------------------------------
def main():
    # --------- ENV and Mode Banner ---------
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        logging.error(f".env not found at {env_path}. See template below.")
        return
    load_dotenv(env_path)

    base_path = os.getenv("BASE_PATH", "").strip()
    data_rel = os.getenv("DATASET_LDA_PATH", "").strip()
    run_name = os.getenv("RUN_NAME", "logo_sensitivity_v1").strip()
    method = os.getenv("FEATURE_METHOD", "sfs").strip().lower()  # "sfs" or "kbest"
    balance = os.getenv("BALANCE_PER_FOLD", "true").strip().lower() in {"1", "true", "yes", "y"}
    best_k = int(os.getenv("BEST_K_FOR_CURVES", "10"))

    config = get_config_from_sprint(
        SPRINT
    )

    mode_str = "SPRINT" if SPRINT else "FULL"
    print()
    print("="*55)
    print(f"            LOGO Wavelength Sensitivity")
    print(f"                  MODE={mode_str}")
    print("="*55)

    if not base_path or not data_rel:
        logging.error("Please set BASE_PATH and DATASET_LDA_PATH in .env")
        return

    dataset_path = Path(base_path) / data_rel
    if not dataset_path.is_file():
        logging.error(f"Dataset not found: {dataset_path}")
        return

    # --------- Data load & (optionally) fast subset ---------
    df = pd.read_csv(dataset_path)
    if config.FAST_SUBSET is not None:
        print("Applying fast subset...")
        df = apply_fast_subset(df, group_col="hs_dir", label_col="label", **config.FAST_SUBSET)
    feat_cols = [c for c in df.columns if str(c).endswith("nm")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()
    groups = df["hs_dir"].apply(_extract_group_from_hs_dir).to_numpy()

    # --------- LOGO splits ---------
    logo_folds = make_logo_splits(X, y, groups, max_folds=config.MAX_FOLDS)
    print(f"Using {len(logo_folds)} LOGO folds. (SPRINT={SPRINT})")

    # --------- Model selection setup ---------
    # Support: both kbest and sfs modes (preserves existing behavior in FULL)
    MODELS = config.MODELS
    K_LIST = config.K_LIST
    SELECTOR_FACTORY = config.SELECTOR_FACTORY
    OUT_DIR = config.RESULTS_PATH
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    # --------- Main experiment loop (per k, per model) ---------
    master_rows = []
    per_k_fold_metrics = {}  # for boxplots
    for k in tqdm(K_LIST, desc=f"Processing k"):
        rows = []
        per_k_fold_metrics[k] = {"roc_auc": []}

        for model_name, model in MODELS.items():
            if callable(SELECTOR_FACTORY):
                # Always use method as third arg for compatibility
                selector = SELECTOR_FACTORY(model, k, method)
            else: # fallback
                selector = SelectKBest(f_classif, k=k)

            # Compose pipeline as: scaler → selector → classifier
            steps = []
            if balance:
                steps.append(("sampler", RandomUnderSampler(random_state=42)))
            steps.append(("scaler", StandardScaler()))
            steps.append(("selector", selector))
            steps.append(("clf", model))
            pipe_cls = ImbPipeline if balance else Pipeline
            pipe = pipe_cls(steps=steps)

            cvres = cross_validate(
                pipe, X, y,
                cv=logo_folds,
                scoring=scoring,
                n_jobs=-1,
                return_estimator=True,
                return_train_score=False
            )
            # Feature tracking as in above
            feature_names = list(feat_cols)
            fold_selections = []
            for est in cvres["estimator"]:
                feats = _get_selected_feature_names(est, feature_names)
                fold_selections.append(feats)
            freq = Counter([f for feats in fold_selections for f in feats])
            consensus_feats = _consensus_top_k(freq, k)
            _save_fold_selections(OUT_DIR, model_name, k, fold_selections)
            _save_frequency(OUT_DIR, model_name, k, freq)

            row = {
                "Method": method,
                "Model": model_name,
                "k": k,
                "Mean Acc.": float(np.mean(cvres["test_accuracy"])),
                "Mean ROC AUC": float(np.mean(cvres["test_roc_auc"])),
                "Mean PR AUC": float(np.mean(cvres["test_pr_auc"])),
                "Mean F1": float(np.mean(cvres["test_f1"])),
                "Prec. (Cracked)": float(np.mean(cvres["test_precision"])),
                "Recall (Cracked)": float(np.mean(cvres["test_recall"])),
                "Train Time (s)": float(np.mean(cvres["fit_time"])),
                "Selected_Consensus_k": "|".join(map(str, consensus_feats)),
            }
            for i, feats in enumerate(fold_selections):
                row[f"Selected_Fold_{i}"] = "|".join(map(str, feats))
            top5_freq = "|".join(map(str, [f for f, _ in freq.most_common(5)]))
            row["Top5_By_Frequency"] = top5_freq

            rows.append(row)
            master_rows.append(row)

            per_k_fold_metrics[k]["roc_auc"].extend(list(cvres["test_roc_auc"]))

            # For SFS: store frequency per k for downstream plots (no harm if always)
            freq_df = pd.Series(freq, dtype=int).reindex(feature_names, fill_value=0)
            k_freq_dir = OUT_DIR / "wavelengths_selected"
            k_freq_dir.mkdir(exist_ok=True)
            freq_df.sort_values(ascending=False).to_frame("fold_frequency").to_csv(
                k_freq_dir / f"wavelengths_selected_{model_name}_k{k:02d}.csv"
            )

        # Save per-k table (CSV + LaTeX)
        k_dir = OUT_DIR / f"k_{k:02d}"
        k_dir.mkdir(exist_ok=True)
        kdf = pd.DataFrame(rows)
        kdf.to_csv(k_dir / "table_k.csv", index=False)
        try:
            kdf.to_latex(k_dir / "table_k.tex", index=False, float_format="%.4f",
                        caption=f"{method.upper()} Performance (k={k}, LOGO{' Balanced' if balance else ''})",
                        label=f"tab:{method}_k{k:02d}", position="ht!")
        except Exception as e:
            logging.warning(f"LaTeX export failed for k={k}: {e}")

    master_df = pd.DataFrame(master_rows).sort_values(["Method", "Model", "k"]).reset_index(drop=True)

    for metric in ["Mean ROC AUC", "Mean PR AUC", "Mean Acc."]:
        master_df[f"Δ {metric}"] = master_df.groupby(["Method", "Model"])[metric].diff().fillna(0.0)

    # Save master table
    master_df.to_csv(OUT_DIR / "all_k_master.csv", index=False)

    # --------- Optionally plotting ---------
    if config.DO_PLOTS:
        # Performance vs k, marginal gains, boxplots, cumulative delta
        plot_performance_vs_k(master_df, OUT_DIR)
        plot_marginal_gain(master_df, OUT_DIR)
        plot_boxplot_per_k(per_k_fold_metrics, "roc_auc", OUT_DIR)
        plot_cumulative_delta(master_df, OUT_DIR)

        # ROC/PR curves for best k
        for model_name, model in MODELS.items():
            tag = f"{method}_{model_name}_k{best_k:02d}"
            # FIX: Always pass three arguments to SELECTOR_FACTORY
            selector = SELECTOR_FACTORY(model, best_k, method)
            steps = []
            if balance:
                steps.append(("sampler", RandomUnderSampler(random_state=42)))
            steps.append(("scaler", StandardScaler()))
            steps.append(("selector", selector))
            steps.append(("clf", model))
            pipe_cls = ImbPipeline if balance else Pipeline
            pipe = pipe_cls(steps=steps)
            y_true_list, y_score_list = [], []
            for tr, te in logo_folds:
                est = pipe.fit(X[tr], y[tr])
                if hasattr(est.named_steps["clf"], "predict_proba"):
                    ys = est.named_steps["clf"].predict_proba(
                        est.named_steps["selector"].transform(
                            est.named_steps["scaler"].transform(X[te])
                        )
                    )[:, 1]
                else:
                    dfc = est.named_steps["clf"].decision_function(
                        est.named_steps["selector"].transform(
                            est.named_steps["scaler"].transform(X[te])
                        )
                    )
                    ys = (dfc - dfc.min()) / (dfc.max() - dfc.min() + 1e-9)
                y_true_list.append(y[te])
                y_score_list.append(ys)
            _mean_roc_pr_curves(y_true_list, y_score_list, OUT_DIR, tag)

        # Feature frequency heatmap (for SFS)
        # (even if method=="kbest", this will run but just show zeros)
        freq_by_k = {}
        freq_dir = OUT_DIR / "wavelengths_selected"
        if freq_dir.exists():
            files = list(freq_dir.glob("wavelengths_selected_*_k*.csv"))
            by_k_model = {}
            for fp in files:
                kval = int(str(fp.stem).split("_k")[-1])
                s = pd.read_csv(fp, index_col=0)["fold_frequency"]
                by_k_model.setdefault(kval, []).append(s)
            for kval, arr in by_k_model.items():
                # sum frequencies from both models (if both exist)
                ssum = pd.concat(arr, axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
                freq_by_k[kval] = ssum
            if len(freq_by_k) > 0:
                cleaned = {k: s.rename(index=lambda x: str(x).replace("wl_", "").replace("nm","")) for k, s in freq_by_k.items()}
                plot_feature_frequency_heatmap_full(cleaned, OUT_DIR)
                plot_feature_frequency_heatmap_topN(cleaned, OUT_DIR, top_n=40)

    # --------- Logging summary ---------
    print("\n==== FINAL SUMMARY ====")
    for idx, row in master_df.iterrows():
        print(f"Model={row['Model']:<20} k={row['k']:<2} Acc={row['Mean Acc.']:.3f} ROC_AUC={row['Mean ROC AUC']:.3f}  Consensus: {row['Selected_Consensus_k']}")

    print(f"\nCSV saved to: {OUT_DIR / 'all_k_master.csv'}")
    if config.DO_PLOTS:
        print(f"Plots saved to: {OUT_DIR}")

    print("\n"+"="*40+"\n")

if __name__ == "__main__":
    main()
