"""
make_bfs_plots_from_csv.py

Generate thesis-ready plots from BFS CSV logs WITHOUT retraining.

This script reads:
- bfs_log_f1.csv
- bfs_log_prauc.csv

And generates:
- thresholds_summary_f1.csv
- thresholds_summary_prauc.csv
- plots/ folder with many plot variants (PDF + PNG + SVG)

Plots include:
1. Objective metric vs n_features (line)
2. Objective metric vs n_features (scatter + line)
3. Normalized metric vs n_features
4. Score drop percent vs n_features
5. Delta metric (first difference) vs n_features
6. Threshold plot with max, 0.5% drop, 1% drop markers
7. Dual-metric plot (F1 and PR-AUC together)
8. Precision and Recall vs n_features
9. Additional metrics (accuracy, macro_f1, roc_auc)
10. Pareto scatter (PR-AUC vs F1)

Author: Feature Selection Pipeline
Date: February 2026
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Disable warnings
warnings.filterwarnings("ignore")

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[ERROR] matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

# Thesis-ready plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# ==================== THRESHOLD COMPUTATION ====================

def compute_thresholds(
    df: pd.DataFrame,
    objective: str,
    metric_col: str,
) -> Dict:
    """
    Compute threshold points for a metric.
    
    Returns dict with:
    - max_score, n_at_max
    - threshold_0p5, n_at_0p5_drop, score_at_0p5_drop
    - threshold_1p0, n_at_1p0_drop, score_at_1p0_drop
    - recommended_n_features
    """
    # Sort by n_features descending (most features first)
    df_sorted = df.sort_values("n_features", ascending=False).reset_index(drop=True)
    
    n_features = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    
    # Max score (among all iterations)
    max_score = np.nanmax(scores)
    max_idx = np.nanargmax(scores)
    n_at_max = int(n_features[max_idx])
    
    # If there are ties, prefer smallest n_features
    max_mask = scores == max_score
    if max_mask.sum() > 1:
        n_at_max = int(n_features[max_mask].min())
    
    # Thresholds
    threshold_0p5 = max_score * (1 - 0.005)  # 0.5% drop
    threshold_1p0 = max_score * (1 - 0.01)   # 1.0% drop
    
    # Find first n_features where score drops below threshold
    # (moving toward fewer features)
    # We iterate from most features to fewest
    n_at_0p5_drop = None
    score_at_0p5_drop = None
    n_at_1p0_drop = None
    score_at_1p0_drop = None
    
    # Sort ascending by n_features to find "first drop" when reducing features
    df_asc = df.sort_values("n_features", ascending=True).reset_index(drop=True)
    n_asc = df_asc["n_features"].values
    s_asc = df_asc[metric_col].values
    
    # Starting from max features, find where score first drops below threshold
    # Actually we want: when reducing features, find the FIRST n where score < threshold
    # So iterate from high n to low n
    for i in range(len(n_asc) - 1, -1, -1):
        if s_asc[i] <= threshold_0p5 and n_at_0p5_drop is None:
            n_at_0p5_drop = int(n_asc[i])
            score_at_0p5_drop = float(s_asc[i])
        if s_asc[i] <= threshold_1p0 and n_at_1p0_drop is None:
            n_at_1p0_drop = int(n_asc[i])
            score_at_1p0_drop = float(s_asc[i])
    
    # If never dropped below threshold, use the minimum n_features tested
    if n_at_0p5_drop is None:
        n_at_0p5_drop = int(n_asc.min())
        score_at_0p5_drop = float(s_asc[n_asc == n_at_0p5_drop][0]) if len(df_asc) > 0 else max_score
    if n_at_1p0_drop is None:
        n_at_1p0_drop = int(n_asc.min())
        score_at_1p0_drop = float(s_asc[n_asc == n_at_1p0_drop][0]) if len(df_asc) > 0 else max_score
    
    # Recommended: n_at_0p5_drop (conservative)
    recommended = n_at_0p5_drop if n_at_0p5_drop is not None else n_at_max
    
    return {
        "objective_name": objective,
        "metric_name": metric_col,
        "max_score": max_score,
        "n_at_max": n_at_max,
        "threshold_0p5": threshold_0p5,
        "n_at_0p5_drop": n_at_0p5_drop,
        "score_at_0p5_drop": score_at_0p5_drop,
        "threshold_1p0": threshold_1p0,
        "n_at_1p0_drop": n_at_1p0_drop,
        "score_at_1p0_drop": score_at_1p0_drop,
        "recommended_n_features": recommended,
    }


# ==================== PLOT HELPERS ====================

def save_plot(fig, plots_dir: Path, name: str):
    """Save plot in PDF, PNG, and SVG formats."""
    for ext in ["pdf", "png", "svg"]:
        path = plots_dir / f"{name}.{ext}"
        fig.savefig(path, format=ext)
    plt.close(fig)


def add_threshold_markers(ax, thresholds: Dict, n_features: np.ndarray, scores: np.ndarray):
    """Add threshold markers to plot."""
    max_score = thresholds["max_score"]
    n_at_max = thresholds["n_at_max"]
    n_at_0p5 = thresholds["n_at_0p5_drop"]
    n_at_1p0 = thresholds["n_at_1p0_drop"]
    
    ymin, ymax = ax.get_ylim()
    
    # Max marker
    ax.axvline(x=n_at_max, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Max @ n={n_at_max}')
    ax.scatter([n_at_max], [max_score], color='green', s=100, zorder=5, marker='*')
    
    # 0.5% drop marker
    if n_at_0p5 is not None and n_at_0p5 != n_at_max:
        ax.axvline(x=n_at_0p5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'0.5% drop @ n={n_at_0p5}')
    
    # 1.0% drop marker
    if n_at_1p0 is not None and n_at_1p0 != n_at_max and n_at_1p0 != n_at_0p5:
        ax.axvline(x=n_at_1p0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'1.0% drop @ n={n_at_1p0}')
    
    # Horizontal threshold lines
    ax.axhline(y=thresholds["threshold_0p5"], color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=thresholds["threshold_1p0"], color='red', linestyle=':', linewidth=1, alpha=0.5)


# ==================== INDIVIDUAL PLOTS ====================

def plot_metric_vs_n_line(df: pd.DataFrame, metric_col: str, objective: str, 
                          thresholds: Dict, plots_dir: Path):
    """Plot 1: Objective metric vs n_features (line)."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_feat, scores, 'b-', linewidth=2, marker='o', markersize=4)
    
    add_threshold_markers(ax, thresholds, n_feat, scores)
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(f"BFS ({objective.upper()}): {metric_col} vs Number of Features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_metric_vs_n_line")


def plot_metric_vs_n_scatter(df: pd.DataFrame, metric_col: str, objective: str,
                              thresholds: Dict, plots_dir: Path):
    """Plot 2: Objective metric vs n_features (scatter + line)."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(n_feat, scores, c='blue', s=30, alpha=0.6, zorder=3)
    ax.plot(n_feat, scores, 'b-', linewidth=1, alpha=0.5)
    
    add_threshold_markers(ax, thresholds, n_feat, scores)
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(f"BFS ({objective.upper()}): {metric_col} vs Number of Features (Scatter)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_metric_vs_n_scatter")


def plot_normalized_metric(df: pd.DataFrame, metric_col: str, objective: str,
                           thresholds: Dict, plots_dir: Path):
    """Plot 3: Normalized metric (metric / max_metric) vs n_features."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    max_score = thresholds["max_score"]
    
    normalized = scores / max_score if max_score > 0 else scores
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_feat, normalized, 'b-', linewidth=2, marker='o', markersize=4)
    
    # Threshold lines
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Max (100%)')
    ax.axhline(y=0.995, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='99.5% of max')
    ax.axhline(y=0.99, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='99% of max')
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(f"Normalized {metric_col} (fraction of max)")
    ax.set_title(f"BFS ({objective.upper()}): Normalized {metric_col} vs Number of Features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(min(0.95, normalized.min() - 0.01), 1.02)
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_normalized")


def plot_score_drop_percent(df: pd.DataFrame, metric_col: str, objective: str,
                            thresholds: Dict, plots_dir: Path):
    """Plot 4: Score drop percent vs n_features."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    max_score = thresholds["max_score"]
    
    drop_pct = (max_score - scores) / max_score * 100 if max_score > 0 else np.zeros_like(scores)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_feat, drop_pct, 'r-', linewidth=2, marker='o', markersize=4)
    
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='0.5% drop')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='1.0% drop')
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score Drop (%)")
    ax.set_title(f"BFS ({objective.upper()}): Score Drop from Maximum vs Number of Features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(-0.5, max(5, drop_pct.max() + 0.5))
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_score_drop_percent")


def plot_delta_metric(df: pd.DataFrame, metric_col: str, objective: str, plots_dir: Path):
    """Plot 5: Delta metric (first difference) vs n_features."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    
    delta = np.diff(scores)
    n_delta = n_feat[1:]  # n_features for delta values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(n_delta, delta, width=0.8, alpha=0.7, color='blue', edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(f"Delta {metric_col}")
    ax.set_title(f"BFS ({objective.upper()}): Change in {metric_col} per Feature Added")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_delta")


def plot_threshold_summary(df: pd.DataFrame, metric_col: str, objective: str,
                           thresholds: Dict, plots_dir: Path):
    """Plot 6: Full threshold plot with annotations."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    scores = df_sorted[metric_col].values
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(n_feat, scores, 'b-', linewidth=2.5, marker='o', markersize=5, label=metric_col)
    
    # Max point
    n_max = thresholds["n_at_max"]
    s_max = thresholds["max_score"]
    ax.scatter([n_max], [s_max], color='green', s=200, zorder=10, marker='*', label=f'Max: {s_max:.4f} @ n={n_max}')
    ax.annotate(f'Max\n{s_max:.4f}\nn={n_max}', xy=(n_max, s_max), xytext=(n_max + 5, s_max + 0.01),
                fontsize=9, ha='left', arrowprops=dict(arrowstyle='->', color='green'))
    
    # 0.5% drop point
    n_0p5 = thresholds["n_at_0p5_drop"]
    s_0p5 = thresholds["score_at_0p5_drop"]
    if n_0p5 is not None and s_0p5 is not None:
        ax.scatter([n_0p5], [s_0p5], color='orange', s=150, zorder=10, marker='D', 
                   label=f'0.5% drop: {s_0p5:.4f} @ n={n_0p5}')
        ax.axvline(x=n_0p5, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 1.0% drop point
    n_1p0 = thresholds["n_at_1p0_drop"]
    s_1p0 = thresholds["score_at_1p0_drop"]
    if n_1p0 is not None and s_1p0 is not None and n_1p0 != n_0p5:
        ax.scatter([n_1p0], [s_1p0], color='red', s=150, zorder=10, marker='s',
                   label=f'1.0% drop: {s_1p0:.4f} @ n={n_1p0}')
        ax.axvline(x=n_1p0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Threshold lines
    ax.axhline(y=thresholds["threshold_0p5"], color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=thresholds["threshold_1p0"], color='red', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"BFS ({objective.upper()}): {metric_col} with Threshold Markers", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_thresholds")


def plot_dual_metric(df_f1: pd.DataFrame, df_prauc: pd.DataFrame, plots_dir: Path):
    """Plot 7: Dual-metric plot (F1 and PR-AUC together)."""
    df_f1_sorted = df_f1.sort_values("n_features")
    df_prauc_sorted = df_prauc.sort_values("n_features")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # F1 on left axis
    color1 = 'tab:blue'
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("CRACK F1", color=color1)
    ax1.plot(df_f1_sorted["n_features"], df_f1_sorted["crack_f1"], 
             color=color1, linewidth=2, marker='o', markersize=4, label='F1 (optimized for F1)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # PR-AUC on right axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("CRACK PR-AUC", color=color2)
    ax2.plot(df_prauc_sorted["n_features"], df_prauc_sorted["crack_pr_auc"],
             color=color2, linewidth=2, marker='s', markersize=4, label='PR-AUC (optimized for PR-AUC)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    ax1.set_title("BFS: CRACK F1 and PR-AUC vs Number of Features")
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, plots_dir, "bfs_dual_metric")


def plot_precision_recall(df: pd.DataFrame, objective: str, plots_dir: Path):
    """Plot 8: Precision and Recall vs n_features."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_feat, df_sorted["crack_precision"], 'b-', linewidth=2, marker='o', markersize=4, label='Precision')
    ax.plot(n_feat, df_sorted["crack_recall"], 'r-', linewidth=2, marker='s', markersize=4, label='Recall')
    ax.plot(n_feat, df_sorted["crack_f1"], 'g--', linewidth=1.5, marker='^', markersize=3, label='F1', alpha=0.7)
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score")
    ax.set_title(f"BFS ({objective.upper()}): CRACK Precision, Recall, F1 vs Number of Features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_precision_recall")


def plot_additional_metrics(df: pd.DataFrame, objective: str, plots_dir: Path):
    """Plot 9: Additional metrics (accuracy, macro_f1, roc_auc)."""
    df_sorted = df.sort_values("n_features")
    n_feat = df_sorted["n_features"].values
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    ax = axes[0]
    ax.plot(n_feat, df_sorted["accuracy"], 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs n_features")
    ax.grid(True, alpha=0.3)
    
    # Macro F1
    ax = axes[1]
    ax.plot(n_feat, df_sorted["macro_f1"], 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 vs n_features")
    ax.grid(True, alpha=0.3)
    
    # ROC-AUC
    ax = axes[2]
    if "roc_auc" in df_sorted.columns:
        ax.plot(n_feat, df_sorted["roc_auc"], 'r-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC vs n_features")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"BFS ({objective.upper()}): Additional Metrics", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_{objective}_additional_metrics")


def plot_pareto(df_f1: pd.DataFrame, df_prauc: pd.DataFrame, 
                thresh_f1: Dict, thresh_prauc: Dict, plots_dir: Path):
    """Plot 10: Pareto scatter (PR-AUC vs F1)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # F1-optimized run
    ax.scatter(df_f1["crack_f1"], df_f1["crack_pr_auc"], c='blue', s=30, alpha=0.5, label='F1-optimized BFS')
    
    # PR-AUC-optimized run
    ax.scatter(df_prauc["crack_f1"], df_prauc["crack_pr_auc"], c='red', s=30, alpha=0.5, label='PR-AUC-optimized BFS')
    
    # Mark key points for F1 run
    n_f1_max = thresh_f1["n_at_max"]
    row_f1_max = df_f1[df_f1["n_features"] == n_f1_max].iloc[0]
    ax.scatter([row_f1_max["crack_f1"]], [row_f1_max["crack_pr_auc"]], 
               c='blue', s=200, marker='*', edgecolors='black', linewidths=1, zorder=10)
    ax.annotate(f'F1-best\nn={n_f1_max}', xy=(row_f1_max["crack_f1"], row_f1_max["crack_pr_auc"]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Mark key points for PR-AUC run
    n_prauc_max = thresh_prauc["n_at_max"]
    row_prauc_max = df_prauc[df_prauc["n_features"] == n_prauc_max].iloc[0]
    ax.scatter([row_prauc_max["crack_f1"]], [row_prauc_max["crack_pr_auc"]], 
               c='red', s=200, marker='*', edgecolors='black', linewidths=1, zorder=10)
    ax.annotate(f'PR-AUC-best\nn={n_prauc_max}', xy=(row_prauc_max["crack_f1"], row_prauc_max["crack_pr_auc"]),
                xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel("CRACK F1")
    ax.set_ylabel("CRACK PR-AUC")
    ax.set_title("BFS: Pareto Plot - CRACK PR-AUC vs F1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_plot(fig, plots_dir, "bfs_pareto")


def plot_both_objectives_same_metric(df_f1: pd.DataFrame, df_prauc: pd.DataFrame,
                                      metric_col: str, plots_dir: Path):
    """Plot same metric from both BFS runs for comparison."""
    df_f1_sorted = df_f1.sort_values("n_features")
    df_prauc_sorted = df_prauc.sort_values("n_features")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_f1_sorted["n_features"], df_f1_sorted[metric_col], 
            'b-', linewidth=2, marker='o', markersize=4, label='F1-optimized BFS')
    ax.plot(df_prauc_sorted["n_features"], df_prauc_sorted[metric_col],
            'r-', linewidth=2, marker='s', markersize=4, label='PR-AUC-optimized BFS')
    
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(f"BFS Comparison: {metric_col} from Both Objectives")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    save_plot(fig, plots_dir, f"bfs_comparison_{metric_col}")


# ==================== DEFAULT CONFIG (for VS Code Play button) ====================
# Modify this path to point to your experiment directory
DEFAULT_EXPERIMENT_DIR = r"C:\Users\yovel\Desktop\Grape_Project\experiments\feature_selection\bfs_full_run\2026-02-01_20-53-40"
# ==================================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate BFS plots from CSV logs")
    parser.add_argument("--experiment_dir", type=str, default=DEFAULT_EXPERIMENT_DIR, help="Path to experiment directory")
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"[ERROR] Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("BFS PLOT GENERATION")
    print("=" * 70)
    print(f"Experiment: {experiment_dir}")
    
    # Load CSV files
    log_f1_path = experiment_dir / "bfs_log_f1.csv"
    log_prauc_path = experiment_dir / "bfs_log_prauc.csv"
    
    if not log_f1_path.exists():
        print(f"[ERROR] F1 log not found: {log_f1_path}")
        sys.exit(1)
    if not log_prauc_path.exists():
        print(f"[ERROR] PR-AUC log not found: {log_prauc_path}")
        sys.exit(1)
    
    df_f1 = pd.read_csv(log_f1_path)
    df_prauc = pd.read_csv(log_prauc_path)
    
    print(f"[LOAD] F1 log: {len(df_f1)} rows")
    print(f"[LOAD] PR-AUC log: {len(df_prauc)} rows")
    
    # Create plots directory
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Compute thresholds
    print("\n[THRESHOLD] Computing threshold summaries...")
    thresh_f1 = compute_thresholds(df_f1, "f1", "crack_f1")
    thresh_prauc = compute_thresholds(df_prauc, "prauc", "crack_pr_auc")
    
    # Save threshold summaries
    thresh_f1_df = pd.DataFrame([thresh_f1])
    thresh_f1_df.to_csv(experiment_dir / "thresholds_summary_f1.csv", index=False)
    print(f"[SAVE] {experiment_dir / 'thresholds_summary_f1.csv'}")
    
    thresh_prauc_df = pd.DataFrame([thresh_prauc])
    thresh_prauc_df.to_csv(experiment_dir / "thresholds_summary_prauc.csv", index=False)
    print(f"[SAVE] {experiment_dir / 'thresholds_summary_prauc.csv'}")
    
    print(f"\n[THRESHOLD] F1 Summary:")
    print(f"  Max: {thresh_f1['max_score']:.4f} @ n={thresh_f1['n_at_max']}")
    print(f"  0.5% drop @ n={thresh_f1['n_at_0p5_drop']}: {thresh_f1['score_at_0p5_drop']:.4f}")
    print(f"  1.0% drop @ n={thresh_f1['n_at_1p0_drop']}: {thresh_f1['score_at_1p0_drop']:.4f}")
    print(f"  Recommended: n={thresh_f1['recommended_n_features']}")
    
    print(f"\n[THRESHOLD] PR-AUC Summary:")
    print(f"  Max: {thresh_prauc['max_score']:.4f} @ n={thresh_prauc['n_at_max']}")
    print(f"  0.5% drop @ n={thresh_prauc['n_at_0p5_drop']}: {thresh_prauc['score_at_0p5_drop']:.4f}")
    print(f"  1.0% drop @ n={thresh_prauc['n_at_1p0_drop']}: {thresh_prauc['score_at_1p0_drop']:.4f}")
    print(f"  Recommended: n={thresh_prauc['recommended_n_features']}")
    
    # Generate all plots
    print("\n[PLOT] Generating plots...")
    
    # F1 objective plots
    print("  F1 objective plots...")
    plot_metric_vs_n_line(df_f1, "crack_f1", "f1", thresh_f1, plots_dir)
    plot_metric_vs_n_scatter(df_f1, "crack_f1", "f1", thresh_f1, plots_dir)
    plot_normalized_metric(df_f1, "crack_f1", "f1", thresh_f1, plots_dir)
    plot_score_drop_percent(df_f1, "crack_f1", "f1", thresh_f1, plots_dir)
    plot_delta_metric(df_f1, "crack_f1", "f1", plots_dir)
    plot_threshold_summary(df_f1, "crack_f1", "f1", thresh_f1, plots_dir)
    plot_precision_recall(df_f1, "f1", plots_dir)
    plot_additional_metrics(df_f1, "f1", plots_dir)
    
    # PR-AUC objective plots
    print("  PR-AUC objective plots...")
    plot_metric_vs_n_line(df_prauc, "crack_pr_auc", "prauc", thresh_prauc, plots_dir)
    plot_metric_vs_n_scatter(df_prauc, "crack_pr_auc", "prauc", thresh_prauc, plots_dir)
    plot_normalized_metric(df_prauc, "crack_pr_auc", "prauc", thresh_prauc, plots_dir)
    plot_score_drop_percent(df_prauc, "crack_pr_auc", "prauc", thresh_prauc, plots_dir)
    plot_delta_metric(df_prauc, "crack_pr_auc", "prauc", plots_dir)
    plot_threshold_summary(df_prauc, "crack_pr_auc", "prauc", thresh_prauc, plots_dir)
    plot_precision_recall(df_prauc, "prauc", plots_dir)
    plot_additional_metrics(df_prauc, "prauc", plots_dir)
    
    # Comparison plots
    print("  Comparison plots...")
    plot_dual_metric(df_f1, df_prauc, plots_dir)
    plot_pareto(df_f1, df_prauc, thresh_f1, thresh_prauc, plots_dir)
    plot_both_objectives_same_metric(df_f1, df_prauc, "crack_f1", plots_dir)
    plot_both_objectives_same_metric(df_f1, df_prauc, "crack_pr_auc", plots_dir)
    plot_both_objectives_same_metric(df_f1, df_prauc, "accuracy", plots_dir)
    
    # Count plots
    n_plots = len(list(plots_dir.glob("*.png")))
    print(f"\n[DONE] Generated {n_plots} plots (PDF + PNG + SVG each)")
    print(f"[DONE] Output: {plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
