"""
analyze_bfs_prauc_stability.py

Post-hoc analysis of the BFS PR-AUC stability experiment (5 seeds).

This script is deterministic and re-runnable.  It reads the output folders
produced by ``run_bfs_prauc_stability_5runs.py``, computes stability metrics,
and generates thesis-ready plots and summary tables.

Outputs (all written to  <STABILITY_RUNS_ROOT>/_analysis/):
============================================================

**Plots  (PNG, 300 DPI)**

| File name                                           | Description |
|-----------------------------------------------------|-------------|
| bfs_prauc_stability_mean_curve.png                  | Mean PR-AUC ± std vs n_features (shaded band) |
| bfs_prauc_stability_per_seed_curves.png             | Individual seed curves overlaid |
| bfs_prauc_stability_best_score_per_seed.png         | Best PR-AUC per seed (bar) |
| bfs_prauc_stability_jaccard_heatmap_top30.png       | 5×5 Jaccard similarity heatmap |
| bfs_prauc_stability_overlap_heatmap_top30.png       | 5×5 overlap-count heatmap |
| bfs_prauc_stability_top20_wavelength_frequency.png  | Top-20 wavelengths by selection frequency |
| bfs_prauc_stability_frequency_spectrum_top30.png    | Frequency vs sorted wavelength (step) |
| bfs_prauc_stability_keypoint_boxplots.png           | Boxplots of PR-AUC at key n_features |
| bfs_prauc_stability_best_n_histogram.png            | Histogram of best_n_features |
| bfs_prauc_stability_rank_distribution_top15.png     | Rank distribution of top-15 wavelengths |
| bfs_prauc_stability_score_drop_mean.png             | Mean score-drop (%) curve with std band |
| bfs_prauc_binned_jaccard_top30_bin{BIN}nm.png        | Binned Jaccard heatmap per bin size |
| bfs_prauc_binned_frequency_top30_bin{BIN}nm.png      | Binned frequency spectrum per bin size |
| bfs_prauc_binned_frequency_comparison_top30_5_10_15nm.png | Comparison of binned frequency spectra |

**Tables  (CSV  +  LaTeX .tex)**

| File name                                         | Description |
|---------------------------------------------------|-------------|
| table_runs_summary.*                              | Seed, best_score, best_n_features, timestamp |
| table_stability_summary.*                         | Mean/std Jaccard, overlap |
| table_top30_wavelengths_by_frequency.*            | Wavelength, frequency, mean_rank |
| table_keypoint_prauc.*                            | Mean±std PR-AUC at n={9,11,30,159} |
| binned_jaccard_matrix_top30_bin{BIN}nm.*           | Binned Jaccard similarity matrix |
| binned_frequency_top30_bin{BIN}nm.*                | Bin-level selection frequency |

Author : Stability Analysis Pipeline
Date   : February 2026
"""

import sys
import json
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Matplotlib (Agg backend for headless plotting)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# Thesis-ready defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ====================== CONFIGURATION (edit here) =========================
_PROJECT_ROOT = Path(__file__).resolve().parents[5]

CONFIG = {
    # Root folder that contains prauc_seed_*/ sub-folders
    "stability_runs_root": str(
        _PROJECT_ROOT / "experiments" / "feature_selection" / "stability_bfs_prauc"
    ),

    # Analysis settings
    "top_k": 30,
    "metric_col": "crack_pr_auc",

    # Key n_features values to highlight in tables/boxplots
    "n_features_grid": [9, 11, 30, 159],

    # Output folder (created inside stability_runs_root)
    "output_subdir": "_analysis",

    # ---- Binned stability analysis ----
    "wl_min": 450,
    "wl_max": 925,
    "bin_sizes_nm": [5, 10, 20],
}
# ===========================================================================


# ==================== DISCOVERY ====================

def discover_runs(root: Path) -> List[dict]:
    """Find all sub-folders that contain a run_manifest.json."""
    runs = []
    for manifest_path in sorted(root.glob("prauc_seed_*/run_manifest.json")):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        run_dir = manifest_path.parent
        runs.append({
            "run_dir": run_dir,
            "seed": manifest["seed"],
            "timestamp": manifest["timestamp"],
            "best_score": manifest["best_score"],
            "best_n_features": manifest["best_n_features"],
            "top_k": manifest.get("top_k_for_stability", CONFIG["top_k"]),
        })
    if not runs:
        raise FileNotFoundError(
            f"No run_manifest.json found under {root}/prauc_seed_*/"
        )
    runs.sort(key=lambda r: r["seed"])
    return runs


def load_run_data(run: dict) -> Tuple[pd.DataFrame, dict]:
    """Load bfs_log_prauc.csv and best_features_prauc.json for one run."""
    log_path = run["run_dir"] / "bfs_log_prauc.csv"
    best_path = run["run_dir"] / "best_features_prauc.json"
    log_df = pd.read_csv(log_path)
    with open(best_path, "r") as f:
        best_info = json.load(f)
    return log_df, best_info


# ==================== STABILITY METRICS ====================

def jaccard(a: set, b: set) -> float:
    """Jaccard index between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compute_pairwise_matrices(
    topk_sets: Dict[int, set],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Jaccard and overlap-count matrices.

    Parameters
    ----------
    topk_sets : dict  seed -> set of wavelength strings

    Returns
    -------
    jaccard_df, overlap_df : DataFrames indexed & columned by seed
    """
    seeds = sorted(topk_sets.keys())
    n = len(seeds)
    jac_mat = np.zeros((n, n))
    ovl_mat = np.zeros((n, n))
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            jac_mat[i, j] = jaccard(topk_sets[si], topk_sets[sj])
            ovl_mat[i, j] = len(topk_sets[si] & topk_sets[sj])
    labels = [f"seed_{s}" for s in seeds]
    return (
        pd.DataFrame(jac_mat, index=labels, columns=labels),
        pd.DataFrame(ovl_mat, index=labels, columns=labels),
    )


def compute_frequency_table(
    topk_sets: Dict[int, set],
    top_k: int,
) -> pd.DataFrame:
    """Build a table: wavelength | frequency | mean_rank.

    ``mean_rank`` is the average position (1-based) among runs that selected
    this wavelength in their top-K list.
    """
    # frequency
    all_wl = set()
    for s in topk_sets.values():
        all_wl |= s
    freq = {wl: 0 for wl in all_wl}
    for s in topk_sets.values():
        for wl in s:
            freq[wl] += 1

    # mean rank (requires ordered lists, not sets – we'll use _topk_lists)
    return freq  # caller will build full table using ordered lists


def compute_frequency_and_rank(
    topk_lists: Dict[int, list],
) -> pd.DataFrame:
    """Wavelength frequency and mean rank across runs.

    Parameters
    ----------
    topk_lists : dict  seed -> ordered list of wavelength strings (best first)
    """
    all_wl: set = set()
    for lst in topk_lists.values():
        all_wl.update(lst)

    records = []
    # str() cast: wavelengths may arrive as numeric types from JSON
    for wl in sorted(all_wl, key=lambda w: float(str(w).replace("nm", "").strip())):
        freq = 0
        ranks = []
        for seed, lst in topk_lists.items():
            if wl in lst:
                freq += 1
                ranks.append(lst.index(wl) + 1)  # 1-based rank
        mean_rank = float(np.mean(ranks)) if ranks else np.nan
        records.append({"wavelength": wl, "frequency": freq, "mean_rank": round(mean_rank, 2)})

    df = pd.DataFrame(records)
    df.sort_values(["frequency", "mean_rank"], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ==================== PLOT HELPERS ====================

def _save(fig, out_dir: Path, stem: str):
    """Save figure as PNG (and optionally PDF/SVG) then close."""
    fig.savefig(out_dir / f"{stem}.png", format="png")
    # Optionally also save vector formats for the thesis
    fig.savefig(out_dir / f"{stem}.pdf", format="pdf")
    plt.close(fig)


# ==================== BINNED STABILITY HELPERS ====================

def parse_wavelength_nm(x) -> float:
    """Convert a wavelength value to float nm.

    Handles formats like ``"723nm"``, ``"723.0"``, ``723``, ``723.0``.
    """
    s = str(x).strip().lower().replace("nm", "")
    return float(s)


def wavelength_to_bin_label(wl_nm: float, wl_min: float, bin_size: int) -> str:
    """Map a wavelength (nm) to its bin label, e.g. ``"450-455"``."""
    bin_idx = int((wl_nm - wl_min) // bin_size)
    lo = wl_min + bin_idx * bin_size
    hi = lo + bin_size
    return f"{int(lo)}-{int(hi)}"


def topk_list_to_bins(
    topk_list: list,
    wl_min: float,
    bin_size: int,
) -> set:
    """Convert a list of wavelength identifiers to a set of bin labels."""
    bins = set()
    for wl in topk_list:
        nm = parse_wavelength_nm(wl)
        bins.add(wavelength_to_bin_label(nm, wl_min, bin_size))
    return bins


def compute_binned_jaccard_matrix(
    topk_lists: Dict[int, list],
    wl_min: float,
    bin_size: int,
) -> Tuple[pd.DataFrame, Dict[int, set]]:
    """Compute pairwise Jaccard on binned wavelength sets.

    Returns (jaccard_df, binned_sets_per_seed).
    """
    binned: Dict[int, set] = {}
    for seed, lst in topk_lists.items():
        binned[seed] = topk_list_to_bins(lst, wl_min, bin_size)

    seeds = sorted(binned.keys())
    n = len(seeds)
    mat = np.zeros((n, n))
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            mat[i, j] = jaccard(binned[si], binned[sj])
    labels = [f"seed_{s}" for s in seeds]
    return pd.DataFrame(mat, index=labels, columns=labels), binned


def compute_binned_frequency(
    binned_sets: Dict[int, set],
    wl_min: float,
    wl_max: float,
    bin_size: int,
) -> pd.DataFrame:
    """Frequency of each bin across runs.

    Returns a DataFrame with columns: bin_label, bin_lo, frequency.
    All possible bins in [wl_min, wl_max) are included (frequency=0 if absent).
    """
    # Build all possible bins
    all_bins = []
    lo = wl_min
    while lo < wl_max:
        hi = lo + bin_size
        lbl = f"{int(lo)}-{int(hi)}"
        all_bins.append((lo, lbl))
        lo = hi

    freq_map = {lbl: 0 for _, lbl in all_bins}
    for bset in binned_sets.values():
        for b in bset:
            if b in freq_map:
                freq_map[b] += 1

    records = [{"bin_label": lbl, "bin_lo": lo, "frequency": freq_map[lbl]}
               for lo, lbl in all_bins]
    return pd.DataFrame(records)


# ==================== BINNED PLOTS ====================

def plot_binned_jaccard_heatmap(
    jac_df: pd.DataFrame,
    bin_size: int,
    top_k: int,
    out_dir: Path,
):
    """Heatmap of binned Jaccard similarity."""
    plot_heatmap(
        jac_df,
        title=f"Binned Jaccard Similarity (Top-{top_k}, bin={bin_size} nm)",
        cmap="YlGnBu",
        fmt="%.3f",
        stem=f"bfs_prauc_binned_jaccard_top30_bin{bin_size}nm",
        out_dir=out_dir,
        vmin=0, vmax=1,
    )


def plot_binned_frequency_spectrum(
    freq_df: pd.DataFrame,
    bin_size: int,
    top_k: int,
    n_seeds: int,
    out_dir: Path,
):
    """Bar plot of bin-level selection frequency."""
    df = freq_df.sort_values("bin_lo")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(df)), df["frequency"].values,
           edgecolor="black", linewidth=0.4, alpha=0.85, color="tab:cyan")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["bin_label"].values, rotation=70, ha="right", fontsize=7)
    ax.set_xlabel(f"Wavelength Bin ({bin_size} nm)")
    ax.set_ylabel(f"Selection Frequency (0–{n_seeds})")
    ax.set_title(f"Binned Wavelength Frequency (Top-{top_k}, bin={bin_size} nm)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, out_dir, f"bfs_prauc_binned_frequency_top30_bin{bin_size}nm")


def plot_binned_frequency_comparison(
    freq_dfs: Dict[int, pd.DataFrame],
    top_k: int,
    n_seeds: int,
    out_dir: Path,
):
    """Overlay binned frequency spectra for all bin sizes on one axis."""
    colors = {5: "tab:blue", 10: "tab:orange", 15: "tab:green"}
    fig, ax = plt.subplots(figsize=(13, 5))
    for bin_size in sorted(freq_dfs.keys()):
        df = freq_dfs[bin_size].sort_values("bin_lo")
        mid = df["bin_lo"].values + bin_size / 2.0
        c = colors.get(bin_size, "gray")
        ax.step(mid, df["frequency"].values, where="mid",
                linewidth=1.8, label=f"bin={bin_size} nm", color=c)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Selection Frequency (0–{n_seeds})")
    ax.set_title(f"Binned Frequency Comparison (Top-{top_k}, bins=5/10/15 nm)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_binned_frequency_comparison_top30_5_10_15nm")


# ==================== PLOTS ====================

def plot_mean_curve(
    all_logs: Dict[int, pd.DataFrame],
    metric_col: str,
    out_dir: Path,
):
    """Plot A: Mean PR-AUC vs n_features with shaded std band."""
    # Align on n_features (use union of all n_features)
    n_union = sorted(
        set().union(*(set(df["n_features"].values) for df in all_logs.values()))
    )
    matrix = np.full((len(all_logs), len(n_union)), np.nan)
    seeds_sorted = sorted(all_logs.keys())
    for i, seed in enumerate(seeds_sorted):
        df = all_logs[seed].set_index("n_features")
        for j, n in enumerate(n_union):
            if n in df.index:
                matrix[i, j] = df.loc[n, metric_col]

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    n_arr = np.array(n_union)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(n_arr, mean, "b-", linewidth=2, label="Mean PR-AUC")
    ax.fill_between(n_arr, mean - std, mean + std, alpha=0.25, color="blue", label="±1 std")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("CRACK PR-AUC")
    ax.set_title("BFS PR-AUC Stability (5 seeds): Mean ± Std vs Number of Features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_mean_curve")


def plot_per_seed_curves(
    all_logs: Dict[int, pd.DataFrame],
    metric_col: str,
    out_dir: Path,
):
    """Plot: Individual seed curves overlaid for visual comparison."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for seed in sorted(all_logs.keys()):
        df = all_logs[seed].sort_values("n_features")
        ax.plot(df["n_features"], df[metric_col], linewidth=1.5,
                marker="o", markersize=2, label=f"Seed {seed}")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("CRACK PR-AUC")
    ax.set_title("BFS PR-AUC: Per-Seed Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_per_seed_curves")


def plot_best_score_per_seed(
    runs: List[dict],
    out_dir: Path,
):
    """Plot B: Best PR-AUC per seed (bar chart) with best_n_features annotated."""
    seeds = [r["seed"] for r in runs]
    scores = [r["best_score"] for r in runs]
    n_feats = [r["best_n_features"] for r in runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(seeds)), scores, tick_label=[f"Seed {s}" for s in seeds],
                  edgecolor="black", linewidth=0.7)
    for i, (bar, nf) in enumerate(zip(bars, n_feats)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"n={nf}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Best CRACK PR-AUC")
    ax.set_title("BFS PR-AUC: Best Score per Seed")
    ax.grid(True, alpha=0.3, axis="y")
    ymin = min(scores) - 0.02
    ymax = max(scores) + 0.02
    ax.set_ylim(max(0, ymin), min(1.0, ymax))
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_best_score_per_seed")


def plot_heatmap(
    mat_df: pd.DataFrame,
    title: str,
    cmap: str,
    fmt: str,
    stem: str,
    out_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Generic heatmap using matplotlib imshow (no seaborn)."""
    data = mat_df.values
    labels = mat_df.columns.tolist()
    n = len(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            color = "white" if val < (data.max() + data.min()) / 2 else "black"
            ax.text(j, i, fmt % val, ha="center", va="center",
                    fontsize=10, color=color)

    ax.set_title(title)
    plt.tight_layout()
    _save(fig, out_dir, stem)


def plot_top20_wavelength_frequency(
    freq_df: pd.DataFrame,
    top_k: int,
    out_dir: Path,
    n_seeds: int = 5,
):
    """Plot E: Bar plot of top-20 wavelengths by selection frequency."""
    df_top = freq_df.head(20).copy()
    df_top = df_top.sort_values("frequency", ascending=True)  # horizontal bar

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(df_top)), df_top["frequency"].values,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["wavelength"].values, fontsize=9)
    ax.set_xlabel(f"Selection Frequency (out of {n_seeds} seeds)")
    ax.set_ylabel("Wavelength")
    ax.set_title(f"Top-20 Wavelengths by Selection Frequency (Top-{top_k} per run)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_top20_wavelength_frequency")


def plot_frequency_spectrum(
    freq_df: pd.DataFrame,
    top_k: int,
    out_dir: Path,
):
    """Plot F: Selection frequency vs sorted wavelength (step plot)."""
    # Sort by wavelength value numerically
    df = freq_df.copy()
    df["_wl_num"] = df["wavelength"].apply(
        lambda w: float(str(w).replace("nm", "").strip())
    )
    df.sort_values("_wl_num", inplace=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(df["_wl_num"].values, df["frequency"].values, where="mid",
            linewidth=1.5, color="tab:blue")
    ax.fill_between(df["_wl_num"].values, df["frequency"].values,
                    step="mid", alpha=0.15, color="tab:blue")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Selection Frequency (0–5)")
    ax.set_title(f"Wavelength Selection Frequency Spectrum (Top-{top_k} per run)")
    ax.set_ylim(-0.2, 5.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_frequency_spectrum_top30")


def plot_keypoint_boxplots(
    all_logs: Dict[int, pd.DataFrame],
    metric_col: str,
    n_grid: List[int],
    out_dir: Path,
):
    """Plot G: Boxplots of PR-AUC at key n_features across runs."""
    data_per_n = {}
    for n_target in n_grid:
        vals = []
        for seed, df in all_logs.items():
            # closest existing n_features row
            idx = (df["n_features"] - n_target).abs().idxmin()
            vals.append(df.loc[idx, metric_col])
        data_per_n[n_target] = vals

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(n_grid))
    bp = ax.boxplot(
        [data_per_n[n] for n in n_grid],
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
    )
    # Style boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_edgecolor("black")

    # Overlay individual points
    for i, n in enumerate(n_grid):
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(data_per_n[n]))
        ax.scatter(np.full(len(data_per_n[n]), i) + jitter,
                   data_per_n[n], color="tab:blue", s=30, zorder=5, alpha=0.8)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"n={n}" for n in n_grid])
    ax.set_ylabel("CRACK PR-AUC")
    ax.set_title("PR-AUC at Key Feature Counts (across 5 seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_keypoint_boxplots")


def plot_best_n_histogram(
    runs: List[dict],
    out_dir: Path,
):
    """Plot H: Histogram of best_n_features across runs."""
    n_feats = [r["best_n_features"] for r in runs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(min(n_feats) - 5, max(n_feats) + 6, max(1, (max(n_feats) - min(n_feats)) // 8 + 1))
    ax.hist(n_feats, bins=bins, edgecolor="black", linewidth=0.7, alpha=0.8)
    for nf in n_feats:
        ax.axvline(x=nf, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Best n_features")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Optimal Feature Count (5 seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_best_n_histogram")


def plot_rank_distribution_top15(
    topk_lists: Dict[int, list],
    freq_df: pd.DataFrame,
    out_dir: Path,
):
    """Plot: Rank distribution (box) for top-15 most frequent wavelengths."""
    top15 = freq_df.head(15)["wavelength"].tolist()

    rank_data = {}
    for wl in top15:
        ranks = []
        for seed, lst in topk_lists.items():
            if wl in lst:
                ranks.append(lst.index(wl) + 1)
        rank_data[wl] = ranks

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(top15))
    bp = ax.boxplot(
        [rank_data[wl] for wl in top15],
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
        vert=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightyellow")
        patch.set_edgecolor("black")

    # Overlay points
    for i, wl in enumerate(top15):
        vals = rank_data[wl]
        jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color="tab:orange", s=25, zorder=5, alpha=0.9)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(top15, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Rank (1 = most important)")
    ax.set_title("Rank Distribution of Top-15 Most Frequent Wavelengths (Top-30 per run)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_rank_distribution_top15")


def plot_score_drop_mean(
    all_logs: Dict[int, pd.DataFrame],
    metric_col: str,
    out_dir: Path,
):
    """Plot: Mean score-drop (%) curve with std band."""
    n_union = sorted(
        set().union(*(set(df["n_features"].values) for df in all_logs.values()))
    )
    matrix = np.full((len(all_logs), len(n_union)), np.nan)
    seeds_sorted = sorted(all_logs.keys())
    for i, seed in enumerate(seeds_sorted):
        df = all_logs[seed].set_index("n_features")
        max_score = df[metric_col].max()
        for j, n in enumerate(n_union):
            if n in df.index:
                matrix[i, j] = (max_score - df.loc[n, metric_col]) / max_score * 100

    mean_drop = np.nanmean(matrix, axis=0)
    std_drop = np.nanstd(matrix, axis=0)
    n_arr = np.array(n_union)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(n_arr, mean_drop, "r-", linewidth=2, label="Mean drop (%)")
    ax.fill_between(n_arr, mean_drop - std_drop, mean_drop + std_drop,
                    alpha=0.2, color="red", label="±1 std")
    ax.axhline(y=0.5, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="0.5% drop")
    ax.axhline(y=1.0, color="darkred", linestyle="--", linewidth=1.5, alpha=0.7, label="1.0% drop")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score Drop from Maximum (%)")
    ax.set_title("BFS PR-AUC Stability: Mean Score Drop (%) ± Std")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.5, max(5, np.nanmax(mean_drop + std_drop) + 1))
    plt.tight_layout()
    _save(fig, out_dir, "bfs_prauc_stability_score_drop_mean")


# ==================== TABLES ====================

def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert a DataFrame to a simple LaTeX tabular snippet."""
    n_cols = len(df.columns)
    col_fmt = "l" + "r" * (n_cols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [str(v) for v in row.values]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    caption: str,
    label: str,
):
    """Save a table as CSV and LaTeX .tex."""
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    tex = _df_to_latex(df, caption, label)
    with open(out_dir / f"{stem}.tex", "w") as f:
        f.write(tex)


def build_table_runs_summary(runs: List[dict]) -> pd.DataFrame:
    """Table 1: per-run summary."""
    rows = []
    for r in runs:
        rows.append({
            "Seed": r["seed"],
            "Best PR-AUC": f'{r["best_score"]:.4f}',
            "Best n_features": r["best_n_features"],
            "Timestamp": r["timestamp"],
        })
    return pd.DataFrame(rows)


def build_table_stability_summary(
    jac_df: pd.DataFrame,
    ovl_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Table 2: mean/std of Jaccard and overlap (off-diagonal only)."""
    n = len(jac_df)
    mask = ~np.eye(n, dtype=bool)
    jac_vals = jac_df.values[mask]
    ovl_vals = ovl_df.values[mask]
    return pd.DataFrame([{
        "Metric": f"Jaccard@{top_k}",
        "Mean": f"{jac_vals.mean():.4f}",
        "Std": f"{jac_vals.std():.4f}",
        "Min": f"{jac_vals.min():.4f}",
        "Max": f"{jac_vals.max():.4f}",
    }, {
        "Metric": f"Overlap@{top_k}",
        "Mean": f"{ovl_vals.mean():.1f}",
        "Std": f"{ovl_vals.std():.1f}",
        "Min": f"{ovl_vals.min():.0f}",
        "Max": f"{ovl_vals.max():.0f}",
    }])


def build_table_keypoint_prauc(
    all_logs: Dict[int, pd.DataFrame],
    metric_col: str,
    n_grid: List[int],
) -> pd.DataFrame:
    """Table 4: PR-AUC at key n_features: mean±std and per-seed values."""
    seeds_sorted = sorted(all_logs.keys())
    rows = []
    for n_target in n_grid:
        per_seed = {}
        vals = []
        for seed in seeds_sorted:
            df = all_logs[seed]
            idx = (df["n_features"] - n_target).abs().idxmin()
            actual_n = int(df.loc[idx, "n_features"])
            v = float(df.loc[idx, metric_col])
            vals.append(v)
            per_seed[f"Seed_{seed}"] = f"{v:.4f}"
        row = {
            "n_features": n_target,
            "Mean": f"{np.mean(vals):.4f}",
            "Std": f"{np.std(vals):.4f}",
        }
        row.update(per_seed)
        rows.append(row)
    return pd.DataFrame(rows)


# ==================== MAIN ====================

def main():
    root = Path(CONFIG["stability_runs_root"])
    out_dir = root / CONFIG["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = CONFIG["top_k"]
    metric_col = CONFIG["metric_col"]
    n_grid = CONFIG["n_features_grid"]

    print("=" * 70)
    print("BFS PR-AUC STABILITY ANALYSIS")
    print("=" * 70)
    print(f"Runs root : {root}")
    print(f"Output    : {out_dir}")
    print(f"Top-K     : {top_k}")
    print(f"Metric    : {metric_col}")
    print(f"Key Ns    : {n_grid}")
    print("=" * 70)

    # 1) Discover runs
    runs = discover_runs(root)
    print(f"\n[INFO] Found {len(runs)} runs:")
    for r in runs:
        print(f"  seed={r['seed']}  best={r['best_score']:.4f}  "
              f"n_feat={r['best_n_features']}  dir={r['run_dir'].name}")

    # 2) Load data
    all_logs: Dict[int, pd.DataFrame] = {}
    topk_lists: Dict[int, list] = {}  # ordered lists (rank matters)
    topk_sets: Dict[int, set] = {}

    for r in runs:
        log_df, best_info = load_run_data(r)
        seed = r["seed"]
        all_logs[seed] = log_df

        selected = best_info["selected_features"][:top_k]
        # Ensure all wavelength identifiers are str (robust against numeric JSON values)
        selected = [str(w) for w in selected]
        topk_lists[seed] = selected
        topk_sets[seed] = set(selected)

    if topk_lists:
        first_seed = next(iter(topk_lists))
        first_wl = topk_lists[first_seed][0] if topk_lists[first_seed] else None
        print(f"[DEBUG] selected_features element type: {type(first_wl).__name__} "
              f"(example: {first_wl!r})")

    # 3) Stability metrics
    print("\n[STABILITY] Computing pairwise matrices...")
    jac_df, ovl_df = compute_pairwise_matrices(topk_sets)
    freq_df = compute_frequency_and_rank(topk_lists)

    mask = ~np.eye(len(jac_df), dtype=bool)
    jac_mean = jac_df.values[mask].mean()
    jac_std = jac_df.values[mask].std()
    ovl_mean = ovl_df.values[mask].mean()
    ovl_std = ovl_df.values[mask].std()
    print(f"  Jaccard@{top_k} : {jac_mean:.4f} ± {jac_std:.4f}")
    print(f"  Overlap@{top_k} : {ovl_mean:.1f} ± {ovl_std:.1f}")

    # 4) Plots
    print("\n[PLOTS] Generating thesis-ready plots...")

    plot_mean_curve(all_logs, metric_col, out_dir)
    print("  ✓ mean curve")

    plot_per_seed_curves(all_logs, metric_col, out_dir)
    print("  ✓ per-seed curves")

    plot_best_score_per_seed(runs, out_dir)
    print("  ✓ best score per seed")

    plot_heatmap(
        jac_df,
        title=f"Pairwise Jaccard Similarity (Top-{top_k})",
        cmap="YlGnBu",
        fmt="%.3f",
        stem="bfs_prauc_stability_jaccard_heatmap_top30",
        out_dir=out_dir,
        vmin=0, vmax=1,
    )
    print("  ✓ Jaccard heatmap")

    plot_heatmap(
        ovl_df,
        title=f"Pairwise Overlap Count (Top-{top_k})",
        cmap="YlOrRd",
        fmt="%.0f",
        stem="bfs_prauc_stability_overlap_heatmap_top30",
        out_dir=out_dir,
        vmin=0, vmax=top_k,
    )
    print("  ✓ overlap heatmap")

    plot_top20_wavelength_frequency(freq_df, top_k, out_dir, n_seeds=len(runs))
    print("  ✓ top-20 wavelength frequency")

    plot_frequency_spectrum(freq_df, top_k, out_dir)
    print("  ✓ frequency spectrum")

    plot_keypoint_boxplots(all_logs, metric_col, n_grid, out_dir)
    print("  ✓ keypoint boxplots")

    plot_best_n_histogram(runs, out_dir)
    print("  ✓ best_n histogram")

    plot_rank_distribution_top15(topk_lists, freq_df, out_dir)
    print("  ✓ rank distribution top-15")

    plot_score_drop_mean(all_logs, metric_col, out_dir)
    print("  ✓ score drop mean curve")

    # 5) Tables
    print("\n[TABLES] Generating summary tables (CSV + LaTeX)...")

    # Table 1 – runs summary
    t1 = build_table_runs_summary(runs)
    save_table(t1, out_dir, "table_runs_summary",
               caption="BFS PR-AUC stability: per-seed results summary.",
               label="tab:bfs_prauc_runs_summary")
    print("  ✓ table_runs_summary")

    # Table 2 – stability summary
    t2 = build_table_stability_summary(jac_df, ovl_df, top_k)
    save_table(t2, out_dir, "table_stability_summary",
               caption=f"BFS PR-AUC stability: pairwise Jaccard and overlap (Top-{top_k}).",
               label="tab:bfs_prauc_stability_summary")
    print("  ✓ table_stability_summary")

    # Table 3 – top-30 wavelengths by frequency
    t3 = freq_df.head(top_k).copy()
    t3.columns = ["Wavelength", "Frequency (0-5)", "Mean Rank"]
    save_table(t3, out_dir, "table_top30_wavelengths_by_frequency",
               caption=f"Top-{top_k} most frequently selected wavelengths across 5 BFS seeds.",
               label="tab:bfs_prauc_top30_wavelengths")
    print("  ✓ table_top30_wavelengths_by_frequency")

    # Table 4 – key-point PR-AUC
    t4 = build_table_keypoint_prauc(all_logs, metric_col, n_grid)
    save_table(t4, out_dir, "table_keypoint_prauc",
               caption=f"CRACK PR-AUC at key feature counts (n={{{','.join(map(str, n_grid))}}}) across 5 seeds.",
               label="tab:bfs_prauc_keypoints")
    print("  ✓ table_keypoint_prauc")

    # Save raw matrices as CSV for reference
    jac_df.to_csv(out_dir / "jaccard_matrix_top30.csv")
    ovl_df.to_csv(out_dir / "overlap_matrix_top30.csv")
    freq_df.to_csv(out_dir / "wavelength_frequency_top30.csv", index=False)

    # ================================================================
    # 6) BINNED STABILITY ANALYSIS
    # ================================================================
    wl_min = CONFIG["wl_min"]
    wl_max = CONFIG["wl_max"]
    bin_sizes = CONFIG["bin_sizes_nm"]
    n_seeds = len(runs)

    print(f"\n[BINNED] Running binned stability analysis (bins={bin_sizes} nm) ...")
    binned_freq_dfs: Dict[int, pd.DataFrame] = {}
    binned_jac_summaries: list = []

    for bin_size in bin_sizes:
        # Jaccard
        bjac_df, binned_sets = compute_binned_jaccard_matrix(
            topk_lists, wl_min, bin_size,
        )
        bjac_df.to_csv(out_dir / f"binned_jaccard_matrix_top30_bin{bin_size}nm.csv")
        plot_binned_jaccard_heatmap(bjac_df, bin_size, top_k, out_dir)
        print(f"  \u2713 binned Jaccard heatmap (bin={bin_size} nm)")

        # Frequency
        bfreq_df = compute_binned_frequency(binned_sets, wl_min, wl_max, bin_size)
        bfreq_df.to_csv(out_dir / f"binned_frequency_top30_bin{bin_size}nm.csv", index=False)
        plot_binned_frequency_spectrum(bfreq_df, bin_size, top_k, n_seeds, out_dir)
        print(f"  \u2713 binned frequency spectrum (bin={bin_size} nm)")

        binned_freq_dfs[bin_size] = bfreq_df

        # Summary stats
        mask_b = ~np.eye(len(bjac_df), dtype=bool)
        bj_vals = bjac_df.values[mask_b]
        binned_jac_summaries.append({
            "bin_size": bin_size,
            "mean": bj_vals.mean(),
            "std": bj_vals.std(),
        })

    # Comparison plot (all bin sizes)
    plot_binned_frequency_comparison(binned_freq_dfs, top_k, n_seeds, out_dir)
    print("  \u2713 binned frequency comparison (5/10/15 nm)")

    # Print binned Jaccard summary
    print(f"\n[BINNED] Binned Jaccard summary (Top-{top_k}):")
    for s in binned_jac_summaries:
        print(f"  bin={s['bin_size']:>2d} nm : {s['mean']:.4f} \u00b1 {s['std']:.4f}")

    n_plots = len(list(out_dir.glob("*.png")))
    n_tables = len(list(out_dir.glob("*.csv")))
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Plots  : {n_plots} PNG (+PDF)")
    print(f"  Tables : {n_tables} CSV (+LaTeX .tex)")
    print(f"  Output : {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
