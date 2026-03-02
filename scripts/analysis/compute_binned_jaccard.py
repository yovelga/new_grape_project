"""
compute_binned_jaccard.py

Compute pairwise Jaccard similarity between BFS wavelength subsets
(5 seeds, varying split) using both exact wavelengths and binned wavelengths.

Binning rationale: neighbouring wavelengths are highly correlated in
hyperspectral data, so if seed A picks 548.55 nm and seed B picks 550.32 nm
they effectively agree.  Binning into 5 / 10 / 20 nm windows before computing
Jaccard gives a fairer stability measure.

Outputs (written to <stability_root>/varyingsplit/_analysis_binned_jaccard/):
  - exact_jaccard_top{K}.csv / .tex          -- exact pairwise Jaccard
  - binned_jaccard_top{K}_bin{B}nm.csv / .tex -- binned pairwise Jaccard
  - summary_binned_jaccard.csv / .tex        -- mean Jaccard per (top_k, bin)
  - Console summary with all numbers

Author : Stability Analysis Pipeline
Date   : March 2026
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Configuration ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

STABILITY_ROOT = (
    _PROJECT_ROOT
    / "experiments"
    / "feature_selection"
    / "stability_bfs_prauc"
    / "varyingsplit"
)

OUTPUT_DIR = STABILITY_ROOT / "_analysis_binned_jaccard"

TOP_K_VALUES = [30, 11]       # subsets to analyse
BIN_SIZES_NM = [5, 10, 20]   # bin widths in nm
WL_MIN = 450.0                # spectral range start (nm)
WL_MAX = 925.0                # spectral range end (nm)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def discover_runs(root: Path) -> List[dict]:
    """Find all prauc_varying_split* run directories."""
    runs = []
    for run_dir in sorted(root.glob("prauc_varying_*")):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            m = json.load(f)
        runs.append({
            "run_dir": run_dir,
            "seed": m.get("split_seed", m.get("seed")),
            "model_seed": m.get("model_seed"),
        })
    runs.sort(key=lambda r: r["seed"])
    return runs


def load_wavelengths(run: dict, top_k: int) -> List[str]:
    """Load selected_wavelengths_at{top_k}.csv and return ordered list."""
    path = run["run_dir"] / f"selected_wavelengths_at{top_k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    return df["wavelength"].tolist()


def parse_nm(wl_str: str) -> float:
    """'548.55nm' → 548.55"""
    return float(str(wl_str).strip().lower().replace("nm", ""))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairwise_jaccard_matrix(
    sets_by_seed: Dict[int, set],
) -> Tuple[pd.DataFrame, float, float]:
    """Return (DataFrame, mean_off_diag, std_off_diag)."""
    seeds = sorted(sets_by_seed)
    n = len(seeds)
    mat = np.zeros((n, n))
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            mat[i, j] = jaccard(sets_by_seed[si], sets_by_seed[sj])
    labels = [f"Seed {s}" for s in seeds]
    df = pd.DataFrame(mat, index=labels, columns=labels)
    mask = ~np.eye(n, dtype=bool)
    return df, mat[mask].mean(), mat[mask].std()


def wavelengths_to_bins(wl_list: List[str], bin_size: float) -> set:
    """Map a list of wavelength strings to a set of bin labels."""
    bins = set()
    for wl in wl_list:
        nm = parse_nm(wl)
        bin_idx = int((nm - WL_MIN) // bin_size)
        lo = WL_MIN + bin_idx * bin_size
        hi = lo + bin_size
        bins.add(f"{int(lo)}-{int(hi)}")
    return bins


# ─── LaTeX formatter ──────────────────────────────────────────────────────────

def jaccard_df_to_latex(
    df: pd.DataFrame, caption: str, label: str,
) -> str:
    """5×5 Jaccard matrix → LaTeX table with '---' on diagonal."""
    n = len(df)
    seeds = df.columns.tolist()
    col_spec = "c" * (n + 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & " + " & ".join([f"{s}" for s in seeds]) + r" \\",
        r"\midrule",
    ]
    for i, row_label in enumerate(seeds):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("---")
            else:
                cells.append(f"{df.iloc[i, j]:.3f}")
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summary_df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Summary table → LaTeX."""
    n_cols = len(df.columns)
    col_fmt = "l" + "r" * (n_cols - 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BINNED JACCARD ANALYSIS  –  BFS 5 seeds (varying split)")
    print("=" * 70)
    print(f"Source : {STABILITY_ROOT}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Top-K  : {TOP_K_VALUES}")
    print(f"Bins   : {BIN_SIZES_NM} nm")
    print("=" * 70)

    # 1) Discover runs
    runs = discover_runs(STABILITY_ROOT)
    print(f"\nFound {len(runs)} runs:")
    for r in runs:
        print(f"  seed={r['seed']}  dir={r['run_dir'].name}")

    if len(runs) == 0:
        print("ERROR: no runs found."); return

    # Summary collector
    summary_rows: list = []

    for top_k in TOP_K_VALUES:
        print(f"\n{'─' * 70}")
        print(f"  TOP-{top_k} WAVELENGTHS")
        print(f"{'─' * 70}")

        # Load wavelength lists per seed
        wl_lists: Dict[int, List[str]] = {}
        for r in runs:
            seed = r["seed"]
            wl_lists[seed] = load_wavelengths(r, top_k)
            print(f"  Seed {seed}: {len(wl_lists[seed])} wavelengths loaded")

        # ── Exact Jaccard ──────────────────────────────────────────────────
        exact_sets = {s: set(wl) for s, wl in wl_lists.items()}
        exact_df, exact_mean, exact_std = pairwise_jaccard_matrix(exact_sets)

        stem_exact = f"exact_jaccard_top{top_k}"
        exact_df.to_csv(OUTPUT_DIR / f"{stem_exact}.csv")
        tex = jaccard_df_to_latex(
            exact_df,
            caption=f"Pairwise Jaccard similarity of top-{top_k} wavelengths "
                    f"across seeds (exact match). Mean Jaccard = {exact_mean:.3f}.",
            label=f"tab:exact_jaccard_{top_k}",
        )
        (OUTPUT_DIR / f"{stem_exact}.tex").write_text(tex, encoding="utf-8")

        print(f"\n  Exact Jaccard (top-{top_k}): mean={exact_mean:.4f} ± {exact_std:.4f}")
        print(exact_df.to_string(float_format="%.3f"))

        summary_rows.append({
            "top_k": top_k,
            "bin_nm": "exact",
            "mean_jaccard": f"{exact_mean:.4f}",
            "std_jaccard": f"{exact_std:.4f}",
        })

        # ── Binned Jaccard for each bin size ──────────────────────────────
        for bin_size in BIN_SIZES_NM:
            binned_sets = {s: wavelengths_to_bins(wl, bin_size)
                          for s, wl in wl_lists.items()}
            bjac_df, bjac_mean, bjac_std = pairwise_jaccard_matrix(binned_sets)

            stem_bin = f"binned_jaccard_top{top_k}_bin{bin_size}nm"
            bjac_df.to_csv(OUTPUT_DIR / f"{stem_bin}.csv")
            tex = jaccard_df_to_latex(
                bjac_df,
                caption=(
                    f"Pairwise Jaccard similarity of top-{top_k} wavelengths "
                    f"across seeds ({bin_size}\\,nm bins). "
                    f"Mean Jaccard = {bjac_mean:.3f}."
                ),
                label=f"tab:binned_jaccard_{top_k}_bin{bin_size}",
            )
            (OUTPUT_DIR / f"{stem_bin}.tex").write_text(tex, encoding="utf-8")

            # Also print the binned sets for insight
            print(f"\n  Binned Jaccard (top-{top_k}, bin={bin_size} nm): "
                  f"mean={bjac_mean:.4f} ± {bjac_std:.4f}")
            print(bjac_df.to_string(float_format="%.3f"))

            # Show number of unique bins per seed
            for s in sorted(binned_sets):
                print(f"    Seed {s}: {len(binned_sets[s])} unique bins")

            summary_rows.append({
                "top_k": top_k,
                "bin_nm": f"{bin_size}",
                "mean_jaccard": f"{bjac_mean:.4f}",
                "std_jaccard": f"{bjac_std:.4f}",
            })

    # ── Summary table ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = ["Top-K", "Bin (nm)", "Mean Jaccard", "Std Jaccard"]
    summary_df.to_csv(OUTPUT_DIR / "summary_binned_jaccard.csv", index=False)

    tex_summary = summary_df_to_latex(
        summary_df,
        caption=(
            "Mean pairwise Jaccard similarity across 5 BFS seeds for exact and "
            "binned wavelength matching (5, 10, 20\\,nm bins)."
        ),
        label="tab:binned_jaccard_summary",
    )
    (OUTPUT_DIR / "summary_binned_jaccard.tex").write_text(tex_summary, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs → {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
