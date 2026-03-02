"""
generate_binned_jaccard_plot.py

Generate a grouped bar chart showing how mean pairwise Jaccard similarity
increases with wavelength binning (exact, 5 nm, 10 nm, 20 nm) for both
the top-30 and top-11 BFS subsets across 5 seeds.

Output: thesis/figures/results/feature_selection/binned_jaccard_summary.png / .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
bins_labels = ["Exact", "5 nm", "10 nm", "20 nm"]
top30_mean  = [0.493, 0.562, 0.697, 0.811]
top30_std   = [0.102, 0.111, 0.081, 0.067]
top11_mean  = [0.398, 0.536, 0.639, 0.640]
top11_std   = [0.062, 0.082, 0.071, 0.065]

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.family": "serif",
})

x = np.arange(len(bins_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5.2))

bars1 = ax.bar(x - width/2, top30_mean, width, yerr=top30_std,
               capsize=4, color="#2E86AB", edgecolor="black", linewidth=0.6,
               label="Top-30", zorder=3, error_kw=dict(lw=1.2))
bars2 = ax.bar(x + width/2, top11_mean, width, yerr=top11_std,
               capsize=4, color="#A23B72", edgecolor="black", linewidth=0.6,
               label="Top-11", zorder=3, error_kw=dict(lw=1.2))

# Annotate values above bars
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Wavelength Matching Resolution")
ax.set_ylabel("Mean Pairwise Jaccard Similarity")
ax.set_title("Effect of Wavelength Binning on Cross-Seed Jaccard Similarity")
ax.set_xticks(x)
ax.set_xticklabels(bins_labels)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.25, axis="y", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent / "results" / "feature_selection"
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "binned_jaccard_summary.png", format="png")
fig.savefig(out_dir / "binned_jaccard_summary.pdf", format="pdf")
plt.close(fig)

print(f"Saved: {out_dir / 'binned_jaccard_summary.png'}")
print(f"Saved: {out_dir / 'binned_jaccard_summary.pdf'}")
