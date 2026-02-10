"""
Analyze wavelength popularity across BFS stability runs – VARYING SPLIT ONLY.
Produces results for both 30 WL and 11 WL subsets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

BASE_DIR = Path("experiments/feature_selection/stability_bfs_prauc")
OUTPUT_BASE = Path("results/feature_selection")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)


def bin_wavelengths(wavelengths, bin_size):
    bins = {}
    for wl in wavelengths:
        bin_center = round(wl / bin_size) * bin_size
        bins[bin_center] = bins.get(bin_center, 0) + 1
    return bins


def collect_wavelengths(source_dir, glob_pattern, wl_file_name):
    """Collect wavelengths from all run directories."""
    wavelengths = []
    per_run = {}
    for run_dir in sorted(source_dir.glob(glob_pattern)):
        wl_file = run_dir / wl_file_name
        if wl_file.exists():
            df = pd.read_csv(wl_file)
            wls = df['wavelength'].str.replace('nm', '').astype(float).tolist()
            wavelengths.extend(wls)
            per_run[run_dir.name] = wls
            print(f"  ✓ {run_dir.name}: {len(wls)} wavelengths")
    return np.array(wavelengths), per_run


def generate_plots(wavelengths, per_run, output_dir, prefix, mode_label, color_primary, n_wl):
    """Generate all 4 plots."""
    n_runs = len(per_run)
    wl_counts = Counter(wavelengths)

    # ── Plot 1: Overview figure ──
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    bins_hist = np.arange(450, 926, 5)
    ax1.hist(wavelengths, bins=bins_hist, alpha=0.7, color=color_primary, edgecolor='black')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution of Selected Wavelengths – {mode_label} (Top {n_wl}, {n_runs} Runs, 5nm bins)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    n_show = min(20, len(wl_counts))
    top_wl = wl_counts.most_common(n_show)
    wavelengths_top = [wl for wl, _ in top_wl]
    counts_top = [c for _, c in top_wl]
    bars = ax2.barh(range(len(wavelengths_top)), counts_top, color='coral', edgecolor='black')
    ax2.set_yticks(range(len(wavelengths_top)))
    ax2.set_yticklabels([f'{wl:.2f}nm' for wl in wavelengths_top], fontsize=9)
    ax2.set_xlabel('Selection Count', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {n_show} Most Selected Wavelengths – {mode_label}', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, counts_top)):
        ax2.text(count + 0.3, i, str(count), va='center', fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    bins_20nm = bin_wavelengths(wavelengths, 20)
    sorted_bins_20 = sorted(bins_20nm.items(), key=lambda x: x[0])
    centers_20 = [c for c, _ in sorted_bins_20]
    counts_20 = [cnt for _, cnt in sorted_bins_20]
    ax3.bar(centers_20, counts_20, width=18, alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax3.set_xlabel('Wavelength Center (nm)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title(f'Wavelength Popularity (20nm bins) – {mode_label}', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    out_file = output_dir / f'{prefix}wavelength_popularity_overview.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {out_file}")
    plt.close()

    # ── Plot 2: Binning comparison ──
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Wavelength Popularity with Different Bin Sizes – {mode_label} (Top {n_wl})',
                  fontsize=16, fontweight='bold')
    bin_configs = [
        (1, 'darkblue', axes[0, 0]),
        (5, 'darkgreen', axes[0, 1]),
        (10, 'darkorange', axes[1, 0]),
        (20, 'darkred', axes[1, 1]),
    ]
    for bin_size, color, ax in bin_configs:
        bins = bin_wavelengths(wavelengths, bin_size)
        sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)[:30]
        centers = [c for c, _ in sorted_bins]
        counts = [cnt for _, cnt in sorted_bins]
        bars = ax.barh(range(len(centers)), counts, color=color, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(centers)))
        ax.set_yticklabels([f'{c:.1f}nm' for c in centers], fontsize=8)
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'Top 30 Wavelengths ({bin_size}nm bins)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=7)
    plt.tight_layout()
    out_file = output_dir / f'{prefix}wavelength_binning_comparison.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {out_file}")
    plt.close()

    # ── Plot 3: Wavelength Stability Heatmap ──
    bin_size_hm = 5
    run_bin_sets = []
    for run_name, wls in per_run.items():
        run_bins = set()
        for wl in wls:
            bin_center = round(wl / bin_size_hm) * bin_size_hm
            run_bins.add(bin_center)
        run_bin_sets.append(run_bins)

    all_bins_used = sorted(set().union(*run_bin_sets))
    stability = {}
    for b in all_bins_used:
        stability[b] = sum(1 for rbs in run_bin_sets if b in rbs)

    bins_sorted = sorted(stability.keys())
    scores = [stability[b] for b in bins_sorted]
    labels = [f'{b - bin_size_hm/2:.0f}–{b + bin_size_hm/2:.0f} nm' for b in bins_sorted]
    stability_df = pd.DataFrame({'Stability': scores}, index=labels)

    fig3, ax = plt.subplots(figsize=(6, max(8, len(bins_sorted) * 0.28)))
    sns.heatmap(stability_df, cmap='YlOrRd', cbar_kws={'label': 'Runs selecting this bin (out of 5)'},
                linewidths=0.5, ax=ax, vmin=0, vmax=n_runs, annot=True, fmt='d',
                xticklabels=True, yticklabels=True)
    ax.set_xlabel('')
    ax.set_ylabel('Wavelength Bin (nm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Wavelength Stability – {mode_label} (Top {n_wl})\n(5 nm bins, score = # runs out of {n_runs})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_file = output_dir / f'{prefix}wavelength_stability_heatmap.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {out_file}")
    plt.close()

    # ── Plot 4: Frequency along spectrum ──
    fig4, ax = plt.subplots(figsize=(16, 6))
    window_size = 2
    wave_range = np.arange(450, 926, window_size)
    frequencies = []
    for center in wave_range:
        count = np.sum((wavelengths >= center - window_size / 2) &
                       (wavelengths < center + window_size / 2))
        frequencies.append(count)
    ax.fill_between(wave_range, frequencies, alpha=0.3, color=color_primary)
    ax.plot(wave_range, frequencies, color=color_primary, linewidth=2)
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Selection Frequency ({window_size}nm windows)', fontsize=12, fontweight='bold')
    ax.set_title(f'Wavelength Selection Frequency Along Spectrum – {mode_label} (Top {n_wl})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = output_dir / f'{prefix}wavelength_frequency_spectrum.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {out_file}")
    plt.close()


# ============================================================================
# RUN ANALYSIS – VARYING SPLIT ONLY, for 30 WL and 11 WL
# ============================================================================

SOURCE_DIR = BASE_DIR / 'varyingsplit'
GLOB = 'prauc_varying_*'
COLOR = '#A23B72'

CONFIGS = [
    {'n_wl': 30, 'wl_file': 'selected_wavelengths_at30.csv', 'prefix': 'stability_varying_30wl_'},
    {'n_wl': 11, 'wl_file': 'selected_wavelengths_at11.csv', 'prefix': 'stability_varying_11wl_'},
]

for cfg in CONFIGS:
    n_wl = cfg['n_wl']
    label = f'Varying Split (Top {n_wl} WL)'
    print("\n" + "=" * 80)
    print(f"  {label.upper()}")
    print("=" * 80)

    wavelengths, per_run = collect_wavelengths(SOURCE_DIR, GLOB, cfg['wl_file'])
    if len(wavelengths) == 0:
        print("  ⚠ No wavelength files found – skipping.")
        continue

    print(f"\n  Total: {len(wavelengths)} wavelengths from {len(per_run)} runs")
    generate_plots(wavelengths, per_run, OUTPUT_BASE, cfg['prefix'], label, COLOR, n_wl)

print("\n" + "=" * 80)
print(f"DONE – all images saved to: {OUTPUT_BASE}")
print("=" * 80)
