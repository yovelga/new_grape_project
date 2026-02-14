"""
Analyze wavelength popularity across BFS stability runs.
Produces SEPARATE results for fixedsplit and varyingsplit in two different folders.
Bins wavelengths into different sizes (1nm, 5nm, 10nm, 20nm) and shows most popular.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

BASE_DIR = Path("experiments/feature_selection/stability_bfs_prauc")


def bin_wavelengths(wavelengths, bin_size):
    """Bin wavelengths into specified bin size (nm)"""
    bins = {}
    for wl in wavelengths:
        bin_center = round(wl / bin_size) * bin_size
        bins[bin_center] = bins.get(bin_center, 0) + 1
    return bins


def collect_wavelengths(source_dir, glob_pattern):
    """Collect wavelengths from all run directories."""
    wavelengths = []
    per_run = {}  # run_name -> list of wavelengths
    for run_dir in sorted(source_dir.glob(glob_pattern)):
        wl_file = run_dir / "selected_wavelengths_at30.csv"
        if wl_file.exists():
            df = pd.read_csv(wl_file)
            wls = df['wavelength'].str.replace('nm', '').astype(float).tolist()
            wavelengths.extend(wls)
            per_run[run_dir.name] = wls
            print(f"  ✓ {run_dir.name}: {len(wls)} wavelengths")
    return np.array(wavelengths), per_run


def print_popularity_tables(wavelengths, label):
    """Print text-based popularity tables for all bin sizes."""
    bin_sizes = [1, 5, 10, 20]
    for bin_size in bin_sizes:
        print(f"\n{'─' * 80}")
        print(f"BIN SIZE: {bin_size} nm")
        print(f"{'─' * 80}")

        bins = bin_wavelengths(wavelengths, bin_size)
        sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Rank':<6} {'Wavelength Range':<35} {'Count':<8} {'%':<8} {'█' * 20}")
        print("─" * 80)

        max_count = sorted_bins[0][1] if sorted_bins else 1
        for rank, (center, count) in enumerate(sorted_bins[:20], 1):
            pct = (count / len(wavelengths)) * 100
            bar_length = int((count / max_count) * 40)
            bar = '█' * bar_length
            if bin_size == 1:
                range_str = f"{center:.2f}nm"
            else:
                half = bin_size / 2
                range_str = f"{center-half:.1f}-{center+half:.1f}nm (c:{center:.0f})"
            print(f"{rank:<6} {range_str:<35} {count:<8} {pct:>5.1f}%   {bar}")


def generate_plots(wavelengths, per_run, output_dir, mode_label, color_primary):
    """Generate all 4 plots for a single split mode."""
    n_runs = len(per_run)
    wl_counts = Counter(wavelengths)

    # ── Plot 1: Overview figure ──
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Histogram
    ax1 = fig.add_subplot(gs[0, :])
    bins_hist = np.arange(450, 926, 5)
    ax1.hist(wavelengths, bins=bins_hist, alpha=0.7, color=color_primary, edgecolor='black')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution of Selected Wavelengths – {mode_label} ({n_runs} Runs, 5nm bins)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Top 20 individual wavelengths
    ax2 = fig.add_subplot(gs[1, 0])
    top_20_wl = wl_counts.most_common(20)
    wavelengths_top20 = [wl for wl, _ in top_20_wl]
    counts_top20 = [c for _, c in top_20_wl]
    bars = ax2.barh(range(len(wavelengths_top20)), counts_top20, color='coral', edgecolor='black')
    ax2.set_yticks(range(len(wavelengths_top20)))
    ax2.set_yticklabels([f'{wl:.2f}nm' for wl in wavelengths_top20], fontsize=9)
    ax2.set_xlabel('Selection Count', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top 20 Most Selected Wavelengths – {mode_label}', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, counts_top20)):
        ax2.text(count + 0.3, i, str(count), va='center', fontsize=8)

    # Binned analysis – 20nm
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

    plt.savefig(output_dir / 'wavelength_popularity_overview.png', bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {output_dir / 'wavelength_popularity_overview.png'}")
    plt.close()

    # ── Plot 2: Binning comparison ──
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Wavelength Popularity with Different Bin Sizes – {mode_label}',
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
    plt.savefig(output_dir / 'wavelength_binning_comparison.png', bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {output_dir / 'wavelength_binning_comparison.png'}")
    plt.close()

    # ── Plot 3: Wavelength Stability Heatmap (5nm bins, score 0-5) ──
    bin_size_hm = 5
    bin_edges = np.arange(450, 926, bin_size_hm)

    # For each run, find which 5nm bins have at least one selected wavelength
    run_bin_sets = []
    for run_name, wls in per_run.items():
        run_bins = set()
        for wl in wls:
            bin_center = round(wl / bin_size_hm) * bin_size_hm
            run_bins.add(bin_center)
        run_bin_sets.append(run_bins)

    # Collect all bins that appear in at least 1 run
    all_bins_used = sorted(set().union(*run_bin_sets))

    # Compute stability score: how many runs selected each bin (0-5)
    stability = {}
    for b in all_bins_used:
        stability[b] = sum(1 for rbs in run_bin_sets if b in rbs)

    # Build a single-column DataFrame for the heatmap
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
    ax.set_title(f'Wavelength Stability – {mode_label}\n(5 nm bins, score = # runs out of {n_runs})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'wavelength_stability_heatmap.png', bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {output_dir / 'wavelength_stability_heatmap.png'}")
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
    ax.set_title(f'Wavelength Selection Frequency Along Spectrum – {mode_label}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'wavelength_frequency_spectrum.png', bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {output_dir / 'wavelength_frequency_spectrum.png'}")
    plt.close()


def print_summary(wavelengths, wl_counts, mode_label):
    """Print summary statistics and top-30 table."""
    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS – {mode_label}")
    print(f"{'=' * 80}")
    print(f"Total wavelength selections: {len(wavelengths)}")
    print(f"Unique wavelengths (exact): {len(set(wavelengths))}")
    print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
    print(f"Mean wavelength: {wavelengths.mean():.2f} nm")
    print(f"Median wavelength: {np.median(wavelengths):.2f} nm")

    print(f"\n{'=' * 80}")
    print(f"TOP 30 INDIVIDUAL WAVELENGTHS (NO BINNING) – {mode_label}")
    print(f"{'=' * 80}")
    print(f"\n{'Rank':<6} {'Wavelength':<15} {'Count':<8} {'%':<8} {'█' * 20}")
    print("─" * 80)
    top1_count = wl_counts.most_common(1)[0][1] if wl_counts else 1
    for rank, (wl, count) in enumerate(wl_counts.most_common(30), 1):
        pct = (count / len(wavelengths)) * 100
        bar_length = int((count / top1_count) * 40)
        bar = '█' * bar_length
        print(f"{rank:<6} {wl:<15.2f} {count:<8} {pct:>5.1f}%   {bar}")


# ============================================================================
# RUN ANALYSIS FOR EACH SPLIT MODE SEPARATELY
# ============================================================================

MODES = [
    {
        'name': 'fixedsplit',
        'label': 'Fixed Split (seed=42)',
        'source_dir': BASE_DIR / 'fixedsplit',
        'glob': 'prauc_fixed_*',
        'output_dir': BASE_DIR / '_analysis_fixedsplit',
        'color': '#2E86AB',
    },
    {
        'name': 'varyingsplit',
        'label': 'Varying Split (seed=model_seed)',
        'source_dir': BASE_DIR / 'varyingsplit',
        'glob': 'prauc_varying_*',
        'output_dir': BASE_DIR / '_analysis_varyingsplit',
        'color': '#A23B72',
    },
]

for mode in MODES:
    mode['output_dir'] = Path(mode['output_dir'])
    mode['output_dir'].mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print(f"  {mode['label'].upper()}")
    print("=" * 80)

    # Collect
    print("\nCollecting wavelengths...")
    wavelengths, per_run = collect_wavelengths(mode['source_dir'], mode['glob'])

    if len(wavelengths) == 0:
        print("  ⚠ No wavelength files found – skipping.")
        continue

    print(f"\n  Total: {len(wavelengths)} wavelengths from {len(per_run)} runs")

    # Text tables
    print(f"\n{'=' * 80}")
    print(f"WAVELENGTH POPULARITY – {mode['label']}")
    print(f"{'=' * 80}")
    print_popularity_tables(wavelengths, mode['label'])

    # Plots
    print(f"\n  Generating plots → {mode['output_dir']} ...")
    generate_plots(wavelengths, per_run, mode['output_dir'], mode['label'], mode['color'])

    # Summary
    wl_counts = Counter(wavelengths)
    print_summary(wavelengths, wl_counts, mode['label'])

print("\n" + "=" * 80)
print("DONE – results saved to:")
for mode in MODES:
    print(f"  • {mode['output_dir']}")
print("=" * 80)
