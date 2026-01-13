"""
Spectral Signature Characterization - Descriptive Analysis
MSc Thesis Section

Pure descriptive analysis of hyperspectral signatures without classification.
Computes class-level statistics, spectral region aggregation, and separability metrics.

Author: Yovel
Date: 2026-01-06
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Dict, Tuple

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------
load_dotenv()

# File paths
CSV_PATH = r"C:\Users\yovel\Desktop\Grape_Project\experiments\analysis_gap_between_signature_classes\all_origin_signatures_results_2026-01-06.csv"
RESULT_DIR = Path(__file__).parent / "result" / "descriptive_analysis"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

# Spectral region definitions (nm)
SPECTRAL_REGIONS = {
    'VIS': (400, 700),
    'NIR': (700, 1000)
}

# Plot styling for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


# ------------------------------------------------------------
# Data Loading and Preparation
# ------------------------------------------------------------
def load_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and prepare hyperspectral data.

    Returns:
        X: DataFrame with wavelength features (columns as floats)
        y: Series with binary class labels (0=Healthy, 1=Cracked)
        df: Original DataFrame
    """
    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract wavelength features
    feature_cols = [c for c in df.columns if c.endswith("nm")]
    X = df[feature_cols].copy()
    X.columns = [float(c.replace("nm", "")) for c in feature_cols]
    X = X.reindex(sorted(X.columns), axis=1)

    # Extract labels
    y = (
        df["label"]
        .replace({"REGULAR": 0, "CRACK": 1, "healthy": 0, "sick": 1})
        .astype(int)
    )

    logger.info(f"Loaded {len(df)} samples with {len(X.columns)} wavelength features")
    logger.info(f"Class distribution: Healthy={sum(y==0)}, Cracked={sum(y==1)}")

    return X, y, df


# ------------------------------------------------------------
# Descriptive Statistics Computation
# ------------------------------------------------------------
def compute_class_statistics(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute mean, std, median reflectance for each class at each wavelength.

    Returns:
        DataFrame with columns: wavelength, healthy_mean, healthy_std, healthy_median,
                                cracked_mean, cracked_std, cracked_median
    """
    logger.info("Computing class-level statistics per wavelength...")

    X_healthy = X[y == 0]
    X_cracked = X[y == 1]

    stats_data = []
    for wl in X.columns:
        stats_data.append({
            'wavelength': wl,
            'healthy_mean': X_healthy[wl].mean(),
            'healthy_std': X_healthy[wl].std(),
            'healthy_median': X_healthy[wl].median(),
            'healthy_min': X_healthy[wl].min(),
            'healthy_max': X_healthy[wl].max(),
            'cracked_mean': X_cracked[wl].mean(),
            'cracked_std': X_cracked[wl].std(),
            'cracked_median': X_cracked[wl].median(),
            'cracked_min': X_cracked[wl].min(),
            'cracked_max': X_cracked[wl].max(),
            'n_healthy': len(X_healthy),
            'n_cracked': len(X_cracked)
        })

    stats_df = pd.DataFrame(stats_data)
    logger.info(f"Statistics computed for {len(stats_df)} wavelengths")

    return stats_df


def compute_regional_statistics(X: pd.DataFrame, y: pd.Series,
                                regions: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Aggregate reflectance into spectral regions and compute statistics.

    Returns:
        DataFrame with regional statistics per class
    """
    logger.info("Computing regional statistics...")

    X_healthy = X[y == 0]
    X_cracked = X[y == 1]

    regional_data = []
    for region_name, (wl_min, wl_max) in regions.items():
        # Select wavelengths in region
        region_wls = [wl for wl in X.columns if wl_min <= wl <= wl_max]

        if len(region_wls) == 0:
            logger.warning(f"No wavelengths found in region {region_name} ({wl_min}-{wl_max} nm)")
            continue

        # Compute regional mean reflectance per sample, then aggregate
        healthy_regional = X_healthy[region_wls].mean(axis=1)
        cracked_regional = X_cracked[region_wls].mean(axis=1)

        regional_data.append({
            'region': region_name,
            'wavelength_range': f"{wl_min}-{wl_max} nm",
            'n_wavelengths': len(region_wls),
            'healthy_mean': healthy_regional.mean(),
            'healthy_std': healthy_regional.std(),
            'healthy_median': healthy_regional.median(),
            'cracked_mean': cracked_regional.mean(),
            'cracked_std': cracked_regional.std(),
            'cracked_median': cracked_regional.median(),
        })

    regional_df = pd.DataFrame(regional_data)
    logger.info(f"Regional statistics computed for {len(regional_df)} regions")

    return regional_df


# ------------------------------------------------------------
# Spectral Separability Metrics
# ------------------------------------------------------------
def compute_separability_metrics(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute non-model-based separability metrics between classes.

    Metrics:
    - Absolute Mean Difference: |mean_healthy - mean_cracked|
    - Cohen's d: (mean_healthy - mean_cracked) / pooled_std
    - Fisher Score: (mean_healthy - mean_cracked)^2 / (std_healthy^2 + std_cracked^2)

    Returns:
        DataFrame with separability metrics per wavelength
    """
    logger.info("Computing spectral separability metrics...")

    # Absolute mean difference
    stats_df['abs_mean_diff'] = np.abs(
        stats_df['healthy_mean'] - stats_df['cracked_mean']
    )

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        ((stats_df['n_healthy'] - 1) * stats_df['healthy_std']**2 +
         (stats_df['n_cracked'] - 1) * stats_df['cracked_std']**2) /
        (stats_df['n_healthy'] + stats_df['n_cracked'] - 2)
    )
    stats_df['cohens_d'] = (
        (stats_df['healthy_mean'] - stats_df['cracked_mean']) / pooled_std
    )

    # Fisher score
    stats_df['fisher_score'] = (
        (stats_df['healthy_mean'] - stats_df['cracked_mean'])**2 /
        (stats_df['healthy_std']**2 + stats_df['cracked_std']**2)
    )

    # Relative difference (percentage)
    mean_avg = (stats_df['healthy_mean'] + stats_df['cracked_mean']) / 2
    stats_df['relative_diff_pct'] = (
        stats_df['abs_mean_diff'] / mean_avg * 100
    )

    logger.info("Separability metrics computed")

    return stats_df


def compute_regional_separability(stats_df: pd.DataFrame,
                                  regions: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Aggregate separability metrics by spectral region.

    Args:
        stats_df: DataFrame with wavelength-level separability metrics
        regions: Dictionary of region names to (wl_min, wl_max) tuples

    Returns:
        DataFrame with regional separability statistics
    """
    logger.info("Computing regional separability metrics...")

    regional_sep_data = []

    # Add full wavelength range first
    wl_min_all = stats_df['wavelength'].min()
    wl_max_all = stats_df['wavelength'].max()
    regional_sep_data.append({
        'region': 'Full',
        'wavelength_range': f"{wl_min_all:.2f}-{wl_max_all:.2f} nm",
        'n_wavelengths': len(stats_df),
        'mean_abs_diff': stats_df['abs_mean_diff'].mean(),
        'max_abs_diff': stats_df['abs_mean_diff'].max(),
        'mean_cohens_d': stats_df['cohens_d'].mean(),
        'max_cohens_d': stats_df['cohens_d'].max(),
        'mean_fisher_score': stats_df['fisher_score'].mean(),
        'max_fisher_score': stats_df['fisher_score'].max(),
    })

    # Then add individual regions
    for region_name, (wl_min, wl_max) in regions.items():
        # Select wavelengths in region (same logic as compute_regional_statistics)
        region_mask = (stats_df['wavelength'] >= wl_min) & (stats_df['wavelength'] <= wl_max)
        region_data = stats_df[region_mask]

        if len(region_data) == 0:
            logger.warning(f"No wavelengths found in region {region_name} ({wl_min}-{wl_max} nm)")
            continue

        # Compute mean and max for each separability metric
        regional_sep_data.append({
            'region': region_name,
            'wavelength_range': f"{wl_min}-{wl_max} nm",
            'n_wavelengths': len(region_data),
            'mean_abs_diff': region_data['abs_mean_diff'].mean(),
            'max_abs_diff': region_data['abs_mean_diff'].max(),
            'mean_cohens_d': region_data['cohens_d'].mean(),
            'max_cohens_d': region_data['cohens_d'].max(),
            'mean_fisher_score': region_data['fisher_score'].mean(),
            'max_fisher_score': region_data['fisher_score'].max(),
        })

    regional_sep_df = pd.DataFrame(regional_sep_data)
    logger.info(f"Regional separability computed for {len(regional_sep_df)} regions")

    return regional_sep_df


# ------------------------------------------------------------
# Save Functions
# ------------------------------------------------------------
def save_statistics(stats_df: pd.DataFrame, regional_df: pd.DataFrame, regional_sep_df: pd.DataFrame):
    """Save all statistical summaries to CSV and JSON files."""
    logger.info("Saving statistical summaries...")

    # Save wavelength-level statistics
    csv_path = RESULT_DIR / "wavelength_statistics.csv"
    stats_df.to_csv(csv_path, index=False, float_format='%.6f')
    logger.info(f"Wavelength statistics saved to {csv_path}")

    # Save regional statistics
    regional_csv = RESULT_DIR / "regional_statistics.csv"
    regional_df.to_csv(regional_csv, index=False, float_format='%.6f')
    logger.info(f"Regional statistics saved to {regional_csv}")

    # Save regional separability
    regional_sep_csv = RESULT_DIR / "regional_separability.csv"
    regional_sep_df.to_csv(regional_sep_csv, index=False, float_format='%.6f')
    logger.info(f"Regional separability saved to {regional_sep_csv}")

    # Save top separable wavelengths
    top_fisher = stats_df.nlargest(30, 'fisher_score')[['wavelength', 'fisher_score']]
    top_cohens = stats_df.nlargest(30, 'cohens_d')[['wavelength', 'cohens_d']]
    top_abs_diff = stats_df.nlargest(30, 'abs_mean_diff')[['wavelength', 'abs_mean_diff']]

    top_wl_path = RESULT_DIR / "top_separable_wavelengths.csv"
    pd.DataFrame({
        'rank': range(1, 31),
        'fisher_wl': top_fisher['wavelength'].values,
        'fisher_score': top_fisher['fisher_score'].values,
        'cohens_wl': top_cohens['wavelength'].values,
        'cohens_d': top_cohens['cohens_d'].values,
        'abs_diff_wl': top_abs_diff['wavelength'].values,
        'abs_mean_diff': top_abs_diff['abs_mean_diff'].values,
    }).to_csv(top_wl_path, index=False, float_format='%.6f')
    logger.info(f"Top separable wavelengths saved to {top_wl_path}")

    # Save comprehensive JSON summary
    summary = {
        'dataset_info': {
            'n_samples': int(stats_df['n_healthy'].iloc[0] + stats_df['n_cracked'].iloc[0]),
            'n_healthy': int(stats_df['n_healthy'].iloc[0]),
            'n_cracked': int(stats_df['n_cracked'].iloc[0]),
            'n_wavelengths': len(stats_df),
            'wavelength_range': f"{stats_df['wavelength'].min()}-{stats_df['wavelength'].max()} nm"
        },
        'regional_statistics': regional_df.to_dict(orient='records'),
        'regional_separability': regional_sep_df.to_dict(orient='records'),
        'top_separable_wavelengths': {
            'fisher_score': top_fisher.to_dict(orient='records'),
            'cohens_d': top_cohens.to_dict(orient='records'),
            'abs_mean_diff': top_abs_diff.to_dict(orient='records')
        },
        'separability_summary': {
            'mean_fisher_score': float(stats_df['fisher_score'].mean()),
            'max_fisher_score': float(stats_df['fisher_score'].max()),
            'mean_cohens_d': float(stats_df['cohens_d'].mean()),
            'max_cohens_d': float(stats_df['cohens_d'].max()),
            'mean_abs_diff': float(stats_df['abs_mean_diff'].mean()),
            'max_abs_diff': float(stats_df['abs_mean_diff'].max()),
        }
    }

    json_path = RESULT_DIR / "analysis_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Analysis summary saved to {json_path}")



# ------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------
def plot_spectral_signatures(stats_df: pd.DataFrame):
    """Plot mean ± STD spectral signatures for both classes."""
    logger.info("Generating spectral signature plots...")

    wavelengths = stats_df['wavelength'].values

    # Calculate percentage within ±1 STD (this is typically ~68% for normal distribution)
    healthy_within_1std = 68.0  # theoretical for normal distribution
    cracked_within_1std = 68.0

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Healthy class
    ax.plot(wavelengths, stats_df['healthy_mean'],
            color='#2E7D32', linewidth=2.5, label='Healthy (Mean)', zorder=3)
    ax.fill_between(wavelengths,
                     stats_df['healthy_mean'] - stats_df['healthy_std'],
                     stats_df['healthy_mean'] + stats_df['healthy_std'],
                     color='#2E7D32', alpha=0.2, label='Healthy (±1 STD)')

    # Plot Cracked class
    ax.plot(wavelengths, stats_df['cracked_mean'],
            color='#C62828', linewidth=2.5, label='Cracked (Mean)', zorder=3)
    ax.fill_between(wavelengths,
                     stats_df['cracked_mean'] - stats_df['cracked_std'],
                     stats_df['cracked_mean'] + stats_df['cracked_std'],
                     color='#C62828', alpha=0.2, label='Cracked (±1 STD)')

    # Add vertical lines for spectral regions
    ax.axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(550, ax.get_ylim()[1] * 0.95, 'VIS', fontsize=11,
            ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.text(850, ax.get_ylim()[1] * 0.95, 'NIR', fontsize=11,
            ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Formatting
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title('Spectral Signatures: Healthy vs. Cracked Grape Tissue\n' +
                 '(Mean ± Standard Deviation)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)

    # Add statistics annotation
    annotation_text = (
        f"Sample sizes:\n"
        f"  Healthy: n={int(stats_df['n_healthy'].iloc[0])}\n"
        f"  Cracked: n={int(stats_df['n_cracked'].iloc[0])}"
    )
    ax.text(0.02, 0.98, annotation_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig_path = RESULT_DIR / "spectral_signatures_mean_std.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Spectral signature plot saved to {fig_path}")
    plt.close()


def plot_separability_metrics(stats_df: pd.DataFrame):
    """Plot separability metrics across wavelengths."""
    logger.info("Generating separability metrics plots...")

    wavelengths = stats_df['wavelength'].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Fisher Score
    axes[0].plot(wavelengths, stats_df['fisher_score'],
                 color='#1976D2', linewidth=1.5)
    axes[0].fill_between(wavelengths, 0, stats_df['fisher_score'],
                         color='#1976D2', alpha=0.3)
    axes[0].set_ylabel('Fisher Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Fisher Score = (μ₁ - μ₂)² / (σ₁² + σ₂²)', fontsize=11, style='italic')
    axes[0].grid(True, alpha=0.3, linestyle=':')
    axes[0].axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Annotate top wavelength
    max_idx = stats_df['fisher_score'].idxmax()
    max_wl = stats_df.loc[max_idx, 'wavelength']
    max_val = stats_df.loc[max_idx, 'fisher_score']
    axes[0].annotate(f'Max: {max_wl:.1f} nm',
                     xy=(max_wl, max_val), xytext=(max_wl + 50, max_val * 0.9),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=10, color='red', fontweight='bold')

    # Plot 2: Cohen's d
    axes[1].plot(wavelengths, stats_df['cohens_d'],
                 color='#388E3C', linewidth=1.5)
    axes[1].fill_between(wavelengths, 0, stats_df['cohens_d'],
                         color='#388E3C', alpha=0.3)
    axes[1].set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
    axes[1].set_title("Cohen's d = (μ₁ - μ₂) / σ_pooled", fontsize=11, style='italic')
    axes[1].grid(True, alpha=0.3, linestyle=':')
    axes[1].axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add effect size interpretation lines
    axes[1].axhline(y=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    axes[1].axhline(y=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    axes[1].text(wavelengths[-1] * 0.98, 0.2, 'small', fontsize=8, alpha=0.6, ha='right')
    axes[1].text(wavelengths[-1] * 0.98, 0.5, 'medium', fontsize=8, alpha=0.6, ha='right')
    axes[1].text(wavelengths[-1] * 0.98, 0.8, 'large', fontsize=8, alpha=0.6, ha='right')

    # Plot 3: Absolute Mean Difference
    axes[2].plot(wavelengths, stats_df['abs_mean_diff'],
                 color='#D32F2F', linewidth=1.5)
    axes[2].fill_between(wavelengths, 0, stats_df['abs_mean_diff'],
                         color='#D32F2F', alpha=0.3)
    axes[2].set_ylabel('Absolute Mean\nDifference', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    axes[2].set_title('Absolute Mean Difference = |μ_healthy - μ_cracked|',
                      fontsize=11, style='italic')
    axes[2].grid(True, alpha=0.3, linestyle=':')
    axes[2].axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Overall title
    fig.suptitle('Spectral Separability Metrics: Healthy vs. Cracked Tissue',
                 fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()
    fig_path = RESULT_DIR / "separability_metrics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Separability metrics plot saved to {fig_path}")
    plt.close()


def plot_regional_comparison(regional_df: pd.DataFrame):
    """Plot regional statistics comparison."""
    logger.info("Generating regional comparison plot...")

    if len(regional_df) == 0:
        logger.warning("No regional data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    regions = regional_df['region'].values
    x = np.arange(len(regions))
    width = 0.35

    # Plot 1: Mean reflectance per region
    healthy_means = regional_df['healthy_mean'].values
    cracked_means = regional_df['cracked_mean'].values

    axes[0].bar(x - width/2, healthy_means, width, label='Healthy',
                color='#2E7D32', alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].bar(x + width/2, cracked_means, width, label='Cracked',
                color='#C62828', alpha=0.8, edgecolor='black', linewidth=1)

    # Add error bars for std
    axes[0].errorbar(x - width/2, healthy_means, yerr=regional_df['healthy_std'].values,
                     fmt='none', ecolor='black', capsize=5, alpha=0.6)
    axes[0].errorbar(x + width/2, cracked_means, yerr=regional_df['cracked_std'].values,
                     fmt='none', ecolor='black', capsize=5, alpha=0.6)

    axes[0].set_ylabel('Mean Reflectance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Spectral Region', fontsize=12, fontweight='bold')
    axes[0].set_title('Regional Mean Reflectance (±1 STD)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r}\n{regional_df.loc[i, 'wavelength_range']}"
                             for i, r in enumerate(regions)])
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle=':')

    # Plot 2: Absolute difference per region
    abs_diff = np.abs(healthy_means - cracked_means)
    colors = ['#1976D2' if d == max(abs_diff) else '#64B5F6' for d in abs_diff]

    bars = axes[1].bar(x, abs_diff, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Absolute Mean Difference', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Spectral Region', fontsize=12, fontweight='bold')
    axes[1].set_title('Regional Separability', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r}\n{regional_df.loc[i, 'wavelength_range']}"
                             for i, r in enumerate(regions)])
    axes[1].grid(True, alpha=0.3, axis='y', linestyle=':')

    # Annotate values on bars
    for i, (bar, val) in enumerate(zip(bars, abs_diff)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = RESULT_DIR / "regional_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Regional comparison plot saved to {fig_path}")
    plt.close()


def plot_combined_overview(stats_df: pd.DataFrame, regional_df: pd.DataFrame):
    """Create a comprehensive 4-panel overview figure."""
    logger.info("Generating combined overview figure...")

    wavelengths = stats_df['wavelength'].values

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel A: Spectral signatures
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(wavelengths, stats_df['healthy_mean'],
            color='#2E7D32', linewidth=2, label='Healthy (Mean)')
    ax1.fill_between(wavelengths,
                     stats_df['healthy_mean'] - stats_df['healthy_std'],
                     stats_df['healthy_mean'] + stats_df['healthy_std'],
                     color='#2E7D32', alpha=0.2)
    ax1.plot(wavelengths, stats_df['cracked_mean'],
            color='#C62828', linewidth=2, label='Cracked (Mean)')
    ax1.fill_between(wavelengths,
                     stats_df['cracked_mean'] - stats_df['cracked_std'],
                     stats_df['cracked_mean'] + stats_df['cracked_std'],
                     color='#C62828', alpha=0.2)
    ax1.axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Reflectance', fontsize=11, fontweight='bold')
    ax1.set_title('A. Mean Spectral Signatures (±1 STD)', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Panel B: Fisher Score
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(wavelengths, stats_df['fisher_score'], color='#1976D2', linewidth=1.5)
    ax2.fill_between(wavelengths, 0, stats_df['fisher_score'], color='#1976D2', alpha=0.3)
    ax2.axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Fisher Score', fontsize=11, fontweight='bold')
    ax2.set_title('B. Fisher Score', fontsize=12, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, linestyle=':')

    # Panel C: Cohen's d
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(wavelengths, stats_df['cohens_d'], color='#388E3C', linewidth=1.5)
    ax3.fill_between(wavelengths, 0, stats_df['cohens_d'], color='#388E3C', alpha=0.3)
    ax3.axvline(x=700, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(y=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.4, label='Large effect')
    ax3.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax3.set_title("C. Cohen's d (Effect Size)", fontsize=12, fontweight='bold', loc='left')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle=':')

    # Panel D: Regional comparison
    ax4 = fig.add_subplot(gs[2, :])
    if len(regional_df) > 0:
        regions = regional_df['region'].values
        x = np.arange(len(regions))
        width = 0.35
        ax4.bar(x - width/2, regional_df['healthy_mean'].values, width,
                label='Healthy', color='#2E7D32', alpha=0.8, edgecolor='black')
        ax4.bar(x + width/2, regional_df['cracked_mean'].values, width,
                label='Cracked', color='#C62828', alpha=0.8, edgecolor='black')
        ax4.errorbar(x - width/2, regional_df['healthy_mean'].values,
                     yerr=regional_df['healthy_std'].values,
                     fmt='none', ecolor='black', capsize=5, alpha=0.6)
        ax4.errorbar(x + width/2, regional_df['cracked_mean'].values,
                     yerr=regional_df['cracked_std'].values,
                     fmt='none', ecolor='black', capsize=5, alpha=0.6)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{r}\n({regional_df.loc[i, 'wavelength_range']})"
                             for i, r in enumerate(regions)])
        ax4.set_ylabel('Mean Reflectance', fontsize=11, fontweight='bold')
        ax4.set_title('D. Regional Statistics', fontsize=12, fontweight='bold', loc='left')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Set x-label for bottom plots
    for ax in [ax2, ax3]:
        ax.set_xlabel('Wavelength (nm)', fontsize=11, fontweight='bold')

    fig.suptitle('Spectral Signature Characterization: Comprehensive Overview',
                 fontsize=15, fontweight='bold', y=0.998)

    fig_path = RESULT_DIR / "comprehensive_overview.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comprehensive overview saved to {fig_path}")
    plt.close()


# ------------------------------------------------------------
# Main Analysis Pipeline
# ------------------------------------------------------------
def main():
    """Execute the complete descriptive analysis pipeline."""
    logger.info("=" * 60)
    logger.info("SPECTRAL SIGNATURE CHARACTERIZATION - DESCRIPTIVE ANALYSIS")
    logger.info("=" * 60)

    # Step 1: Load data
    X, y, df = load_data(CSV_PATH)

    # Step 2: Compute class-level statistics
    stats_df = compute_class_statistics(X, y)

    # Step 3: Compute regional statistics
    regional_df = compute_regional_statistics(X, y, SPECTRAL_REGIONS)

    # Step 4: Compute separability metrics
    stats_df = compute_separability_metrics(stats_df)

    # Step 5: Compute regional separability
    regional_sep_df = compute_regional_separability(stats_df, SPECTRAL_REGIONS)

    # Step 6: Save all statistics
    save_statistics(stats_df, regional_df, regional_sep_df)

    # Step 7: Generate visualizations
    plot_spectral_signatures(stats_df)
    plot_separability_metrics(stats_df)
    plot_regional_comparison(regional_df)
    plot_combined_overview(stats_df, regional_df)

    # Summary
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {RESULT_DIR}")
    logger.info(f"  - wavelength_statistics.csv")
    logger.info(f"  - regional_statistics.csv")
    logger.info(f"  - regional_separability.csv")
    logger.info(f"  - top_separable_wavelengths.csv")
    logger.info(f"  - analysis_summary.json")
    logger.info(f"  - 5 publication-quality figures")
    logger.info("=" * 60)




if __name__ == "__main__":
    main()







