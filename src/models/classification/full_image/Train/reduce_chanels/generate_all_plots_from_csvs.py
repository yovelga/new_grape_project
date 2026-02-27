"""
Comprehensive Plot Generation Script
Generates all thesis-ready plots from the CSV results files using wavelength labels
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
from wavelengths import WAVELENGTHS

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Define paths
results_dir = Path(str(_PROJECT_ROOT / r"src/models/classification/full_image/Train/reduce_chanels/results_multi_class_reduce_CRACK_F1"))
output_dir = results_dir / "new_plots"
output_dir.mkdir(exist_ok=True)

def band_to_wavelength(band_name):
    """Convert band_XX to wavelength in nm"""
    if isinstance(band_name, str) and band_name.startswith('band_'):
        band_num = int(band_name.split('_')[1])
        return WAVELENGTHS.get(band_num, band_num)
    return band_name

def format_wavelength(wl):
    """Format wavelength for display"""
    return f"{wl:.1f}nm"

print("="*80)
print("COMPREHENSIVE PLOT GENERATION FROM CSV FILES")
print("="*80)

# ============================================================================
# PLOT 1: Feature Selection Performance Metrics
# ============================================================================
print("\n[1/8] Generating SFS Performance Metrics plot...")
log_df = pd.read_csv(results_dir / "feature_selection_log_crack_f1.csv")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot all metrics
ax.plot(log_df['num_features'], log_df['crack_f1'], 'o-', linewidth=3, markersize=8,
        label='CRACK F1 Score ⭐', color='crimson')
ax.plot(log_df['num_features'], log_df['crack_f2'], 's-', linewidth=2, markersize=6,
        label='CRACK F2 Score', color='orangered', alpha=0.8)
ax.plot(log_df['num_features'], log_df['accuracy'], '^-', linewidth=2, markersize=6,
        label='Overall Accuracy', color='steelblue', alpha=0.7)
ax.plot(log_df['num_features'], log_df['crack_precision'], 'd-', linewidth=2, markersize=6,
        label='CRACK Precision', color='green', alpha=0.7)
ax.plot(log_df['num_features'], log_df['crack_recall'], 'v-', linewidth=2, markersize=6,
        label='CRACK Recall', color='purple', alpha=0.7)
ax.plot(log_df['num_features'], log_df['crack_auc'], 'p-', linewidth=2, markersize=6,
        label='CRACK AUC', color='teal', alpha=0.7)

# Mark the best model
best_idx = log_df['crack_f1'].idxmax()
best_f1 = log_df.loc[best_idx, 'crack_f1']
best_n = log_df.loc[best_idx, 'num_features']
ax.plot(best_n, best_f1, 'r*', markersize=25, label=f'Best Model ({int(best_n)} wavelengths)',
        markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('Number of Selected Wavelengths', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Sequential Feature Selection Performance\nCRACK Class F1-Score Optimization',
             fontweight='bold', pad=20)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(output_dir / "01_sfs_performance_all_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_sfs_performance_all_metrics.png")

# ============================================================================
# PLOT 2: CRACK-Specific Metrics Evolution
# ============================================================================
print("\n[2/8] Generating CRACK-specific metrics evolution...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top: F1 and F2 scores
ax1.plot(log_df['num_features'], log_df['crack_f1'], 'o-', linewidth=3, markersize=8,
         label='F1 Score', color='crimson')
ax1.plot(log_df['num_features'], log_df['crack_f2'], 's-', linewidth=2.5, markersize=7,
         label='F2 Score', color='orangered')
ax1.axvline(x=best_n, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('CRACK Class Performance Metrics Evolution', fontweight='bold', pad=15)
ax1.legend(loc='lower right', framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Bottom: Precision and Recall
ax2.plot(log_df['num_features'], log_df['crack_precision'], 'd-', linewidth=2.5, markersize=7,
         label='Precision', color='green')
ax2.plot(log_df['num_features'], log_df['crack_recall'], 'v-', linewidth=2.5, markersize=7,
         label='Recall', color='purple')
ax2.axvline(x=best_n, color='red', linestyle='--', alpha=0.5, linewidth=2,
            label=f'Optimal ({int(best_n)} wavelengths)')
ax2.set_xlabel('Number of Selected Wavelengths', fontweight='bold')
ax2.set_ylabel('Score', fontweight='bold')
ax2.legend(loc='lower right', framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(output_dir / "02_crack_metrics_evolution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_crack_metrics_evolution.png")

# ============================================================================
# PLOT 3: Threshold Optimization Curve
# ============================================================================
print("\n[3/8] Generating Threshold Optimization plot...")
threshold_df = pd.read_csv(results_dir / "threshold_optimization_log.csv")

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(threshold_df['threshold'], threshold_df['crack_f1'], '-', linewidth=3,
        label='F1 Score', color='crimson')
ax.plot(threshold_df['threshold'], threshold_df['crack_precision'], '--', linewidth=2,
        label='Precision', color='green', alpha=0.8)
ax.plot(threshold_df['threshold'], threshold_df['crack_recall'], '--', linewidth=2,
        label='Recall', color='purple', alpha=0.8)

# Mark optimal threshold
best_threshold_idx = threshold_df['crack_f1'].idxmax()
best_threshold = threshold_df.loc[best_threshold_idx, 'threshold']
best_f1_at_threshold = threshold_df.loc[best_threshold_idx, 'crack_f1']

ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2.5,
           label=f'Optimal Threshold = {best_threshold:.3f}')
ax.plot(best_threshold, best_f1_at_threshold, 'r*', markersize=20,
        markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('Classification Threshold', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Threshold Optimization for CRACK Class\nMaximizing F1-Score',
             fontweight='bold', pad=20)
ax.legend(loc='best', framealpha=0.95, fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

# Add annotation
ax.annotate(f'Best F1 = {best_f1_at_threshold:.4f}',
            xy=(best_threshold, best_f1_at_threshold),
            xytext=(best_threshold + 0.15, best_f1_at_threshold - 0.15),
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))

plt.tight_layout()
plt.savefig(output_dir / "03_threshold_optimization.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_threshold_optimization.png")

# ============================================================================
# PLOT 4: Feature Selection Frequency
# ============================================================================
print("\n[4/8] Generating Feature Selection Frequency plot...")
freq_df = pd.read_csv(results_dir / "feature_selection_frequency.csv")
# Filter only selected features
freq_df_selected = freq_df[freq_df['Selection_Count'] > 0].copy()
freq_df_selected['Wavelength'] = freq_df_selected['Feature'].apply(band_to_wavelength)
freq_df_selected['Wavelength_Label'] = freq_df_selected['Wavelength'].apply(format_wavelength)
freq_df_selected = freq_df_selected.sort_values('Selection_Count', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.RdYlGn(freq_df_selected['Selection_Count'] / freq_df_selected['Selection_Count'].max())
bars = ax.barh(range(len(freq_df_selected)), freq_df_selected['Selection_Count'], color=colors, alpha=0.8)

ax.set_yticks(range(len(freq_df_selected)))
ax.set_yticklabels(freq_df_selected['Wavelength_Label'], fontsize=10)
ax.set_xlabel('Selection Frequency (out of 20 iterations)', fontweight='bold')
ax.set_ylabel('Wavelength', fontweight='bold')
ax.set_title('Wavelength Selection Frequency During Sequential Feature Selection\n(Higher = More Stable/Important)',
             fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(freq_df_selected.iterrows()):
    ax.text(row['Selection_Count'] + 0.3, i, f"{row['Selection_Count']} ({row['Selection_Rate']})",
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "04_wavelength_selection_frequency.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_wavelength_selection_frequency.png")

# ============================================================================
# PLOT 5: LDA Coefficients for Best Model
# ============================================================================
print("\n[5/8] Generating LDA Coefficients plot...")
wavelengths_df = pd.read_csv(results_dir / "best_model_wavelengths.csv")
wavelengths_df['Wavelength'] = wavelengths_df['Wavelength_Name'].apply(band_to_wavelength)
wavelengths_df['Wavelength_Label'] = wavelengths_df['Wavelength'].apply(format_wavelength)
wavelengths_df = wavelengths_df.sort_values('Abs_Coefficient', ascending=True)

fig, ax = plt.subplots(figsize=(12, 12))
colors = ['crimson' if x < 0 else 'forestgreen' for x in wavelengths_df['LDA_Coefficient']]
bars = ax.barh(range(len(wavelengths_df)), wavelengths_df['LDA_Coefficient'], color=colors, alpha=0.75)

ax.set_yticks(range(len(wavelengths_df)))
ax.set_yticklabels(wavelengths_df['Wavelength_Label'], fontsize=11)
ax.set_xlabel('LDA Coefficient Value', fontweight='bold', fontsize=13)
ax.set_ylabel('Wavelength', fontweight='bold', fontsize=13)
ax.set_title('LDA Coefficients for CRACK Class Detection\n(Sorted by Absolute Importance)',
             fontweight='bold', pad=20, fontsize=14)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
legend_elements = [
    Patch(facecolor='forestgreen', alpha=0.75, label='Positive (↑ CRACK probability)'),
    Patch(facecolor='crimson', alpha=0.75, label='Negative (↓ CRACK probability)')
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=11)

# Add coefficient values
for i, (idx, row) in enumerate(wavelengths_df.iterrows()):
    x_pos = row['LDA_Coefficient']
    offset = 5 if x_pos > 0 else -5
    ha = 'left' if x_pos > 0 else 'right'
    ax.text(x_pos + offset, i, f"{x_pos:.1f}", va='center', ha=ha, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "05_lda_coefficients_best_model.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_lda_coefficients_best_model.png")

# ============================================================================
# PLOT 6: Combined Feature Importance (Frequency + Coefficients)
# ============================================================================
print("\n[6/8] Generating Combined Feature Importance plot...")

# Merge frequency and coefficient data
best_features = wavelengths_df['Wavelength_Name'].values
freq_dict = dict(zip(freq_df['Feature'], freq_df['Selection_Count']))
wavelengths_df['Frequency'] = wavelengths_df['Wavelength_Name'].map(freq_dict)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Left: Coefficients
colors = ['crimson' if x < 0 else 'forestgreen' for x in wavelengths_df['LDA_Coefficient']]
ax1.barh(range(len(wavelengths_df)), wavelengths_df['LDA_Coefficient'], color=colors, alpha=0.75)
ax1.set_yticks(range(len(wavelengths_df)))
ax1.set_yticklabels(wavelengths_df['Wavelength_Label'], fontsize=10)
ax1.set_xlabel('LDA Coefficient', fontweight='bold')
ax1.set_title('(A) Feature Contribution to CRACK Detection', fontweight='bold', pad=15)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# Right: Selection Frequency
freq_colors = plt.cm.YlOrRd(wavelengths_df['Frequency'] / wavelengths_df['Frequency'].max())
ax2.barh(range(len(wavelengths_df)), wavelengths_df['Frequency'], color=freq_colors, alpha=0.8)
ax2.set_yticks(range(len(wavelengths_df)))
ax2.set_yticklabels(wavelengths_df['Wavelength_Label'], fontsize=10)
ax2.set_xlabel('Selection Count (out of 20)', fontweight='bold')
ax2.set_title('(B) Feature Selection Stability', fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Comprehensive Feature Importance Analysis for Best Model',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_dir / "06_combined_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_combined_feature_importance.png")

# ============================================================================
# PLOT 7: Performance Comparison (Global vs CRACK-specific)
# ============================================================================
print("\n[7/8] Generating Performance Comparison plot...")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot global metrics
ax.plot(log_df['num_features'], log_df['accuracy'], 's-', linewidth=2.5, markersize=7,
        label='Global Accuracy', color='steelblue')
ax.plot(log_df['num_features'], log_df['f1_weighted'], 'd-', linewidth=2.5, markersize=7,
        label='Weighted F1 (all classes)', color='navy', alpha=0.7)

# Plot CRACK-specific metrics
ax.plot(log_df['num_features'], log_df['crack_f1'], 'o-', linewidth=3, markersize=8,
        label='CRACK F1 Score ⭐', color='crimson')
ax.plot(log_df['num_features'], log_df['crack_auc'], '^-', linewidth=2.5, markersize=7,
        label='CRACK AUC', color='teal')

# Mark optimal point
ax.axvline(x=best_n, color='red', linestyle='--', alpha=0.5, linewidth=2.5,
           label=f'Optimal: {int(best_n)} wavelengths')
ax.plot(best_n, best_f1, 'r*', markersize=25, markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('Number of Selected Wavelengths', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Global vs. CRACK-Specific Performance Metrics\nOptimization Target: CRACK F1-Score',
             fontweight='bold', pad=20)
ax.legend(loc='lower right', framealpha=0.95, fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(output_dir / "07_global_vs_crack_performance.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_global_vs_crack_performance.png")

# ============================================================================
# PLOT 8: Feature Evolution Heatmap
# ============================================================================
print("\n[8/8] Generating Feature Evolution Heatmap...")

# Extract selected features at each step
feature_matrix = []
all_features_set = set()

for idx, row in log_df.iterrows():
    features_str = row['selected_features']
    features_list = ast.literal_eval(features_str)
    all_features_set.update(features_list)
    feature_matrix.append(features_list)

# Get only the features that appear in the best model
best_features_str = log_df.loc[best_idx, 'selected_features']
best_features_list = ast.literal_eval(best_features_str)

# Create binary matrix
sorted_features = sorted(best_features_list,
                        key=lambda x: wavelengths_df[wavelengths_df['Wavelength_Name']==x]['Abs_Coefficient'].values[0],
                        reverse=True)

matrix = np.zeros((len(log_df), len(sorted_features)))
for i, features in enumerate(feature_matrix):
    for j, feature in enumerate(sorted_features):
        if feature in features:
            matrix[i, j] = 1

# Convert to wavelengths
feature_labels = [format_wavelength(band_to_wavelength(f)) for f in sorted_features]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')

ax.set_xticks(range(len(log_df)))
ax.set_xticklabels(log_df['num_features'], fontsize=10)
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(feature_labels, fontsize=9)

ax.set_xlabel('Number of Selected Wavelengths (SFS Step)', fontweight='bold', fontsize=12)
ax.set_ylabel('Wavelength', fontweight='bold', fontsize=12)
ax.set_title('Sequential Feature Selection Evolution\n(Features sorted by importance in final model)',
             fontweight='bold', pad=20, fontsize=14)

# Add vertical line at best model
ax.axvline(x=best_idx, color='blue', linestyle='--', linewidth=3, alpha=0.7)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Selected', fontweight='bold', fontsize=11)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Not Selected', 'Selected'])

plt.tight_layout()
plt.savefig(output_dir / "08_feature_evolution_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_feature_evolution_heatmap.png")

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUMMARY REPORT...")
print("="*80)

summary_text = f"""
COMPREHENSIVE PLOT GENERATION SUMMARY
{'='*80}

Output Directory: {output_dir}

Generated Plots:
1. 01_sfs_performance_all_metrics.png
   - Complete view of all performance metrics during SFS
   - Shows CRACK F1, F2, Precision, Recall, AUC, and Overall Accuracy
   
2. 02_crack_metrics_evolution.png
   - Detailed evolution of CRACK-specific metrics
   - Two-panel plot: F1/F2 scores and Precision/Recall
   
3. 03_threshold_optimization.png
   - Threshold optimization curve showing F1, Precision, and Recall
   - Marks optimal threshold that maximizes CRACK F1-Score
   
4. 04_wavelength_selection_frequency.png
   - Bar chart showing how often each wavelength was selected
   - Color-coded by frequency (red to green gradient)
   
5. 05_lda_coefficients_best_model.png
   - LDA coefficients for each wavelength in the final model
   - Green = positive contribution, Red = negative contribution
   
6. 06_combined_feature_importance.png
   - Two-panel comparison: Coefficients vs. Selection Frequency
   - Comprehensive view of feature importance and stability
   
7. 07_global_vs_crack_performance.png
   - Comparison of global metrics vs. CRACK-specific metrics
   - Shows the optimization focuses on CRACK class
   
8. 08_feature_evolution_heatmap.png
   - Heatmap showing which features were selected at each SFS step
   - Visual timeline of feature selection process

Best Model Statistics:
- Number of wavelengths: {int(best_n)}
- CRACK F1-Score: {best_f1:.4f}
- CRACK F2-Score: {log_df.loc[best_idx, 'crack_f2']:.4f}
- Overall Accuracy: {log_df.loc[best_idx, 'accuracy']:.4f}
- CRACK Precision: {log_df.loc[best_idx, 'crack_precision']:.4f}
- CRACK Recall: {log_df.loc[best_idx, 'crack_recall']:.4f}
- CRACK AUC: {log_df.loc[best_idx, 'crack_auc']:.4f}
- Optimal Threshold: {best_threshold:.3f}

Selected Wavelengths (sorted by importance):
"""

for idx, row in wavelengths_df.iterrows():
    direction = "↑" if row['LDA_Coefficient'] > 0 else "↓"
    summary_text += f"  {direction} {row['Wavelength_Label'].ljust(10)} (coef: {row['LDA_Coefficient']:7.2f})\n"

summary_text += f"\n{'='*80}\n"
summary_text += "All plots have been successfully generated with wavelength labels!\n"
summary_text += "These plots are thesis-ready with high resolution (300 DPI).\n"
summary_text += f"{'='*80}\n"

# Save summary
with open(output_dir / "generation_summary.txt", "w", encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)
print(f"\nAll plots generated successfully!")
print(f"Output directory: {output_dir}")
print("="*80)

