"""
Script to regenerate plots with wavelengths instead of band numbers
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import joblib
from wavelengths import WAVELENGTHS

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

# ============================================================================
# 1. Feature Selection Frequency Plot
# ============================================================================
print("Generating Feature Selection Frequency plot...")
freq_df = pd.read_csv(results_dir / "feature_selection_frequency.csv")

# Convert band numbers to wavelengths
freq_df['Wavelength'] = freq_df['Feature'].apply(band_to_wavelength)
freq_df['Wavelength_Label'] = freq_df['Wavelength'].apply(format_wavelength)

# Sort by selection count
freq_df = freq_df.sort_values('Selection_Count', ascending=True)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(freq_df)), freq_df['Selection_Count'], color='steelblue', alpha=0.8)

# Color gradient based on frequency
colors = plt.cm.RdYlGn(freq_df['Selection_Count'] / freq_df['Selection_Count'].max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(range(len(freq_df)))
ax.set_yticklabels(freq_df['Wavelength_Label'], fontsize=10)
ax.set_xlabel('Selection Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Wavelength', fontsize=12, fontweight='bold')
ax.set_title('Feature Selection Frequency Across CV Folds\n(Wavelengths)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(freq_df.iterrows()):
    ax.text(row['Selection_Count'] + 0.3, i, f"{row['Selection_Count']} ({row['Selection_Rate']})",
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "feature_selection_frequency_wavelengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: feature_selection_frequency_wavelengths.png")

# ============================================================================
# 2. LDA Coefficients Plot
# ============================================================================
print("Generating LDA Coefficients plot...")
wavelengths_df = pd.read_csv(results_dir / "best_model_wavelengths.csv")

# Convert band names to wavelengths
wavelengths_df['Wavelength'] = wavelengths_df['Wavelength_Name'].apply(band_to_wavelength)
wavelengths_df['Wavelength_Label'] = wavelengths_df['Wavelength'].apply(format_wavelength)

# Sort by absolute coefficient
wavelengths_df = wavelengths_df.sort_values('Abs_Coefficient', ascending=True)

# Create plot
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['red' if x < 0 else 'green' for x in wavelengths_df['LDA_Coefficient']]
bars = ax.barh(range(len(wavelengths_df)), wavelengths_df['LDA_Coefficient'], color=colors, alpha=0.7)

ax.set_yticks(range(len(wavelengths_df)))
ax.set_yticklabels(wavelengths_df['Wavelength_Label'], fontsize=10)
ax.set_xlabel('LDA Coefficient Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Wavelength', fontsize=12, fontweight='bold')
ax.set_title('LDA Coefficients for CRACK Class\n(Sorted by Absolute Value)',
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Positive (increases crack probability)'),
    Patch(facecolor='red', alpha=0.7, label='Negative (decreases crack probability)')
]
ax.legend(handles=legend_elements, loc='best', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "lda_coefficients_crack_class_wavelengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: lda_coefficients_crack_class_wavelengths.png")

# ============================================================================
# 3. Selected Features Importance Plot (with wavelengths)
# ============================================================================
print("Generating Selected Features Importance plot...")

# Create plot combining both metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Left plot: Absolute LDA Coefficients
wavelengths_sorted = wavelengths_df.sort_values('Abs_Coefficient', ascending=False)
bars1 = ax1.barh(range(len(wavelengths_sorted)), wavelengths_sorted['Abs_Coefficient'],
                  color='steelblue', alpha=0.8)
colors1 = plt.cm.YlOrRd(wavelengths_sorted['Abs_Coefficient'] / wavelengths_sorted['Abs_Coefficient'].max())
for bar, color in zip(bars1, colors1):
    bar.set_color(color)

ax1.set_yticks(range(len(wavelengths_sorted)))
ax1.set_yticklabels(wavelengths_sorted['Wavelength_Label'], fontsize=9)
ax1.set_xlabel('Absolute LDA Coefficient', fontsize=11, fontweight='bold')
ax1.set_ylabel('Wavelength', fontsize=11, fontweight='bold')
ax1.set_title('Feature Importance: LDA Coefficients', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Right plot: Selection Frequency
# Merge with frequency data
freq_lookup = dict(zip(freq_df['Feature'], freq_df['Selection_Count']))
wavelengths_sorted['Selection_Count'] = wavelengths_sorted['Wavelength_Name'].map(freq_lookup).fillna(0)
wavelengths_sorted_freq = wavelengths_sorted.sort_values('Selection_Count', ascending=False)

bars2 = ax2.barh(range(len(wavelengths_sorted_freq)), wavelengths_sorted_freq['Selection_Count'],
                  color='coral', alpha=0.8)
colors2 = plt.cm.RdYlGn(wavelengths_sorted_freq['Selection_Count'] / wavelengths_sorted_freq['Selection_Count'].max())
for bar, color in zip(bars2, colors2):
    bar.set_color(color)

ax2.set_yticks(range(len(wavelengths_sorted_freq)))
ax2.set_yticklabels(wavelengths_sorted_freq['Wavelength_Label'], fontsize=9)
ax2.set_xlabel('Selection Frequency (out of 20 folds)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Wavelength', fontsize=11, fontweight='bold')
ax2.set_title('Feature Importance: Selection Frequency', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.suptitle('Selected Features Importance Analysis\n(Wavelengths)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / "selected_features_importance_wavelengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: selected_features_importance_wavelengths.png")

# ============================================================================
# 4. SFS Performance Enhanced Plot
# ============================================================================
print("Generating SFS Performance Enhanced plot...")
sfs_log = pd.read_csv(results_dir / "feature_selection_log_crack_f1.csv")

# Parse selected features
def parse_features(feature_str):
    if isinstance(feature_str, str):
        try:
            return ast.literal_eval(feature_str)
        except:
            return []
    return []

sfs_log['selected_features_list'] = sfs_log['selected_features'].apply(parse_features)

# Create wavelength labels for x-axis
def create_wavelength_labels(features_list):
    if not features_list:
        return ""
    wavelengths = [band_to_wavelength(f) for f in features_list]
    if len(wavelengths) <= 3:
        return ", ".join([format_wavelength(w) for w in wavelengths])
    else:
        return f"{len(wavelengths)} wavelengths"

sfs_log['x_label'] = sfs_log['selected_features_list'].apply(create_wavelength_labels)

# Create the plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Crack F1 and F2 scores
ax1.plot(sfs_log['num_features'], sfs_log['crack_f1'], 'o-', label='Crack F1',
         linewidth=2, markersize=8, color='red')
ax1.plot(sfs_log['num_features'], sfs_log['crack_f2'], 's-', label='Crack F2',
         linewidth=2, markersize=7, color='darkred')
ax1.set_xlabel('Number of Features (Wavelengths)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('CRACK Class Performance', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Overall Performance Metrics
ax2.plot(sfs_log['num_features'], sfs_log['accuracy'], 'o-', label='Accuracy',
         linewidth=2, markersize=8)
ax2.plot(sfs_log['num_features'], sfs_log['f1_weighted'], 's-', label='F1 Weighted',
         linewidth=2, markersize=7)
ax2.plot(sfs_log['num_features'], sfs_log['precision_weighted'], '^-', label='Precision Weighted',
         linewidth=2, markersize=7)
ax2.plot(sfs_log['num_features'], sfs_log['recall_weighted'], 'v-', label='Recall Weighted',
         linewidth=2, markersize=7)
ax2.set_xlabel('Number of Features (Wavelengths)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Overall Model Performance', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: AUC Scores
ax3.plot(sfs_log['num_features'], sfs_log['crack_auc'], 'o-', label='Crack AUC',
         linewidth=2, markersize=8, color='purple')
ax3.plot(sfs_log['num_features'], sfs_log['roc_auc_weighted'], 's-', label='Weighted AUC',
         linewidth=2, markersize=7, color='blue')
ax3.set_xlabel('Number of Features (Wavelengths)', fontsize=11, fontweight='bold')
ax3.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
ax3.set_title('ROC AUC Performance', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Crack-specific metrics
ax4.plot(sfs_log['num_features'], sfs_log['crack_precision'], 'o-', label='Crack Precision',
         linewidth=2, markersize=8, color='green')
ax4.plot(sfs_log['num_features'], sfs_log['crack_recall'], 's-', label='Crack Recall',
         linewidth=2, markersize=7, color='orange')
ax4.set_xlabel('Number of Features (Wavelengths)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('CRACK Class Precision & Recall', fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('Sequential Feature Selection Performance\n(Wavelength-based Features)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / "sfs_performance_enhanced_wavelengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: sfs_performance_enhanced_wavelengths.png")

# ============================================================================
# 5. Feature Evolution Plot (showing which wavelengths were added)
# ============================================================================
print("Generating Feature Evolution plot...")

# Create a plot showing which features were added at each step
fig, ax = plt.subplots(figsize=(14, 10))

# Prepare data
max_features = sfs_log['num_features'].max()
feature_matrix = np.zeros((max_features, max_features))

for idx, row in sfs_log.iterrows():
    num_feat = row['num_features']
    features = row['selected_features_list']
    for feat in features:
        # Get wavelength
        wl = band_to_wavelength(feat)
        # Find the position based on first appearance
        for i, r in sfs_log.iterrows():
            if feat in r['selected_features_list']:
                feat_idx = list(sfs_log.loc[i, 'selected_features_list']).index(feat)
                break
        feature_matrix[i, num_feat-1] = 1

# Get unique wavelengths in order of selection
all_wavelengths_ordered = []
for idx, row in sfs_log.iterrows():
    features = row['selected_features_list']
    for feat in features:
        wl = band_to_wavelength(feat)
        if wl not in [w[0] for w in all_wavelengths_ordered]:
            all_wavelengths_ordered.append((wl, feat))

# Create heatmap showing progression
progression_matrix = []
for step_idx, row in sfs_log.iterrows():
    step_features = row['selected_features_list']
    step_row = []
    for wl, feat in all_wavelengths_ordered:
        if feat in step_features:
            step_row.append(1)
        else:
            step_row.append(0)
    progression_matrix.append(step_row)

progression_matrix = np.array(progression_matrix).T

# Plot
im = ax.imshow(progression_matrix, cmap='YlGn', aspect='auto', interpolation='nearest')
ax.set_xlabel('Number of Features Selected', fontsize=12, fontweight='bold')
ax.set_ylabel('Wavelength', fontsize=12, fontweight='bold')
ax.set_title('Feature Selection Progression\n(Which wavelengths were selected at each step)',
             fontsize=14, fontweight='bold', pad=20)

# Set ticks
ax.set_xticks(range(len(sfs_log)))
ax.set_xticklabels(range(1, len(sfs_log)+1), fontsize=9)
ax.set_yticks(range(len(all_wavelengths_ordered)))
ax.set_yticklabels([format_wavelength(wl) for wl, _ in all_wavelengths_ordered], fontsize=9)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Selected', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "feature_evolution_wavelengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: feature_evolution_wavelengths.png")

# ============================================================================
# 6. Detailed Feature Selection Steps with Wavelengths
# ============================================================================
print("Generating Detailed Feature Selection Steps plot...")

fig, ax = plt.subplots(figsize=(16, 12))

# Create a text-based visualization
y_pos = len(sfs_log)
for idx, row in sfs_log.iterrows():
    num_feat = row['num_features']
    features = row['selected_features_list']

    # Convert to wavelengths
    wavelengths = [band_to_wavelength(f) for f in features]
    wl_labels = [format_wavelength(w) for w in wavelengths]

    # Get the newly added feature
    if idx > 0:
        prev_features = sfs_log.iloc[idx-1]['selected_features_list']
        new_feature = [f for f in features if f not in prev_features]
        if new_feature:
            new_wl = band_to_wavelength(new_feature[0])
            new_label = f" +{format_wavelength(new_wl)}"
        else:
            new_label = ""
    else:
        new_label = f" {format_wavelength(wavelengths[0])}"

    # Create label
    label = f"Step {num_feat:2d}: " + new_label

    # Color based on F1 score
    color = plt.cm.RdYlGn(row['crack_f1'])

    # Plot bar
    bar = ax.barh(y_pos, row['crack_f1'], color=color, alpha=0.8, height=0.7)

    # Add label
    ax.text(0.01, y_pos, label, va='center', fontsize=9, fontweight='bold')

    # Add F1 score
    ax.text(row['crack_f1'] + 0.01, y_pos, f"F1={row['crack_f1']:.4f}",
            va='center', fontsize=8)

    y_pos -= 1

ax.set_xlabel('Crack F1 Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Selection Step', fontsize=12, fontweight='bold')
ax.set_title('Sequential Feature Selection: Step-by-Step Performance\n(Showing newly added wavelength at each step)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_yticks([])
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 1.0)

plt.tight_layout()
plt.savefig(output_dir / "sfs_detailed_steps_wavelengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: sfs_detailed_steps_wavelengths.png")

# ============================================================================
# 7. Summary Report
# ============================================================================
print("\nGenerating summary report...")

summary_text = f"""
WAVELENGTH-BASED PLOT GENERATION SUMMARY
{'='*80}

Output Directory: {output_dir}

Generated Plots:
1. feature_selection_frequency_wavelengths.png
   - Shows how often each wavelength was selected across CV folds
   
2. lda_coefficients_crack_class_wavelengths.png
   - LDA coefficients for each wavelength in the final model
   - Green = positive (increases crack probability)
   - Red = negative (decreases crack probability)
   
3. selected_features_importance_wavelengths.png
   - Combined view of LDA coefficients and selection frequency
   
4. sfs_performance_enhanced_wavelengths.png
   - Performance metrics across different numbers of wavelengths
   - Shows Crack F1/F2, Accuracy, AUC, Precision, and Recall
   
5. feature_evolution_wavelengths.png
   - Heatmap showing which wavelengths were selected at each step
   
6. sfs_detailed_steps_wavelengths.png
   - Step-by-step view of feature selection with newly added wavelengths

Final Model Statistics:
- Number of wavelengths: {len(wavelengths_df)}
- Best Crack F1 Score: {sfs_log['crack_f1'].max():.4f}
- Best Crack F2 Score: {sfs_log['crack_f2'].max():.4f}
- Final Accuracy: {sfs_log.iloc[-1]['accuracy']:.4f}

Selected Wavelengths (in order of importance):
"""

for idx, row in wavelengths_df.sort_values('Abs_Coefficient', ascending=False).iterrows():
    wl = row['Wavelength']
    coef = row['LDA_Coefficient']
    direction = "↑" if coef > 0 else "↓"
    summary_text += f"\n  {direction} {format_wavelength(wl)} (coef: {coef:7.2f})"

summary_text += f"\n\n{'='*80}\n"
summary_text += "All plots have been successfully generated with wavelength labels!\n"

# Save summary
with open(output_dir / "generation_summary.txt", "w", encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)
print(f"\nAll plots generated successfully!")
print(f"Output directory: {output_dir}")

