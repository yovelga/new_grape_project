from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
"""
THESIS PLOT USAGE GUIDE
========================

This guide helps you choose which plots to use in different sections of your thesis.

INTRODUCTION / MOTIVATION
--------------------------
None needed yet - these are results plots.


METHODOLOGY SECTION
--------------------
Plot: 08_feature_evolution_heatmap.png
Purpose: Explain the Sequential Feature Selection (SFS) methodology
Caption: "Sequential feature selection process showing the evolution of selected
         wavelengths across 20 iterations. Features are sorted by their importance
         in the final model."

Plot: 03_threshold_optimization.png
Purpose: Explain the threshold optimization methodology
Caption: "Optimization of classification threshold to maximize F1-Score for the
         CRACK class. The optimal threshold of 0.620 balances precision and recall."


RESULTS SECTION
----------------

1. Model Performance Overview
   Plot: 01_sfs_performance_all_metrics.png
   Caption: "Performance metrics across different numbers of selected wavelengths.
            The model achieves optimal CRACK F1-Score with 18 wavelengths."

2. CRACK Class Performance Detail
   Plot: 02_crack_metrics_evolution.png
   Caption: "Evolution of CRACK-specific metrics during feature selection. (A) F1
            and F2 scores. (B) Precision and Recall. Vertical dashed line indicates
            the optimal model with 18 wavelengths."

3. Feature Importance
   Plot: 05_lda_coefficients_best_model.png
   Caption: "LDA coefficients for the 18 selected wavelengths in the final model.
            Positive coefficients (green) increase CRACK probability, while negative
            coefficients (red) decrease it. Features are sorted by absolute importance."

4. Feature Stability
   Plot: 04_wavelength_selection_frequency.png
   Caption: "Selection frequency of wavelengths across 20 SFS iterations. Higher
            frequency indicates more stable and important features. Band 132 (783.7nm)
            was selected in all 20 iterations."

5. Comprehensive Feature Analysis
   Plot: 06_combined_feature_importance.png
   Caption: "Comprehensive feature importance analysis. (A) LDA coefficients showing
            the direction and magnitude of each wavelength's contribution. (B) Selection
            frequency indicating feature stability."


DISCUSSION SECTION
-------------------

1. Spectral Interpretation
   Plot: 05_lda_coefficients_best_model.png
   Caption: Same as above
   Discussion Points:
   - Wavelengths around 566nm, 469nm (blue-green region) negatively correlate with cracks
   - Wavelengths around 816nm, 631nm (red-NIR region) positively correlate with cracks
   - Discuss biological/physical reasons (chlorophyll, cell structure, water content)

2. Model Optimization
   Plot: 07_global_vs_crack_performance.png
   Caption: "Comparison of global (overall accuracy, weighted F1) versus CRACK-specific
            metrics. The optimization specifically targets CRACK F1-Score rather than
            global accuracy."
   Discussion Points:
   - Why CRACK-specific optimization is important
   - Trade-offs between classes

3. Dimensionality Reduction
   Plot: 01_sfs_performance_all_metrics.png
   Caption: Same as Results section
   Discussion Points:
   - Reduced from 204 to 18 wavelengths (91% reduction)
   - Minimal performance loss
   - Benefits: faster processing, reduced storage, noise reduction


APPENDIX
---------
Include all remaining plots for completeness:
- feature_evolution_wavelengths.png
- sfs_detailed_steps_wavelengths.png
- sfs_performance_enhanced_wavelengths.png


KEY STATISTICS TO REPORT IN TEXT
---------------------------------
From best_model_report_crack_f1.txt:

Model Performance:
- CRACK F1-Score: 0.8712 (with optimal threshold 0.620)
- CRACK F2-Score: 0.8725
- Overall Accuracy: 95.36%
- CRACK Precision: 90.03%
- CRACK Recall: 84.39%
- CRACK AUC: 98.78%

Feature Selection:
- Original features: 204 wavelengths
- Selected features: 18 wavelengths
- Reduction: 91.2%
- Selection method: Sequential Forward Selection (SFS)
- Optimization target: CRACK F1-Score with 3-fold cross-validation

Data:
- Training set: 80% (balanced via RandomOverSampler)
- Test set: 20% (101,101 samples, unbalanced/real-world distribution)
- Classes: BACKGROUND, BRANCH, CRACK, PLASTIC, REGULAR

Most Important Wavelengths:
1. 566.2nm (band_59) - coefficient: -117.08
2. 469.7nm (band_26) - coefficient: -116.03
3. 513.4nm (band_41) - coefficient: +100.32

Most Stable Features (selection frequency):
1. 783.7nm (band_132) - 100% (20/20)
2. 804.8nm (band_139) - 95% (19/20)
3. 738.5nm (band_117) - 90% (18/20)


FIGURE QUALITY CHECKLIST
--------------------------
✓ All plots are 300 DPI (publication quality)
✓ All axes are labeled clearly
✓ All legends are present and readable
✓ Wavelengths are in nm (not band numbers)
✓ Colors are colorblind-friendly where possible
✓ Font sizes are appropriate for publication
✓ Grid lines improve readability without cluttering


SUGGESTED PLOT COMBINATIONS FOR MULTI-PANEL FIGURES
-----------------------------------------------------

Figure 1: "Model Development and Optimization"
├─ Panel A: 01_sfs_performance_all_metrics.png
└─ Panel B: 03_threshold_optimization.png

Figure 2: "CRACK Class Performance Analysis"
├─ Panel A: 02_crack_metrics_evolution.png (already 2 panels)

Figure 3: "Feature Importance and Stability"
├─ Panel A: 05_lda_coefficients_best_model.png
└─ Panel B: 04_wavelength_selection_frequency.png
(OR use the already combined: 06_combined_feature_importance.png)

Figure 4: "Feature Selection Evolution"
└─ 08_feature_evolution_heatmap.png (standalone)


COMMON THESIS QUESTIONS & WHICH PLOT ANSWERS THEM
---------------------------------------------------

Q: "How did you select the features?"
A: Show 08_feature_evolution_heatmap.png and explain SFS process

Q: "Why 18 wavelengths?"
A: Show 01_sfs_performance_all_metrics.png - performance plateaus after 18

Q: "Which wavelengths are most important?"
A: Show 05_lda_coefficients_best_model.png - sorted by importance

Q: "Are the selected features stable?"
A: Show 04_wavelength_selection_frequency.png - top features selected 90-100% of time

Q: "How did you optimize the threshold?"
A: Show 03_threshold_optimization.png - maximized F1 at threshold 0.620

Q: "How does your model perform on the CRACK class?"
A: Show 02_crack_metrics_evolution.png and report F1=0.8712, Recall=84.39%

Q: "Did you sacrifice overall accuracy for CRACK detection?"
A: Show 07_global_vs_crack_performance.png - no, accuracy remains 95.36%


LATEX CAPTION TEMPLATES
-------------------------

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{01_sfs_performance_all_metrics.png}
\caption{Performance metrics across different numbers of selected wavelengths during
Sequential Feature Selection (SFS). The model achieves optimal CRACK F1-Score of 0.8140
with 18 wavelengths, marked with a star. All metrics show improvement with additional
features until plateauing around 15-18 wavelengths.}
\label{fig:sfs_performance}
\end{figure}

"""

# Save to file
output_path = str(_PROJECT_ROOT / r"src/models/classification/full_image/Train/reduce_chanels/results_multi_class_reduce_CRACK_F1/new_plots/THESIS_PLOT_USAGE_GUIDE.txt")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(__doc__)

print("Thesis Plot Usage Guide created successfully!")
print(f"Location: {output_path}")

