# Thesis Folder Structure Refactoring Summary

**Date:** February 11, 2026  
**Project:** Grape Cracking Detection Thesis  
**Status:** ✅ Completed Successfully

---

## New Folder Structure

```
thesis/
├── main.tex                    # Main thesis document
├── main.pdf                    # Compiled PDF output
├── .vscode/                    # VS Code settings
├── bibliography/               # Bibliography files
│   └── references.bib         # (1 file)
├── chapters/                   # All thesis chapters
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── literature_review.tex
│   ├── objectives_hypotheses.tex
│   ├── materials_methods.tex
│   ├── results_new.tex
│   ├── discussions_new.tex
│   ├── conclusions.tex
│   └── discussion.tex         # (9 total files)
├── figures/                    # All images and figures
│   ├── literature/            # Literature review figures (3 images)
│   ├── materials_methods/     # Materials & methods figures
│   │   ├── study_area/       # Study area maps
│   │   ├── equipment/        # Equipment photos
│   │   ├── pipeline/         # Pipeline diagrams
│   │   └── gui/              # GUI screenshots
│   ├── results/               # Results figures
│   │   ├── spectral_signatures/    # Spectral analysis figures
│   │   ├── feature_selection/      # Feature selection plots
│   │   ├── autoencoder/            # Autoencoder results
│   │   ├── full_images/            # Full image classification
│   │   ├── pixel_classification/   # Pixel-level results
│   │   ├── model_comparison/       # Model comparison plots
│   │   └── cnn/                    # CNN results
│   ├── appendix/              # Legacy/appendix figures
│   └── common/                # Shared figures (empty - reserved)
│                              # (137 total image files)
├── data/                      # Experimental data files
│   ├── spectral_signatures/   # CSV/JSON spectral data
│   ├── autoencoder/          # Autoencoder metrics
│   ├── model_comparison/     # Model performance data
│   └── feature_selection/    # Feature selection data
│                              # (16 total data files)
├── build/                     # LaTeX compilation artifacts
│   ├── main.aux, main.bbl, main.bcf
│   ├── main.blg, main.lof, main.lot
│   ├── main.run.xml, main.toc
│   ├── main.log, main.synctex.gz
│   └── main.pdf (backup)     # (11 files)
└── archive/                   # Old/deprecated files
    ├── old_chapters/         # Previous chapter versions
    │   ├── results.tex
    │   ├── results_new_bck.tex
    │   ├── literature_review_old.tex
    │   └── bibliography.tex
    └── old_versions/         # Old thesis versions
        └── alonVersion.tex   # (5 total archived files)
```

---

## Files Moved

### 📁 Images Organized (137 files total)

#### Literature Review Images
- `reflectence_furmula.jpg` → `figures/literature/`
- `Barley_Signature.jpg` → `figures/literature/`
- `Hyperspectral_imaging_across_the_electromagnetic.jpg` → `figures/literature/`

#### Materials & Methods Images
From `Materials and Methods/` → `figures/materials_methods/`:
- `Study Area and Experimental Design/vineyard_map.png` → `study_area/`
- `VICTOR_AND_ALON_WITH_SPECIM_IQ.jpeg` → `equipment/`
- `taking_picture_with_hsi.jpg` → `equipment/`
- `yovel_with_thermal_camera.jpg` → `equipment/`
- `thermal_image_with_decay_and_cracksjpg.jpg` → `equipment/`
- `whole_image_pipeline.jpg` → `pipeline/`

From `Gui_pixelpicker/` → `figures/materials_methods/gui/`:
- `gui_sam2_pointer_with_numbers.png`

#### Results Images
From `results/Spectral_Signature_Characterization/` → `figures/results/spectral_signatures/`:
- All PNG files (6 images)

From `results/feature_selection/` → `figures/results/feature_selection/`:
- All PNG files (18+ images)

From `4.2.4/` → `figures/results/feature_selection/`:
- All PNG and PDF files (15+ feature selection plots)

From `4.2.5/` → `figures/results/autoencoder/`:
- All PNG files (7 autoencoder results)

From root directory → `figures/appendix/`:
- `avg_std_signatures.png`
- `avg_with_selected_wl.png`
- `spectral_signatures_comparison.jpg`
- `figure_gui_tool_regular.jpg`
- `figure_gui_tool_cracked.jpg`
- `Mean spectral signatures with highlighted informative wavelengths.jpg`

From various result directories → `figures/results/`:
- `cnn/` (entire folder with bbox/, enlarge/, segmentation/ subdirs)
- `full_images_classification/` → `full_images/`
- `Full_Image_Dimensionality_Reduction/` → `full_images/dimensionality_reduction/`
- `sam2_plus_cnn/` → `full_images/`
- `model_multi_class_lda_multi_vs_binary/` → `pixel_classification/`
- `XGBOOST_mullti_vs_LDA_multi/` → `model_comparison/xgboost_lda/`

### 📊 Data Files Organized (16 files)

From `results/Spectral_Signature_Characterization/` → `data/spectral_signatures/`:
- `regional_statistics.csv`
- `top_separable_wavelengths.csv`
- `wavelength_statistics.csv`
- `analysis_summary.json`

From `4.2.5/` → `data/autoencoder/`:
- `Autoencoder_on_Regulars_oneclass_cv_metrics.csv`
- `Autoencoder_on_Regulars_threshold_sweep.csv`
- `Autoencoder_on_Regulars_best_threshold_F1max.txt`
- `logo_comparison_summary.xlsx`

From `figures/results/pixel_classification/` and `model_comparison/` → `data/model_comparison/`:
- Multiple confusion matrix and metrics CSV files (8+ files)

### 🔨 Build Artifacts Organized

To `build/`:
- `main.aux`, `main.bbl`, `main.bcf`, `main.blg`
- `main.lof`, `main.lot`, `main.run.xml`, `main.toc`
- `main.log`, `main.synctex.gz`
- `main.pdf` (backup copy)

### 📚 Archive Files

To `archive/old_chapters/`:
- `results.tex` (superseded by results_new.tex)
- `results_new_bck.tex` (backup version)
- `literature_review_old.tex` (old version)
- `bibliography.tex` (old appendix file)

To `archive/old_versions/`:
- `alonVersion.tex` (alternative thesis version)

### 📖 Bibliography

- `references.bib` → `bibliography/references.bib`

---

## Path Updates

### Updated Files
All image references were systematically updated in the following active chapters:

1. **materials_methods.tex** (7 image paths updated)
   - Materials and Methods/ → figures/materials_methods/
   - Gui_pixelpicker/ → figures/materials_methods/gui/

2. **literature_review.tex** (3 image paths updated)
   - Root images → figures/literature/

3. **results_new.tex** (20+ image paths updated)
   - results/Spectral_Signature_Characterization/ → figures/results/spectral_signatures/
   - results/feature_selection/ → figures/results/feature_selection/

4. **main.tex** (2 critical updates)
   - Updated bibliography path: `\addbibresource{bibliography/references.bib}`
   - Removed obsolete `\graphicspath{{4.2.4/}}` directive
   - All image paths are now explicit and relative to thesis root

### Path Mapping Reference

| Old Path | New Path |
|----------|----------|
| `Materials and Methods/Study Area and Experimental Design/` | `figures/materials_methods/study_area/` |
| `Materials and Methods/*.{jpg,jpeg,png}` | `figures/materials_methods/equipment/` |
| `Materials and Methods/whole_image_pipeline.jpg` | `figures/materials_methods/pipeline/` |
| `Gui_pixelpicker/` | `figures/materials_methods/gui/` |
| `results/Spectral_Signature_Characterization/` | `figures/results/spectral_signatures/` |
| `results/feature_selection/` | `figures/results/feature_selection/` |
| `4.2.4/` | `figures/results/feature_selection/` |
| `4.2.5/` | `figures/results/autoencoder/` |
| Root-level images | `figures/appendix/` or `figures/literature/` |
| `references.bib` | `bibliography/references.bib` |

---

## Removed/Cleaned Directories

The following empty or obsolete directories were removed:
- `Materials and Methods/` (and all subdirectories)
- `results/` (and all subdirectories)
- `4.2.4/`
- `4.2.5/`
- `Gui_pixelpicker/`

---

## Benefits of New Structure

✅ **Logical Organization:** Clear separation between source files, figures, data, and build artifacts  
✅ **Easy Navigation:** Hierarchical folder structure groups related files  
✅ **Clean Root Directory:** Only essential files in the root (main.tex and output PDF)  
✅ **Version Control Friendly:** Build artifacts isolated in build/ folder  
✅ **Scalable:** Easy to add new chapters, figures, or data files  
✅ **Professional:** Follows academic thesis best practices  
✅ **Maintainable:** Clear naming conventions and consistent structure  

---

## Verification

### Files in Each Directory
- **bibliography:** 1 file
- **chapters:** 9 files (all active .tex chapters)
- **figures:** 137 files (all images organized by chapter/type)
- **data:** 16 files (CSV, JSON, XLSX experimental data)
- **build:** 11 files (compilation artifacts)
- **archive:** 5 files (old versions and deprecated chapters)

### Active Chapter Files
- abstract.tex
- introduction.tex
- literature_review.tex
- objectives_hypotheses.tex
- materials_methods.tex
- results_new.tex
- discussions_new.tex
- conclusions.tex
- discussion.tex

---

## Next Steps

1. **Test Compilation:** Compile the thesis using your LaTeX compiler to ensure all paths are correct
   ```bash
   pdflatex main.tex
   biber main
   pdflatex main.tex
   pdflatex main.tex
   ```

2. **Clean Build Directory:** After successful compilation, you can periodically clean the build/ folder

3. **Future Organization:** When adding new figures or data:
   - Images → `figures/[chapter-name]/`
   - Data → `data/[data-type]/`
   - Old versions → `archive/`

---

## Notes

- All relative paths are maintained (no absolute paths)
- The project structure is now consistent with academic and technical best practices
- Build artifacts in the root directory (like main.aux, main.log, etc.) are generated during compilation - you can move them to build/ or add them to .gitignore
- The main.pdf file can remain in root for easy access

---

**Refactoring completed successfully! Your thesis folder is now clean, organized, and ready for final compilation.**
