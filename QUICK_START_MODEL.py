"""
Quick Start Guide: LDA Best F1 Weighted Model
==============================================

The LDA Best F1 Weighted model has been successfully connected to the late detection UI!

WHAT'S NEW:
-----------
* New model added: "LDA Best F1 Weighted"
* Automatic feature selection (204 -> 20 bands)
* Optimized for balanced multi-class performance
* Set as default model option

HOW TO USE:
-----------
1. Navigate to the model directory:
   cd src/models/classification/full_image/infernce_with_new_model_with_sam2_reduce_bands

2. Launch the UI:
   python late_detection_ui.py

3. In the UI:
   - Select "LDA Best F1 Weighted" from the Model dropdown (should be selected by default)
   - Load your hyperspectral images
   - The model will automatically apply feature selection
   - View CRACK detection results

TECHNICAL DETAILS:
------------------
Model Type: LDA with Sequential Feature Selection
Input: 204 hyperspectral bands
Selected Features: 20 bands (automatically applied)
Classes: BACKGROUND, BRANCH, CRACK, PLASTIC, REGULAR
Target: CRACK detection (index 2)
Optimization: Weighted F1-Score

Selected Bands:
- band_0, band_13, band_21, band_44, band_57
- band_61, band_63, band_74, band_90, band_97
- band_106, band_116, band_128, band_136, band_154
- band_155, band_177, band_180, band_188, band_201

BATCH PROCESSING:
-----------------
For batch dataset processing, use:
   python run_late_detection_inference.py

The model is available in the AVAILABLE_MODELS dictionary.

TROUBLESHOOTING:
----------------
If you encounter any issues:
1. Run test_model_integration.py to verify the model works
2. Check that the model file exists in the same directory
3. Ensure all dependencies are installed (sklearn, imblearn, joblib)

TESTING:
--------
Run integration tests:
   python test_model_integration.py

Expected output: "ALL TESTS PASSED!"

For more details, see: MODEL_INTEGRATION_SUMMARY.md
"""

print(__doc__)

