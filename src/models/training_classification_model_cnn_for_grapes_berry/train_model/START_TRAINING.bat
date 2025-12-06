@echo off
REM Quick launcher for training with context-aware square crops
REM Run this batch file to start training

echo ============================================================
echo STARTING TRAINING WITH CONTEXT-AWARE SQUARE CROPS
echo ============================================================
echo.
echo Mode: context_square_segmentation
echo Features:
echo   - Square crops (no distortion)
echo   - 40%% background context
echo   - Segmentation mask applied
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python main.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Training failed!
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo TRAINING COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Check the 'new_models_YYYYMMDD_HHMMSS' folder for results.
echo.
pause

