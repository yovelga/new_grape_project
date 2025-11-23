@echo off
REM Quick Start Script for HSI Crack Detection UI
REM This script launches the late detection UI with proper paths

echo ========================================
echo HSI Crack Detection - Late Detection UI
echo ========================================
echo.
echo Project: C:\Users\yovel\Desktop\Grape_Project
echo Model: OLD LDA [1=CRACK, 0=regular]
echo.
echo Starting UI...
echo.

cd /d "C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model"
"C:\Users\yovel\Desktop\Grape_Project\.venv\Scripts\python.exe" late_detection_ui.py

pause
