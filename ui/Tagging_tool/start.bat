@echo off
REM Startup script for Tagging Tool
echo ============================================================
echo Starting Tagging Tool
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "app.py" (
    echo Error: app.py not found. Please run this script from the Tagging_tool directory.
    pause
    exit /b 1
)

REM Run validation
echo Running setup validation...
python validate_setup.py
if errorlevel 1 (
    echo.
    echo Validation failed. Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Starting Streamlit application...
echo ============================================================
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

