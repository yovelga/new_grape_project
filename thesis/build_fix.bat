@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis

echo Clearing biber cache...
rmdir /s /q "%TEMP%\par-*" 2>nul
for /d %%i in ("%TEMP%\biber_cache*") do rmdir /s /q "%%i" 2>nul
for /d %%i in ("%TEMP%\par-*") do rmdir /s /q "%%i" 2>nul

echo Cleaning aux files...
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul

echo Pass 1...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
if exist main.bcf (echo BCF OK) else (echo BCF MISSING - STOP && goto :end)

echo Biber...
biber --input-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" --output-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" main 2>&1
if exist main.bbl (echo BBL OK) else (echo BBL MISSING)

echo Pass 2...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1

echo Pass 3...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1

echo.
echo === ERRORS ===
findstr "^!" main.log
echo.
echo === OUTPUT ===
findstr "Output written" main.log

:end
