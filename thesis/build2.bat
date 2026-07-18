@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis

rem Clean everything
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul

rem Pass 1
pdflatex -interaction=nonstopmode main.tex > nul 2>&1

rem Check bcf exists
if exist main.bcf (echo BCF exists after pass1) else (echo BCF MISSING after pass1)

rem Biber
biber main > nul 2>&1
if errorlevel 1 (echo BIBER FAILED) else (echo BIBER OK)

rem Check bbl exists
if exist main.bbl (echo BBL exists after biber) else (echo BBL MISSING after biber)

rem Pass 2
pdflatex -interaction=nonstopmode main.tex > nul 2>&1

rem Pass 3
pdflatex -interaction=nonstopmode main.tex > nul 2>&1

rem Results
echo === ERRORS ===
findstr "^!" main.log
echo.
echo === OUTPUT ===
findstr "Output written" main.log
echo.
echo === DONE ===
