@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis

del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul

echo PASS1: pdflatex
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
set E1=0
for /f %%A in ('findstr /R /N "^!" main.log ^| find /C ":"') do set E1=%%A
echo errors-pass1=%E1%

echo BIBER
biber --input-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" --output-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" main > nul 2>&1

echo PASS2: pdflatex
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
set E2=0
for /f %%A in ('findstr /R /N "^!" main.log ^| find /C ":"') do set E2=%%A
echo errors-pass2=%E2%

echo PASS3: pdflatex
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
set E3=0
for /f %%A in ('findstr /R /N "^!" main.log ^| find /C ":"') do set E3=%%A
echo errors-pass3=%E3%

echo.
findstr "Output written" main.log
findstr "undefined references" main.log
findstr "Please (re)run Biber" main.log
