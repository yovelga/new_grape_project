@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul
echo Pass 1...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
echo Biber with full path...
biber "C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\main" 2>&1
echo.
echo Biber with input-directory...
biber --input-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" main 2>&1
