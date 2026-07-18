@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot error_lines.txt 2>nul
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
biber --input-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" --output-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" main > nul 2>&1
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
findstr "^!" main.log > error_lines.txt 2>&1
findstr "Output written" main.log >> error_lines.txt 2>&1
