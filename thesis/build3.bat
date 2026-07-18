@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis
echo CWD is: %CD%
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul
echo Running pdflatex...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
echo After pdflatex, CWD is: %CD%
dir main.bcf 2>nul
echo Running biber...
biber --debug main 2>&1
echo After biber, CWD is: %CD%
