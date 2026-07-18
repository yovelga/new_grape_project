@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis
echo Checking files...
if exist frontmatter\title_inner_en.tex (echo FOUND: title_inner_en.tex) else (echo MISSING: title_inner_en.tex)
if exist frontmatter\title_outer_en.tex (echo FOUND: title_outer_en.tex) else (echo MISSING: title_outer_en.tex)
if exist frontmatter\title_inner_he.tex (echo FOUND: title_inner_he.tex) else (echo MISSING: title_inner_he.tex)
if exist frontmatter\title_outer_he.tex (echo FOUND: title_outer_he.tex) else (echo MISSING: title_outer_he.tex)
if exist frontmatter\abstract_he.tex (echo FOUND: abstract_he.tex) else (echo MISSING: abstract_he.tex)
if exist chapters\acknowledgments.tex (echo FOUND: acknowledgments.tex) else (echo MISSING: acknowledgments.tex)
echo.
echo Cleaning...
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul
echo.
echo Pass 1...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
if errorlevel 1 (echo PASS1 FAILED) else (echo PASS1 OK)
echo.
echo Biber...
biber main > nul 2>&1
if errorlevel 1 (echo BIBER FAILED) else (echo BIBER OK)
echo.
echo Pass 2...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
if errorlevel 1 (echo PASS2 FAILED) else (echo PASS2 OK)
echo.
echo Pass 3...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
if errorlevel 1 (echo PASS3 FAILED) else (echo PASS3 OK)
echo.
echo === ERRORS ===
findstr "^!" main.log
echo.
echo === OUTPUT ===
findstr "Output written" main.log
