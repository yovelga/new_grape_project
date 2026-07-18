@echo off
cd /d C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis

echo === DIAGNOSTIC BUILD ===
echo.

echo Step 1: Check biber location...
where biber
echo.

echo Step 2: Check biber version...
biber --version
echo.

echo Step 3: Clean all aux files...
del /q main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.lof main.lot 2>nul
echo Cleaned.
echo.

echo Step 4: Run pdflatex pass 1...
pdflatex -interaction=nonstopmode main.tex > nul 2>&1
echo pdflatex done. Checking BCF...
echo.

echo Step 5: BCF file info...
if exist main.bcf (
    echo BCF EXISTS
    dir main.bcf
    echo.
    echo First 3 lines of BCF:
    powershell -c "Get-Content main.bcf -Head 3"
) else (
    echo BCF DOES NOT EXIST
    goto :end
)
echo.

echo Step 6: Pin BCF file (force OneDrive download)...
attrib -U +P main.bcf 2>nul
attrib -U +P main.run.xml 2>nul
attrib -U +P references.bib 2>nul
echo Pinned.
echo.

echo Step 7: Try biber with full path...
biber "C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\main" 2>&1
echo.
echo Biber exit code: %ERRORLEVEL%
echo.

if exist main.bbl (
    echo BBL EXISTS - biber worked!
    echo.
    echo Step 8: Running remaining pdflatex passes...
    pdflatex -interaction=nonstopmode main.tex > nul 2>&1
    pdflatex -interaction=nonstopmode main.tex > nul 2>&1
    echo.
    echo === OUTPUT ===
    findstr "Output written" main.log
    echo.
    echo === ERRORS ===
    findstr "^!" main.log
) else (
    echo BBL DOES NOT EXIST
    echo.
    echo Step 7b: Try biber with --input-directory...
    biber --input-directory="C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis" main 2>&1
    echo.
    echo Biber exit code: %ERRORLEVEL%
    if exist main.bbl (
        echo BBL NOW EXISTS
        pdflatex -interaction=nonstopmode main.tex > nul 2>&1
        pdflatex -interaction=nonstopmode main.tex > nul 2>&1
        findstr "Output written" main.log
    ) else (
        echo STILL NO BBL
        echo.
        echo Step 7c: Try copying to temp and building there...
        mkdir "%TEMP%\thesis_build" 2>nul
        xcopy /y /s "C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\*" "%TEMP%\thesis_build\" > nul 2>&1
        cd /d "%TEMP%\thesis_build"
        echo Building from %CD%...
        biber main 2>&1
        echo Biber exit code: %ERRORLEVEL%
        if exist main.bbl (
            echo TEMP BUILD BBL EXISTS - OneDrive was the problem!
            pdflatex -interaction=nonstopmode main.tex > nul 2>&1
            pdflatex -interaction=nonstopmode main.tex > nul 2>&1
            findstr "Output written" main.log
            echo.
            echo Copying results back...
            copy /y main.bbl "C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\" > nul
            copy /y main.pdf "C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\" > nul
        ) else (
            echo TEMP BUILD ALSO FAILED
        )
    )
)

:end
echo.
echo === DONE ===
