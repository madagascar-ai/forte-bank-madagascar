@echo off
rem Double-click to label. First run installs X-AnyLabeling 4.1.3 into %USERPROFILE%\xal (a few minutes).
cd /d "%~dp0"
if "%XAL_VENV%"=="" (set "VENV=%USERPROFILE%\xal") else (set "VENV=%XAL_VENV%")
if exist "%VENV%\Scripts\xanylabeling.exe" goto run
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>nul
if errorlevel 1 (
  echo Python 3.11+ is required: https://www.python.org/downloads/ - install it and run again.
  pause
  exit /b 1
)
echo First run: installing X-AnyLabeling, this takes a few minutes...
py -3 -m venv "%VENV%" || goto failed
"%VENV%\Scripts\pip" install -q x-anylabeling-cvhub==4.1.3 || goto failed
:run
"%VENV%\Scripts\xanylabeling.exe" --filename images --labels classes.txt --validatelabel exact --autosave --nodata --no-auto-update-check
if errorlevel 1 pause
exit /b 0
:failed
echo Installation failed. Send the text above to Diyar (no frame screenshots).
pause
exit /b 1
