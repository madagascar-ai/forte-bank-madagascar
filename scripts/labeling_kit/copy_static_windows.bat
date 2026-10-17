@echo off
rem Double-click after the first frame of a video is labeled and checked. Close X-AnyLabeling first.
cd /d "%~dp0"
if "%XAL_VENV%"=="" (set "VENV=%USERPROFILE%\xal") else (set "VENV=%XAL_VENV%")
"%VENV%\Scripts\python.exe" propagate_static.py
pause
