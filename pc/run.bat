@echo off

echo.
echo ================================================
echo   Installing required Python packages...
echo ================================================

python -m pip install --upgrade ^
    openpyxl==3.1.5 ^
    matplotlib==3.7.2 ^
    numpy==1.26.4 ^
    pygame==2.6.1 ^
    scipy==1.15.3

echo.
echo ================================================
echo   Running Python script...
echo ================================================

REM Move to the script directory
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

python behavioral_master.py

echo.
pause
