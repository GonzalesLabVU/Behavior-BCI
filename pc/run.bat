@echo off
setlocal ENABLEDELAYEDEXPANSION

set REPO_RAW=https://raw.githubusercontent.com/GonzalesLabVU/Behavior-BCI/main
set REPO_ZIP=https://github.com/GonzalesLabVU/Behavior-BCI/archive/refs/heads/main.zip

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo =================================================
echo Updating files from GitHub...
echo =================================================

REM --- Core Python files ---
curl -fsSL "%REPO_RAW%/behavioral_master.py" -o "behavioral_master.py" || goto :download_fail
curl -fsSL "%REPO_RAW%/cursor_utils.py" -o "cursor_utils.py" || goto :download_fail
curl -fsSL "%REPO_RAW%/plot_utils.py" -o "plot_utils.py" || goto :download_fail

REM --- Data files ---
curl -fsSL "%REPO_RAW%/ACD_data.xlsx" -o "ACD_data.xlsx" || goto :download_fail
curl -fsSL "%REPO_RAW%/EGI_data.xlsx" -o "EGI_data.xlsx" || goto :download_fail
curl -fsSL "%REPO_RAW%/QVWX_data.xlsx" -o "QVWX_data.xlsx" || goto :download_fail
curl -fsSL "%REPO_RAW%/animal_map.json" -o "animal_map.json" || goto :download_fail

REM --- Arduino files ---
echo Downloading behavioral_controller folder...
curl -fsSL "%REPO_ZIP%" -o "repo_tmp.zip" || goto :download_fail

if exist "Behavior-BCI-main" rmdir /S /Q "Behavior-BCI-main" >nul 2>&1
if exist "behavioral_controller" rmdir /S /Q "behavioral_controller" >nul 2>&1

tar -xf "repo_tmp.zip" || goto :download_fail
del "repo_tmp.zip" >nul 2>&1

if not exist "Behavior-BCI-main\behavioral_controller" goto :download_fail

xcopy /E /I /Y "Behavior-BCI-main\behavioral_controller" "behavioral_controller\" >nul
rmdir /S /Q "Behavior-BCI-main" >nul 2>&1

echo Files updated successfully
TIMEOUT /T 3 /NOBREAK > NUL

echo.
echo ================================================
echo   Installing required Python packages...
echo ================================================

python -m pip install --upgrade pip
python -m pip install --upgrade ^
    pyserial==3.5 ^
    openpyxl==3.1.5 ^
    matplotlib==3.7.2 ^
    numpy==1.26.4 ^
    pygame==2.6.1 ^
    scipy==1.15.3

echo.
echo ================================================
echo   Running Python script...
echo ================================================

python behavioral_master.py

echo.
pause
exit /b 0

:download_fail
echo.
echo =================================================
echo ERROR: Failed to download or extract files
echo =================================================
echo Repo: https://github.com/GonzalesLabVU/Behavior-BCI/tree/main
echo check your internet connection and that the listed paths exist in the repo root
pause
exit /b 1
