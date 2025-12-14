@echo off
setlocal ENABLEDELAYEDEXPANSION

set REPO_RAW=https://raw.githubusercontent.com/GonzalesLabVU/Behavior-BCI/main
set REPO_ZIP=https://github.com/GonzalesLabVU/Behavior-BCI/archive/refs/heads/main.zip

set "REPO_URL=https://github.com/GonzalesLabVU/Behavior-BCI.git"
set "BRANCH=main"

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
set "PY_EXIT=%ERRORLEVEL%"

echo.
echo ================================================
echo Script finished (exit code %PY_EXIT%)
echo Preparing to push latest .xlsx data file...
echo ================================================

set "LATEST_FILE="
for /f "usebackq delims=" %%F in (`
  powershell -NoProfile -Command ^
    "$f=Get-ChildItem -LiteralPath '%SCRIPT_DIR%' -Filter *_data.xlsx -File -ErrorAction SilentlyContinue ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1; if($null -eq $f){ exit 2 } else { $f.Name }"
`) do set "LATEST_FILE=%%F"

if "%LATEST_FILE%"=="" (
  echo No .xlsx files found in "%SCRIPT_DIR%". Skipping push.
  goto :after_push
)

echo Latest .xlsx detected: "%LATEST_FILE%"

REM --- Check git ---
git --version >nul 2>&1
if errorlevel 1 (
  echo WARNING: git not found. Skipping push.
  goto :after_push
)

set "REPO_DIR=%SCRIPT_DIR%Behavior-BCI_repo"

if not exist "%REPO_DIR%\.git" (
  echo Cloning repo into: "%REPO_DIR%"
  git clone "%REPO_URL%" "%REPO_DIR%"
  if errorlevel 1 (
    echo WARNING: git clone failed. Skipping push.
    goto :after_push
  )
)

cd /d "%REPO_DIR%"

git fetch origin >nul 2>&1
git checkout "%BRANCH%" >nul 2>&1 || goto :after_push
git pull origin "%BRANCH%" >nul 2>&1 || goto :after_push

set "DEST_PATH="
set "MATCH_COUNT=0"

for /f "usebackq delims=" %%P in (`git ls-files`) do (
  for %%B in ("%%P") do (
    if /I "%%~nxB"=="%LATEST_FILE%" (
      set "DEST_PATH=%%P"
      set /a MATCH_COUNT+=1
    )
  )
)

if "%DEST_PATH%"=="" (
  echo WARNING: "%LATEST_FILE%" is not tracked in the repo. Skipping push.
  cd /d "%SCRIPT_DIR%"
  goto :after_push
)

if %MATCH_COUNT% GTR 1 (
  echo WARNING: Multiple tracked files named "%LATEST_FILE%" found. Skipping push to avoid ambiguity.
  cd /d "%SCRIPT_DIR%"
  goto :after_push
)

echo Repo destination: "%DEST_PATH%"

copy /Y "%SCRIPT_DIR%%LATEST_FILE%" "%DEST_PATH%" >nul
if errorlevel 1 (
  echo WARNING: Copy into repo failed. Skipping push.
  cd /d "%SCRIPT_DIR%"
  goto :after_push
)

git add "%DEST_PATH%"

git diff --cached --quiet
if not errorlevel 1 (
  echo No changes detected in "%LATEST_FILE%". Nothing to push.
  cd /d "%SCRIPT_DIR%"
  goto :after_push
)

for /f "usebackq delims=" %%T in (`
  powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"
`) do set "NOW_TS=%%T"

git commit -m "Update %LATEST_FILE% (%NOW_TS%)"
if errorlevel 1 (
  echo WARNING: Commit failed. Skipping push.
  cd /d "%SCRIPT_DIR%"
  goto :after_push
)

git push origin "%BRANCH%"
if errorlevel 1 (
  echo WARNING: Push failed. Check GitHub auth.
) else (
  echo Push complete: "%LATEST_FILE%"
)

cd /d "%SCRIPT_DIR%"

:after_push
echo.
pause
exit /b %PY_EXIT%

:download_fail
echo.
echo =================================================
echo ERROR: Failed to download or extract files
echo =================================================
echo Repo: https://github.com/GonzalesLabVU/Behavior-BCI/tree/main
echo check your internet connection and that the listed paths exist in the repo root
pause
exit /b 1
