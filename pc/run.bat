@echo off
echo.
setlocal EnableExtensions EnableDelayedExpansion

cls
call :sleep 2

echo Getting script directory...
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
call :sleep

rem ================================================================
rem  CONFIGURATION
rem  To add, remove, or rename files: edit the lists below.
rem  Only basenames are needed — locations in the repo are resolved
rem  automatically via the GitHub Trees API, so moving files in the
rem  repo requires no changes here.
rem ================================================================
set "REPO_OWNER=GonzalesLabVU"
set "REPO_NAME=Behavior-BCI"
set "BRANCH=main"
set "FQBN=arduino:avr:mega"
set "ARDUINO_CLI=arduino-cli"

set "FC=0"
set /a FC+=1 & set "PULL_FILE[!FC!]=behavioral_master.py"
set /a FC+=1 & set "PULL_FILE[!FC!]=TCPClient.py"
set /a FC+=1 & set "PULL_FILE[!FC!]=cursor_utils.py"
set /a FC+=1 & set "PULL_FILE[!FC!]=plot_utils.py"
set /a FC+=1 & set "PULL_FILE[!FC!]=animal_map.json"
set /a FC+=1 & set "PULL_FILE[!FC!]=requirements.txt"
set /a FC+=1 & set "PULL_FILE[!FC!]=errors.log"

set "DC=0"
set /a DC+=1 & set "PULL_DIR[!DC!]=behavioral_controller"

rem ================================================================
rem  OUTPUT DIRECTORY
rem  Usage:  run.bat                  -> pull files directly into SCRIPT_DIR
rem          run.bat \some_subfolder  -> pull files into SCRIPT_DIR\some_subfolder
rem                                     (created if it does not already exist)
rem  When a subfolder argument is supplied the overwrite-confirmation
rem  prompt is skipped and local files in SCRIPT_DIR are never touched.
rem ================================================================
set "OUT_DIR=%SCRIPT_DIR%"
if not "%~1"=="" (
    set "ARG=%~1"
    if "!ARG:~0,1!"=="\" set "ARG=!ARG:~1!"
    set "OUT_DIR=%SCRIPT_DIR%\!ARG!"
    if not exist "!OUT_DIR!\" (
        echo Creating output directory: !OUT_DIR!
        mkdir "!OUT_DIR!" || call :kill "Failed to create output directory: !OUT_DIR!"
    )
    echo Output directory: !OUT_DIR!
    echo.
)

rem ================================================================
rem  UPDATE PROMPT
rem ================================================================
set "DO_UPDATE=N"
set /p "DO_UPDATE=Update local files? [y/N]: "
if "%DO_UPDATE%"=="" set "DO_UPDATE=N"
echo.

if /i "%DO_UPDATE%"=="Y" (
    rem Only ask for overwrite confirmation when writing directly to SCRIPT_DIR
    if /i "!OUT_DIR!"=="%SCRIPT_DIR%" (
        set "CONFIRM=N"
        set /p "CONFIRM=This will overwrite local files. Continue? [y/N]: "
        if "!CONFIRM!"=="" set "CONFIRM=N"
        if /i "!CONFIRM!" NEQ "Y" (
            set "DO_UPDATE=N"
            echo Update cancelled.
            echo.
        )
    )
)

if /i "%DO_UPDATE%"=="Y" (
    echo Making sure pip is up to date...
    python -m pip install --upgrade pip -q
    call :sleep

    call :verifyGit     || call :kill "Git installation failed or git.exe not on PATH"
    call :sleep
    call :verifyArduino || call :kill "arduino-cli installation failed or not on PATH"
    call :sleep

    rem Fetch the full repo file tree once and cache it locally.
    rem All per-file path lookups query this cache — no extra API calls.
    echo Fetching repository file tree...
    set "TREE_CACHE=%TEMP%\bci_tree_%RANDOM%.json"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri 'https://api.github.com/repos/%REPO_OWNER%/%REPO_NAME%/git/trees/%BRANCH%?recursive=1' -OutFile '!TREE_CACHE!' -UseBasicParsing" >nul 2>&1
    if not exist "!TREE_CACHE!" call :kill "Failed to fetch repository tree from GitHub API"
    echo File tree cached.
    call :sleep

    echo Downloading files...
    for /l %%i in (1,1,%FC%) do (
        set "FNAME=!PULL_FILE[%%i]!"
        echo   Pulling !FNAME!...
        call :pullFileByName "!FNAME!" || call :kill "Failed to pull !FNAME!"
    )
    call :sleep

    echo Downloading folders...
    for /l %%i in (1,1,%DC%) do (
        set "DNAME=!PULL_DIR[%%i]!"
        echo   Pulling !DNAME!\...
        call :pullFolderByName "!DNAME!" || call :kill "Failed to pull folder !DNAME!"
    )

    if exist "!TREE_CACHE!" del "!TREE_CACHE!" >nul 2>&1
    call :sleep

    echo Installing required packages...
    if not exist "!OUT_DIR!\requirements.txt" (
        echo [ERROR] requirements.txt not found at !OUT_DIR!\requirements.txt
        call :kill "requirements.txt missing"
    )
    python -m pip install -r "!OUT_DIR!\requirements.txt" -q || call :kill "pip install failed"
)

call :sleep

echo Searching for Arduino...
call :detectPort || echo [WARNING] No Arduino detected
call :uploadSketch "behavioral_controller" || call :kill "Arduino compile/upload failed"

call :sleep

echo Running Python script...
python -m behavioral_master

call :sleep
echo.
goto :eof

rem ================================================================
rem  SUBROUTINES
rem ================================================================

rem ----------------------------------------------------------------
rem :verifyGit
rem   Ensures git is available on PATH, installing it if necessary.
rem ----------------------------------------------------------------
:verifyGit
    @echo off
    setlocal EnableExtensions

    echo Verifying git installation...
    where git >nul 2>&1
    if not errorlevel 1 (endlocal & exit /b 0)

    echo Git not found, attempting to install...

    where winget >nul 2>&1
    if not errorlevel 1 winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements >nul 2>&1

    where git >nul 2>&1
    if errorlevel 1 (
        where choco >nul 2>&1
        if not errorlevel 1 choco install git -y >nul 2>&1
    )

    where git >nul 2>&1
    if errorlevel 1 (
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "$o = Join-Path $env:TEMP 'Git-64-bit.exe';" ^
            "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/latest/download/Git-64-bit.exe' -OutFile $o -UseBasicParsing;" ^
            "Start-Process $o -ArgumentList '/VERYSILENT','/NORESTART','/SUPPRESSMSGBOXES' -Wait"
    )

    if exist "%ProgramFiles%\Git\cmd\git.exe"      set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
    if exist "%ProgramFiles(x86)%\Git\cmd\git.exe" set "PATH=%ProgramFiles(x86)%\Git\cmd;%PATH%"

    where git >nul 2>&1
    if errorlevel 1 (endlocal & exit /b 1)
    for /f "delims=" %%v in ('git --version 2^>nul') do echo %%v
    endlocal & exit /b 0

rem ----------------------------------------------------------------
rem :verifyArduino
rem   Ensures arduino-cli is available, installing it if necessary,
rem   and initialises the arduino:avr core and Servo library.
rem ----------------------------------------------------------------
:verifyArduino
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    echo Verifying arduino-cli installation...
    where arduino-cli >nul 2>&1
    if not errorlevel 1 goto :arduinoReady

    echo arduino-cli not found, attempting to install...
    where winget >nul 2>&1
    if not errorlevel 1 winget install --id ArduinoSA.CLI -e --source winget --accept-package-agreements --accept-source-agreements >nul 2>&1

    if exist "%LocalAppData%\Programs\Arduino CLI\arduino-cli.exe" set "PATH=%LocalAppData%\Programs\Arduino CLI;%PATH%"
    if exist "%ProgramFiles%\Arduino CLI\arduino-cli.exe"          set "PATH=%ProgramFiles%\Arduino CLI;%PATH%"

    where arduino-cli >nul 2>&1
    if errorlevel 1 (endlocal & exit /b 1)

:arduinoReady
    arduino-cli config init >nul 2>&1
    arduino-cli core update-index >nul 2>&1
    arduino-cli core list | findstr /i "arduino:avr" >nul 2>&1 || arduino-cli core install arduino:avr >nul 2>&1
    arduino-cli lib install Servo >nul 2>&1
    endlocal & exit /b 0

rem ----------------------------------------------------------------
rem :pullFileByName  <basename>
rem   Looks up <basename> in the cached repo tree and downloads it
rem   directly from raw.githubusercontent.com into OUT_DIR.
rem ----------------------------------------------------------------
:pullFileByName
    setlocal EnableExtensions EnableDelayedExpansion
    set "FNAME=%~1"

    for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$j = Get-Content '%TREE_CACHE%' -Raw | ConvertFrom-Json;" ^
        "$m = $j.tree | Where-Object { $_.type -eq 'blob' -and ($_.path -split '/')[-1] -eq '%FNAME%' } | Select-Object -First 1;" ^
        "if ($m) { $m.path } else { exit 1 }"`) do set "FPATH=%%P"

    if not defined FPATH (
        echo [ERROR] '%FNAME%' not found in repository tree
        endlocal & exit /b 1
    )

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/%REPO_OWNER%/%REPO_NAME%/%BRANCH%/!FPATH!' -OutFile '%OUT_DIR%\%FNAME%' -UseBasicParsing" >nul 2>&1

    endlocal & exit /b %ERRORLEVEL%

rem ----------------------------------------------------------------
rem :pullFolderByName  <foldername>
rem   Looks up <foldername> in the cached repo tree, then delegates
rem   to :pullFolder using the resolved path.
rem ----------------------------------------------------------------
:pullFolderByName
    setlocal EnableExtensions EnableDelayedExpansion
    set "DNAME=%~1"

    for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$j = Get-Content '%TREE_CACHE%' -Raw | ConvertFrom-Json;" ^
        "$m = $j.tree | Where-Object { $_.type -eq 'tree' -and ($_.path -split '/')[-1] -eq '%DNAME%' } | Select-Object -First 1;" ^
        "if ($m) { $m.path } else { exit 1 }"`) do set "DPATH=%%P"

    if not defined DPATH (
        echo [ERROR] Folder '%DNAME%' not found in repository tree
        endlocal & exit /b 1
    )

    call :pullFolder "https://github.com/%REPO_OWNER%/%REPO_NAME%/tree/%BRANCH%/!DPATH!" "%OUT_DIR%"
    endlocal & exit /b %ERRORLEVEL%

rem ----------------------------------------------------------------
rem :pullFolder  <github-tree-url>  [dest-parent-dir]
rem   Sparse-clones a single folder from a GitHub tree URL and
rem   copies it into dest-parent-dir (defaults to SCRIPT_DIR).
rem ----------------------------------------------------------------
:pullFolder
    @echo off
    setlocal EnableDelayedExpansion

    set "RC=1"
    set "URL=%~1"
    set "DEST_PARENT=%~2"
    if not defined DEST_PARENT set "DEST_PARENT=%SCRIPT_DIR%"

    if not defined URL (
        echo [pullFolder] Missing URL argument
        endlocal & exit /b 1
    )

    :trimFolderURL
        if "%URL:~-1%"=="/" set "URL=%URL:~0,-1%" & goto trimFolderURL
        if "%URL:~-1%"=="\" set "URL=%URL:~0,-1%" & goto trimFolderURL

    set "U=%URL:https://=%"
    for /f "tokens=1-5* delims=/" %%a in ("%U%") do (
        set "HOST=%%a"
        set "OWNER=%%b"
        set "REPO=%%c"
        set "TREE=%%d"
        set "BRANCH_L=%%e"
        set "SUBPATH=%%f"
    )

    if /i not "%HOST%"=="github.com" (
        echo [pullFolder] URL host must be github.com, got "%HOST%"
        endlocal & exit /b 1
    )
    if /i not "%TREE%"=="tree" (
        echo [pullFolder] Invalid GitHub tree URL
        endlocal & exit /b 1
    )

    set "REPO_URL=https://github.com/%OWNER%/%REPO%.git"
    set "SUBPATH_WIN=%SUBPATH:/=\%"
    for %%L in ("%SUBPATH_WIN%") do set "LEAF=%%~nxL"
    set "TMP=_repo_tmp_%RANDOM%_%RANDOM%"
    set "PUSHED=0"

    rmdir /s /q "%TMP%" 2>nul
    mkdir "%TMP%"                                                              || (set "RC=1" & goto :folderCleanup)
    pushd "%TMP%"                                                              || (set "RC=1" & goto :folderCleanup)
    set "PUSHED=1"

    git init >nul 2>&1                                                         || (set "RC=1" & goto :folderCleanup)
    git remote add origin "%REPO_URL%" >nul 2>&1                              || (set "RC=1" & goto :folderCleanup)
    git fetch --quiet --no-progress --depth 1 origin "%BRANCH_L%" >nul        || (set "RC=1" & goto :folderCleanup)
    git checkout --quiet --detach FETCH_HEAD >nul                              || (set "RC=1" & goto :folderCleanup)
    git sparse-checkout init --cone >nul 2>&1                                  || (set "RC=1" & goto :folderCleanup)
    git sparse-checkout set "%SUBPATH%" >nul 2>&1                              || (set "RC=1" & goto :folderCleanup)

    popd
    set "PUSHED=0"

    :trimSub
        if "%SUBPATH:~-1%"=="/" set "SUBPATH=%SUBPATH:~0,-1%" & goto trimSub
        if "%SUBPATH:~-1%"=="\" set "SUBPATH=%SUBPATH:~0,-1%" & goto trimSub

    rmdir /s /q "%DEST_PARENT%\%LEAF%" 2>nul
    xcopy "%TMP%\%SUBPATH_WIN%" "%DEST_PARENT%\%LEAF%\" /E /I /Y >nul         || (set "RC=1" & goto :folderCleanup)
    set "RC=0"

:folderCleanup
    if "%PUSHED%"=="1" popd
    cd /d "%SCRIPT_DIR%" >nul 2>&1

    set "DELETED=0"
    for /l %%i in (1,1,5) do (
        rmdir /s /q "%TMP%" 2>nul
        if not exist "%TMP%\" set "DELETED=1"
        if not "!DELETED!"=="1" call :sleep 2
    )
    if "%DELETED%"=="0" (
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "if (Test-Path '%TMP%') { Remove-Item -LiteralPath '%TMP%' -Recurse -Force -ErrorAction SilentlyContinue }" >nul 2>&1
    )
    endlocal & exit /b %RC%

rem ----------------------------------------------------------------
rem :detectPort
rem   Detects the COM port of a connected arduino:avr:mega board
rem   and exports it as the PORT environment variable.
rem ----------------------------------------------------------------
:detectPort
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    set "PORT="

    for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command ^
        "$ErrorActionPreference='SilentlyContinue';" ^
        "$out = arduino-cli board list --format json 2>$null;" ^
        "if ($LASTEXITCODE -ne 0 -or -not $out) { exit 0 };" ^
        "$j = $out | ConvertFrom-Json;" ^
        "$p = $j.ports | Where-Object { $_.matching_boards.fqbn -contains 'arduino:avr:mega' } | Select-Object -First 1;" ^
        "if ($p) { $p.address }"`) do set "PORT=%%P"

    if defined PORT (endlocal & set "PORT=%PORT%" & exit /b 0)

    for /f "tokens=1" %%A in ('arduino-cli board list 2^>nul ^| findstr /i "arduino:avr:mega"') do (
        endlocal & set "PORT=%%A" & exit /b 0
    )

    endlocal & exit /b 1

rem ----------------------------------------------------------------
rem :uploadSketch  <sketch-folder-name>
rem   Compiles and uploads the named .ino sketch from SCRIPT_DIR.
rem   The arduino sketch is always sourced from SCRIPT_DIR so that
rem   a test-mode pull (with a subfolder argument) does not affect
rem   the live firmware.
rem ----------------------------------------------------------------
:uploadSketch
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    set "SKETCH_DIR=%SCRIPT_DIR%\%~1"

    if not defined SKETCH_DIR    (echo [ERROR] Missing sketch folder argument  & endlocal & exit /b 1)
    if not exist "%SKETCH_DIR%\" (echo [ERROR] Folder not found: "%SKETCH_DIR%" & endlocal & exit /b 1)

    set "INO="
    for %%I in ("%SKETCH_DIR%\*.ino") do (set "INO=%%~fI" & goto :gotINO)
:gotINO
    if not defined INO  (echo [ERROR] No .ino file found in "%SKETCH_DIR%" & endlocal & exit /b 1)
    if not defined FQBN (echo [ERROR] FQBN not set                          & endlocal & exit /b 1)
    if not defined PORT (echo [ERROR] PORT not set                          & endlocal & exit /b 1)

    echo Compiling sketch...
    "%ARDUINO_CLI%" compile --fqbn "%FQBN%" "%SKETCH_DIR%" >nul 2>&1
    if errorlevel 1 (echo [ERROR] Sketch compilation failed & endlocal & exit /b 1)

    echo Uploading sketch...
    "%ARDUINO_CLI%" upload --port "%PORT%" --fqbn "%FQBN%" "%SKETCH_DIR%" >nul 2>&1
    if errorlevel 1 (echo [ERROR] Sketch upload failed & endlocal & exit /b 1)

    endlocal & exit /b 0

rem ----------------------------------------------------------------
rem :sleep  [seconds]   (default 0.5)
rem ----------------------------------------------------------------
:sleep
    setlocal
    set "DT=%~1"
    if not defined DT set "DT=0.5"
    python -c "import time; time.sleep(%DT%)"
    endlocal & exit /b 0

rem ----------------------------------------------------------------
rem :kill  <message>
rem   Prints a fatal-error message, pauses, and exits the script.
rem ----------------------------------------------------------------
:kill
    echo.
    echo [FATAL] %~1
    echo.
    endlocal
    pause
    exit 1
