@echo off
echo.
setlocal EnableExtensions EnableDelayedExpansion

cls
call :sleep 2

echo Getting script directory...
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

rem ================================================================
rem  CONFIGURATION
rem ================================================================
set "FQBN=arduino:avr:mega"
set "ARDUINO_CLI=arduino-cli"

rem ================================================================
rem  VERIFY & UPDATE ARDUINO ENVIRONMENT
rem ================================================================
echo Verifying arduino-cli and updating core/libraries...
call :verifyArduino || call :kill "arduino-cli installation or update failed"
call :sleep

rem ================================================================
rem  DETECT BOARD & UPLOAD LOCAL SKETCH
rem ================================================================
echo Searching for Arduino Mega...
call :detectPort || call :kill "No Arduino Mega detected on COM ports"
call :uploadSketch "behavioral_controller" || call :kill "Arduino compile/upload failed"

call :sleep

rem ================================================================
rem  RUN PYTHON SCRIPT
rem ================================================================
echo Running Python script...
python -m behavioral_master

call :sleep
echo.
goto :eof

rem ================================================================
rem  SUBROUTINES
rem ================================================================

rem ----------------------------------------------------------------
rem :verifyArduino
rem   Ensures arduino-cli is installed, updates core/lib indexes,
rem   and upgrades arduino:avr and Servo to their latest versions.
rem ----------------------------------------------------------------
:verifyArduino
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

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

    echo   Updating core index and arduino:avr...
    arduino-cli core update-index >nul 2>&1
    arduino-cli core install arduino:avr >nul 2>&1
    arduino-cli core upgrade arduino:avr >nul 2>&1

    echo   Updating library index and Servo...
    arduino-cli lib update-index >nul 2>&1
    arduino-cli lib install Servo >nul 2>&1
    arduino-cli lib upgrade Servo >nul 2>&1

    endlocal & exit /b 0

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

    echo Compiling sketch from %SKETCH_DIR%...
    "%ARDUINO_CLI%" compile --fqbn "%FQBN%" "%SKETCH_DIR%" >nul 2>&1
    if errorlevel 1 (echo [ERROR] Sketch compilation failed & endlocal & exit /b 1)

    echo Uploading sketch to %PORT%...
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
rem ----------------------------------------------------------------
:kill
    echo.
    echo [FATAL] %~1
    echo.
    endlocal
    pause
    exit 1