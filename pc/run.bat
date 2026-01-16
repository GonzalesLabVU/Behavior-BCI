@echo off
echo.
setlocal EnableExtensions EnableDelayedExpansion

call :selfUpdate
if "%ERRORLEVEL%"=="99" exit /b 0
if errorlevel 1 call :kill "selfUpdate failed"

echo Getting script directory...
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "DO_UPDATE=Y"
set /p "DO_UPDATE=Update local files? [Y/n]: "
if not defined DO_UPDATE set "DO_UPDATE=Y"

if /i "%DO_UPDATE%"=="Y" (
    echo Making sure pip is up to date...
    python -m pip install --upgrade pip -q

    echo Verifying git installation...
    where git >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Git not found, attempting to install...

        where winget >nul 2>&1
        if not errorlevel 1 (
            winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements >nul 2>&1
        )

        where git >nul 2>&1
        if errorlevel 1 (
            where choco >nul 2>&1
            if not errorlevel 1 (
                choco install git -y >nul 2>&1
            )
        )

        where git >nul 2>&1
        if errorlevel 1 (
            powershell -NoProfile -ExecutionPolicy Bypass -Command ^
                "$ErrorActionPreference='Stop';" ^
                "$u='https://github.com/git-for-windows/git/releases/latest/download/Git-64-bit.exe';" ^
                "$o=Join-Path $env:TEMP 'Git-64-bit.exe';" ^
                "Invoke-WebRequest -Uri $u -OutFile $o -UseBasicParsing;" ^
                "Start-Process -FilePath $o -ArgumentList '/VERYSILENT','/NORESTART','/SUPPRESSMSGBOXES' -Wait;"
        )

        if exist "%ProgramFiles%\Git\cmd\git.exe" set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
        if exist "%ProgramFiles(x86)%\Git\cmd\git.exe" set "PATH=%ProgramFiles(x86)%\Git\cmd;%PATH%"

        where git >nul 2>&1
        if errorlevel 1 (
            call :kill "Git installation failed or git.exe not on PATH"
        ) else (
            for /f "delims=" %%v in ('git --version 2^>nul') do echo %%v
        )
    )

    echo Verifying arduino-cli installation...
    where arduino-cli >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] arduino-cli not found on path
        echo Installing arduino-cli...
        where winget >nul 2>&1 || call :kill "winget not found; install arduino-cli manually"
        winget install --id ArduinoSA.CLI -e --source winget --accept-package-agreements --accept-source-agreements >nul 2>&1
    )

    where arduino-cli >nul 2>&1
    if errorlevel 1 (
        if exist "%LocalAppData%\Programs\Arduino CLI\arduino-cli.exe" set "PATH=%LocalAppData%\Programs\Arduino CLI;%PATH%"
        if exist "%ProgramFiles%\Arduino CLI\arduino-cli.exe" set "PATH=%ProgramFiles%\Arduino CLI;%PATH%"
    )
    where arduino-cli >nul 2>&1 || call :kill "arduino-cli install successful but arduino-cli.exe not found on PATH"

    arduino-cli config init >nul 2>&1
    arduino-cli core update-index >nul 2>&1
    arduino-cli core list | findstr /i "arduino:avr" >nul 2>&1 || arduino-cli core install arduino:avr >nul 2>&1

    echo Installing required Arduino libraries...
    arduino-cli lib install Servo >nul 2>&1

    arduino-cli version >nul 2>&1
    if errorlevel 1 call :kill "arduino-cli is present but not runnable"

    echo Downloading latest file versions...
    call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/behavioral_master.py" || call :kill "pullFile subroutine failed for behavioral_master.py"
    call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/cursor_utils.py" || call :kill "pullFile failed for cursor_utils.py"
    call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/plot_utils.py" || call :kill "pullFile failed for plot_utils.py"
    call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/config/animal_map.json" || call :kill "pullFile failed for animal_map.json"
    call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/config/requirements.txt" || call :kill "pullFile failed for requirements.txt"
    call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/config/errors.log" || call :kill "pullFile failed for errors.log"
    call :pullFolder "https://github.com/GonzalesLabVU/Behavior-BCI/tree/main/arduino/behavioral_controller" || call :kill "pullFolder subroutine failed for behavioral_controller\"

    call :sleep 1

    echo Installing required dependencies...
    if not exist "requirements.txt" (
        echo requirements.txt not found in %SCRIPT_DIR%
        exit /b 1
    )
    python -m pip install -r requirements.txt -q || exit /b 1
)

echo Searching for Arduino...
set "ARDUINO_CLI=arduino-cli"
set "FQBN=arduino:avr:mega"
call :detectPort || call :kill "No Arduino Mega 2560 detected"

if /i "%DO_UPDATE%"=="Y" (
    call :uploadSketch "behavioral_controller" || call :kill "Arduino upload failed"
)

echo Running Python script...
python -m behavioral_master
call :sleep 5

echo.
echo Session finished
echo.

goto :eof

REM -----------------------------------
REM SUBROUTINES
REM -----------------------------------

:selfUpdate
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    set "RAW_SELF_URL=https://raw.githubusercontent.com/GonzalesLabVU/Behavior-BCI/main/pc/run.bat"

    set "SELF=%~f0"
    set "NEW=%TEMP%\%~n0_new%~x0"
    set "UPD=%TEMP%\%~n0_upd.cmd"

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$u='%RAW_SELF_URL%'; $o='%NEW%';" ^
        "try { Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $o; exit 0 } catch { exit 1 }" >nul 2>&1

    if not exist "%NEW%" endlocal & exit /b 0

    fc /b "%SELF%" "%NEW%" >nul 2>&1
    if not errorlevel 1 (
        del "%NEW%" >nul 2>&1
        endlocal & exit /b 0
    )

    >"%UPD%" echo @echo off
    >>"%UPD%" echo ping 127.0.0.1 -n 2 ^>nul
    >>"%UPD%" echo copy /y "%NEW%" "%SELF%" ^>nul
    >>"%UPD%" echo del "%NEW%" ^>nul 2^>^&1
    >>"%UPD%" echo start "" "%SELF%"
    >>"%UPD%" echo del "%%~f0"

    start "" cmd /c "%UPD%"
    endlocal & exit /b 99

:pullFile
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    set "URL=%~1"
    if not defined URL (
        echo Missing URL argument
        endlocal & exit /b 1
    )

    set "RAW_URL=%URL:github.com=raw.githubusercontent.com%"
    set "RAW_URL=%RAW_URL:/blob/=/%"

    for %%F in ("%URL%") do set "FILENAME=%%~nxF"
    if not defined FILENAME (
        echo Could not resolve filename from URL
        endlocal & exit /b 1
    )

    set "DEST=%~dp0%FILENAME%"

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$u='%RAW_URL%'; $o='%DEST%';" ^
        "try { Invoke-WebRequest -Uri $u -OutFile $o -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"

    set "RC=%ERRORLEVEL%"
    if not "%RC%"=="0" (
        endlocal & exit /b %RC%
    )

    endlocal & exit /b 0

:pullFolder
    @echo off
    setlocal EnableDelayedExpansion
  
    set "RC=1"
    set "URL=%~1"
    if not defined URL (
        echo [pullFolder] Missing URL argument
        endlocal & exit /b 1
    )

    :trimFolderURL
        if "%URL:~-1%"=="/" set "URL=%URL:~0,-1%" & goto trimFolderURL
        if "%URL:~-1%"=="\" set "URL=%URL:~0,-1%" & goto trimFolderURL

    set "U=%URL:https://=%"
    set "U=%U:http://=%"

    for /f "tokens=1-5* delims=/" %%a in ("%U%") do (
        set "HOST=%%a"
        set "OWNER=%%b"
        set "REPO=%%c"
        set "TREE=%%d"
        set "BRANCH=%%e"
        set "SUBPATH=%%f"
    )

    if /i not "%HOST%"=="github.com" (
        echo URL host must be github.com, but got "%HOST%"
        endlocal & exit /b 1
    )

    if /i not "%TREE%"=="tree" (
        echo Invalid GitHub tree URL
        endlocal & exit /b 1
    )

    set "REPO_URL=https://github.com/%OWNER%/%REPO%.git"
    set "TMP=_repo_tmp_%RANDOM%_%RANDOM%"
    set "SUBPATH_WIN=%SUBPATH:/=\%"

    for %%L in ("%SUBPATH_WIN%") do set "LEAF=%%~nxL"

    set "PUSHED=0"
    rmdir /s /q "%TMP%" 2>nul
    mkdir "%TMP%" || (set "RC=1" & goto :folderCleanup)
    pushd "%TMP%" || (set "RC=1" & goto :folderCleanup)
    set "PUSHED=1"

    git init >nul 2>&1 || (set "RC=1" & goto :folderCleanup)
    git remote add origin "%REPO_URL%" >nul 2>&1 || (set "RC=1" & goto :folderCleanup)
    git fetch --quiet --no-progress --depth 1 origin "%BRANCH%" >nul || (set "RC=1" & goto :folderCleanup)
    git checkout --quiet --detach FETCH_HEAD >nul || (set "RC=1" & goto :folderCleanup)
    git sparse-checkout init --cone >nul 2>&1 || (set "RC=1" & goto :folderCleanup)
    git sparse-checkout set "%SUBPATH%" >nul 2>&1 || (set "RC=1" & goto :folderCleanup)

    popd
    set "PUSHED=0"

    :trimSub
        if "%SUBPATH:~-1%"=="/" set "SUBPATH=%SUBPATH:~0,-1%" & goto trimSub
        if "%SUBPATH:~-1%"=="\" set "SUBPATH=%SUBPATH:~0,-1%" & goto trimSub

    rmdir /s /q "%LEAF%" 2>nul
    xcopy "%TMP%\%SUBPATH_WIN%" "%LEAF%\" /E /I /Y >nul || (set "RC=1" & goto :folderCleanup)

    set "RC=0"
   
:folderCleanup
    if "%PUSHED%"=="1" popd
    cd /d "%~dp0" >nul 2>&1

    set "DELETED=0"
    for /l %%i in (1,1,5) do (
        rmdir /s /q "%TMP%" 2>nul
        if not exist "%TMP%\" (
            set "DELETED=1"
        ) else (
            call :sleep 2
        )
    )

    if "%DELETED%"=="0" (
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "if (Test-Path '%TMP%') { Remove-Item -LiteralPath '%TMP%' -Recurse -Force -ErrorAction SilentlyContinue }" >nul 2>&1
    )

    endlocal & exit /b %RC%

:detectPort
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    set "PORT="

    for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; $out = arduino-cli board list --format json 2>$null; if($LASTEXITCODE -ne 0 -or -not $out){ exit 0 }; $j = $out | ConvertFrom-Json; $p = $j.ports | Where-Object { $_.matching_boards.fqbn -contains 'arduino:avr:mega' } | Select-Object -First 1; if($p){ $p.address }"`) do set "PORT=%%P"
    if defined PORT (
        endlocal & set "PORT=%PORT%" & exit /b 0
    )

    for /f "tokens=1" %%A in ('arduino-cli board list 2^>nul ^| findstr /i "arduino:avr:mega"') do (
        set "PORT=%%A"
        goto :portFound
    )

:portFound
    endlocal & set "PORT=%PORT%"
    if not defined PORT exit /b 1
    exit /b 0

:uploadSketch
    @echo off
    setlocal EnableExtensions EnableDelayedExpansion

    set "SKETCH_DIR=%~1"
    if not defined SKETCH_DIR (
        echo [ERROR] Missing folder argument for :uploadSketch subroutine
        endlocal & exit /b 1
    )

    set "ABS_SKETCH_DIR=%~dp0%SKETCH_DIR%"
    if not exist "%ABS_SKETCH_DIR%\" (
        echo [ERROR] Folder not found: "%ABS_SKETCH_DIR%"
        endlocal & exit /b 1
    )

    set "INO="
    for %%I in ("%ABS_SKETCH_DIR%\*.ino") do (
        set "INO=%%~fI"
        goto :gotINO
    )

:gotINO
    if not defined INO (
        echo [ERROR] No .ino files found in "%ABS_SKETCH_DIR%"
        endlocal & exit /b 1
    )

    if not defined FQBN (
        echo [ERROR] FQBN not set
        endlocal & exit /b 1
    )

    if not defined PORT (
        echo [ERROR] PORT not set
        endlocal & exit /b 1
    )

    echo Compiling sketch folder...
    "%ARDUINO_CLI%" compile --fqbn "%FQBN%" "%ABS_SKETCH_DIR%" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Sketch compilation failed
        endlocal & exit /b 1
    )

    echo Uploading sketch...
    "%ARDUINO_CLI%" upload -p "%PORT%" --fqbn "%FQBN%" "%ABS_SKETCH_DIR%" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Sketch upload failed
        endlocal & exit /b 1
    )

    endlocal & exit /b 0

:sleep
    python -c "import time; time.sleep(%1)"
    exit /b 0

:kill
    echo.
    echo Fatal error: %~1
    echo.
    endlocal
    pause
    exit 1

:eof
    exit /b 0
    pause
