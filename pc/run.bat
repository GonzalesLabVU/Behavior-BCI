@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo.

call :selfUpdate
if "%ERRORLEVEL%"=="99" exit /b 0
if errorlevel 1 call :kill "selfUpdate failed"

echo.
echo Getting script directory...
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Making sure pip is up to date...
python -m pip install --upgrade pip -q

<nul set /p "=Downloading latest file versions..."
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/behavioral_master.py" || call :kill "pullFile subroutine failed for behavioral_master.py"
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/cursor_utils.py"
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/plot_utils.py"
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/run.bat"
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/config/animal_map.json"
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/config/requirements.txt"
call :pullFile "https://github.com/GonzalesLabVU/Behavior-BCI/blob/main/pc/config/errors.log"
call :pullFolder "https://github.com/GonzalesLabVU/Behavior-BCI/tree/main/arduino/behavioral_controller" || call :kill "pullFolder subroutine failed"
echo done

echo Installing required dependencies...
if not exist "requirements.txt" (
    echo requirements.txt not found in %SCRIPT_DIR%
    exit /b 1
)
python -m pip install -r requirements.txt -q || exit /b 1

echo Running Python script...
python -m behavioral_master
call :sleep 5

echo.
echo Session finished
echo.

goto :eof

REM ------------------------
REM Subroutines
REM ------------------------

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

:sleep
    python -c "import time; time.sleep(%1)"
    exit /b 0

:kill
    echo.
    echo Downloading latest file versions...failed
    echo Fatal error: %~1
    echo.
    endlocal
    pause
    exit 1

pause
