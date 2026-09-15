@echo off

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Unblock-File -LiteralPath '%SCRIPT_DIR%\launcher.ps1' -ErrorAction SilentlyContinue"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\launcher.ps1" %*

pause
