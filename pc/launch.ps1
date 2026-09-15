#Requires -Version 5.1

<#
    run.ps1

    Behavioral training setup/launch script
#>

# ======================================================================
# CONFIGURATION
# ======================================================================
$script:FQBN = "arduino:avr:mega"
$script:ARDUINO_CLI = "arduino-cli"

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
function Exit-Fatal {
    param([string]$Message)

    Write-Host ""
    Write-Host "[FATAL] $Message"
    Write-Host ""

    Read-Host "Press Enter to exit" | Out-Null
    exit 1
}

function Initialize-ArduinoCli {
    Write-Host "Verifying arduino-cli installation..."

    if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) {
        Write-Host "arduino-cli not found, attempting to install..."

        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id ArduinoSA.CLI -e --source winget --accept-package-agreements --accept-source-agreements *> $null
        }

        # Add common install locations to PATH
        if (Test-Path "$env:ProgramFiles\Arduino CLI\arduino-cli.exe") {
            $env:PATH = "$env:ProgramFiles\Arduino CLI;$env:PATH"
        }

        if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) { return $false }
    }

    # Initialize configurations, make sure the AVR core and Servo library are present
    arduino-cli config init *> $null
    arduino-cli core update-index *> $null

    $coreList = arduino-cli core list 2>$null
    if (-not ($coreList | Select-String -Pattern "arduino:avr" -Quiet)) {
        arduino-cli core install arduino:avr *> $null
    }

    arduino-cli lib install Servo *> $null

    return $true
}

function Find-ArduinoPort {
    try {
        $out = arduino-cli board list --format json 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $json = $out | ConvertFrom-Json
            $match = $json.ports | Where-Object { $_.matching_boards.fqbn -contains "arduino:avr:mega" } |
                Select-Object -First 1

            if ($match) { return $match.address }
        }
    }
    catch { }

    $textList = arduino-cli board list 2>$null
    $line = $textList | Select-String -Pattern "arduino:avr:mega" -SimpleMatch | Select-Object -First 1

    if ($line) {
        $firstToken = ($line.Line -split '\s+')[0]

        if ($firstToken) { return $firstToken }
    }

    return $null
}

function Write-Sketch {
    param([string]$SketchFolderName)

    $sketchDir = Join-Path $script:SCRIPT_DIR $SketchFolderName

    if (-not $SketchFolderName) {
        Write-Host "[ERROR] Missing sketch folder argument"
        return $false
    }
    if (-not (Test-Path $sketchDir -PathType Container)) {
        Write-Host "[ERROR] Folder not found: `"$sketchDir`""
        return $false
    }

    $ino = Get-ChildItem -Path $sketchDir -Filter "*.ino" -File | Select-Object -First 1

    if (-not $ino) {
        Write-Host "[ERROR] No .ino file found in `"$sketchDir`""
        return $false
    }

    if (-not $script:FQBN) {
        Write-Host "[ERROR] FQBN not set"
        return $false
    }
    if (-not $script:PORT) {
        Write-Host "[ERROR] PORT not set"
        return $false
    }

    Write-Host "Compiling Arduino sketch..."
    & $script:ARDUINO_CLI compile --fqbn $script:FQBN $sketchDir *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Sketch compilation failed"
        return $false
    }

    Write-Host "Uploading Arduino sketch..."
    & $script:ARDUINO_CLI upload --port $script:PORT --fqbn $script:FQBN $sketchDir *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Sketch upload failed"
        return $false
    }

    return $true
}

# ======================================================================
# MAIN
# ======================================================================
Clear-Host
Start-Sleep -Seconds 2

Write-Host "Resolving script directory..."
$script:SCRIPT_DIR = $PSScriptRoot
Set-Location $script:SCRIPT_DIR
Start-Sleep -Seconds 1

$script:DO_UPDATE = Read-Host "Install/update required Python packages? [y/N]"
if (-not $script:DO_UPDATE) {
    $script:DO_UPDATE = "N"
}
Write-Host ""

if ($script:DO_UPDATE -ieq "Y") {
    Write-Host "Making sure pip is up to date..."
    python -m pip install --upgrade pip -q
    Start-Sleep -Seconds 1

    Write-Host "Installing required packages..."
    $requirementsPath = Join-Path $script:SCRIPT_DIR ".\requirements.txt"
    if (-not (Test-Path $requirementsPath)) {
        Write-Host "[ERROR] requirements.txt not found at $requirementsPath"
        Exit-Fatal "requirements.txt missing"
    }

    python -m pip install -r $requirementsPath -q
    if ($LASTEXITCODE -ne 0) {
        Exit-Fatal "pip install failed"
    }
    Start-Sleep -Seconds 1
}
Start-Sleep -Seconds 1

if (-not (Initialize-ArduinoCli)) {
        Exit-Fatal "arduino-cli installation failed or not found on PATH"
    }

Write-Host "Searching for Arduino..."
$script:PORT = Find-ArduinoPort
if (-not $script:PORT) {
    Write-Host "[ERROR] No Arduino detected"
    Exit-Fatal "Arduino not detected"
}

if (-not (Write-Sketch -SketchFolderName "behavioral_controller")) {
    Exit-Fatal "Arduino compile/upload failed"
}
Start-Sleep -Seconds 1

Write-Host "Running Python script..."
python -m behavioral_master
Start-Sleep -Seconds 1
Write-Host ""
