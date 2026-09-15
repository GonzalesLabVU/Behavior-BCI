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
$script:CORE_HEADERS = @("Wire.h", "SPI.h", "EEPROM.h", "SoftwareSerial.h")
$script:HEADER_TO_LIBRARY = @{
    "Servo.h" = "Servo"
}

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
function Exit-Fatal {
    param([string]$Message)

    Write-Host ""
    Write-Host "[FATAL] $Message"
    Write-Host ""

    exit 1
}

function Get-MissingHeaders {
    param([string[]]$CompileOutput)

    $pattern = "fatal error:\s*([A-Za-z0-9_]+\.h):\s*No such file or directory"
    $headers = foreach ($line in $CompileOutput) {
        if ($line -match $pattern) { $matches[1] }
    }

    return $headers | Select-Object -Unique
}

function Test-LibraryInstalled {
    param([string]$LibraryName)

    try {
        $out = arduino-cli lib list --format json 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) {
            return $false
        }
        $json = $out | ConvertFrom-Json
        $match = $json.installed_libraries | Where-Object { $_.library.name -eq $LibraryName }

        return [bool]$match
    }
    catch {
        return $false
    }
}

function Show-LibraryDiagnostics {
    param([string]$SketchDir)

    Write-Host "[DIAG] Full arduino-cli library list:"
    arduino-cli lib list 2>&1 | ForEach-Object { Write-Host "  $_" }

    Write-Host "[DIAG] arduino-cli config dump:"
    arduino-cli config dump 2>&1 | ForEach-Object { Write-Host "  $_" }

    foreach ($sketchConfigName in @("sketch.yaml", "sketch.json")) {
        $sketchConfigPath = Join-Path $SketchDir $sketchConfigName
        if (Test-Path $sketchConfigPath) {
            Write-Host "[DIAG] Found '$sketchConfigName' in sketch folder - contents:"
            Get-Content $sketchConfigPath | ForEach-Object { Write-Host "  $_" }
            Write-Host "[DIAG] A sketch-level profile can pin its own library list, overriding whatever's globally installed"
        }
    }
}

function Resolve-MissingLibrary {
    param([string]$HeaderName)

    if ($script:CORE_HEADERS -contains $HeaderName) {
        Write-Host "[WARNING] '$HeaderName' should ship with the arduino:avr core - reinstalling won't help"
        return $false
    }

    if (-not $script:HEADER_TO_LIBRARY.ContainsKey($HeaderName)) {
        Write-Host "[WARNING] No known library mapping for missing header '$HeaderName' - add one to `$script:HEADER_TO_LIBRARY"
        return $false
    }

    $libName = $script:HEADER_TO_LIBRARY[$HeaderName]
    Write-Host "  Attempting to install library '$libName' for missing header '$HeaderName'..."
    $installOutput = arduino-cli lib install $libName 2>&1
    $installOutput | ForEach-Object { Write-Host "  $_" }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] arduino-cli reported a failure installing '$libName'"
        return $false
    }

    if (-not (Test-LibraryInstalled -LibraryName $libName)) {
        Write-Host "[WARNING] arduino-cli reported success, but '$libName' isn't appearing in 'arduino-cli lib list'"
        Write-Host "[WARNING] Diagnostics:"
        arduino-cli lib list 2>&1 | ForEach-Object { Write-Host "  $_" }
        arduino-cli config dump 2>&1 | ForEach-Object { Write-Host "  $_" }
        return $false
    }

    return $true
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

    arduino-cli lib update-index *> $null
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
    $maxAttempts = 3
    $attempt = 0
    $compileSucceeded = $false
    $lastOutput = $null
    $alreadyResolved = @{}

    while ($attempt -lt $maxAttempts -and -not $compileSucceeded) {
        $attempt++
        $compileOutput = & $script:ARDUINO_CLI compile --fqbn $script:FQBN $sketchDir 2>&1
        $lastOutput = $compileOutput

        if ($LASTEXITCODE -eq 0) {
            $compileSucceeded = $true
            break
        }

        $missingHeaders = Get-MissingHeaders -CompileOutput $compileOutput
        if ($missingHeaders.Count -eq 0) {
            break
        }

        Write-Host "[INFO] Compile attempt $attempt failed due to missing header(s): $($missingHeaders -join ', ')"

        $resolvedAny = $false
        foreach ($header in $missingHeaders) {
            if ($alreadyResolved.ContainsKey($header)) {
                Write-Host "[WARNING] '$header' was already confirmed installed on a previous attempt, but the compiler still can't find it"
                Write-Host "[WARNING] Reinstalling it again won't help - here's the environment state:"
                Show-LibraryDiagnostics -SketchDir $sketchDir
                continue
            }

            if (Resolve-MissingLibrary -HeaderName $header) {
                $resolvedAny = $true
                $alreadyResolved[$header] = $true
            }
        }

        if (-not $resolvedAny) { break }
    }

    if (-not $compileSucceeded) {
        Write-Host "[ERROR] Sketch compilation failed"
        $lastOutput | ForEach-Object { Write-Host $_ }
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

Write-Host ""
$script:DO_UPDATE = Read-Host "Install/update required Python packages? [y/N]"
if (-not $script:DO_UPDATE) {
    $script:DO_UPDATE = "N"
}
Write-Host ""

if ($script:DO_UPDATE -ieq "Y") {
    Write-Host "Making sure pip is up to date..."
    python -m pip install --upgrade pip -q

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
}

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

Write-Host "Running Python script..."
python -m behavioral_master
Start-Sleep -Seconds 1
Write-Host ""
