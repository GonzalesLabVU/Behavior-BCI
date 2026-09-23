#Requires -Version 5.1

<#
    run.ps1

    Behavioral training setup/launch script
#>

# ======================================================================
# CONFIGURATION
# ======================================================================
$script:PYTHON_VERSION = "3.12"
$script:ARDUINO_CLI = "arduino-cli"
$script:CANDIDATE_BOARDS = @(
    [PSCustomObject]@{ Fqbn = "arduino:avr:mega"; Profile = "mega2560" },
    [PSCustomObject]@{ Fqbn = "arduino:avr:uno"; Profile = "uno" }
)
$script:CORE_HEADERS = @("Wire.h", "SPI.h", "EEPROM.h", "SoftwareSerial.h")
$script:HEADER_TO_LIBRARY = @{
    "Servo.h" = "Servo"
}

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
function Exit-Fatal {
    param([string]$Message)

    throw $Message
}

function Check-Python310Available {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        return $false
    }

    & py "-$script:PYTHON_VERSION" --version *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-VenvPythonVersion {
    if (-not (Test-Path $script:VENV_PYTHON)) {
        return $false
    }

    $pyvenvCfg = Join-Path $script:VENV_DIR "pyvenv.cfg"
    if (-not (Test-Path $pyvenvCfg)) {
        return $false
    }

    $versionLine = Get-Content $pyvenvCfg |
        Where-Object { $_ -match '^\s*version\s*=' } | 
        Select-Object -First 1
    if (-not $versionLine) {
        return $false
    }

    return ($versionLine -match [regex]::Escape($script:PYTHON_VERSION))
}

function Initialize-PythonVenv {
    if (Test-VenvPythonVersion) {
        return $true
    }

    if (-not (Check-Python310Available)) {
        Write-Host "Python $($script:PYTHON_VERSION) not found; attempting to install..."

        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Python.Python.$($script:PYTHON_VERSION) -e --source winget `
                --accept-package-agreements `
                --accept-source-agreements *> $null
        }

        if (-not (Check-Python310Available)) {
            Write-Host "[ERROR] Python $($script:PYTHON_VERSION) still isn't available via the 'py' launcher"
            return $false
        }
    }

    if (Test-Path $script:VENV_DIR) {
        Write-Host "Existing virtual environment doesn't match Python $($script:PYTHON_VERSION); rebuilding virtual environment..."
        Remove-Item -Recurse -Force $script:VENV_DIR
    }
    else {
        Write-Host "Creating Python $($script:PYTHON_VERSION) virtual environment..."
    }

    & py "-$script:PYTHON_VERSION" -m venv $script:VENV_DIR
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $script:VENV_PYTHON)) {
        Write-Host "[ERROR] Failed to create virtual environment at $($script:VENV_DIR)"
        return $false
    }

    return $true
}

function Check-PySpinAvailable {
    & $script:VENV_PYTHON -c "import PySpin" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-ExpectedPySpinTag {
    return "cp" + ($script:PYTHON_VERSION -replace '\.', '')
}

function Install-PySpinWheel {
    if (Check-PySpinAvailable) {
        return $true
    }

    $expectedTag = Get-ExpectedPySpinTag
    $candidateWheels = Get-ChildItem -Path (Join-Path $script:SCRIPT_DIR "spinnaker_python-*.whl") -File -ErrorAction SilentlyContinue

    if (-not $candidateWheels) {
        return $false
    }

    $matchingWheel = $candidateWheels |
        Where-Object { $_.Name -match [regex]::Escape($expectedTag) -and $_.Name -match "win_amd64" } |
        Select-Object -First 1

    if (-not $matchingWheel) {
        Write-Host "[WARNING] Found PySpin wheel(s) in the project root, but none match Python $($script:PYTHON_VERSION) ($expectedTag/win_amd64):"
        foreach ($wheel in $candidateWheels) {
            Write-Host "`t$($wheel.Name)"
        }

        return $false
    }

    Write-Host "`tFound matching PySpin wheel: $($matchingWheel.Name)"
    Write-Host "Installing PySpin..."
    & $script:VENV_PYTHON -m pip install $matchingWheel.FullName -q --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Failed to install $($matchingWheel.Name)"
        return $false
    }

    if (-not (Check-PySpinAvailable)) {
        Write-Host "[WARNING] Installed $($matchingWheel.Name), but PySpin still isn't importable"
        return $false
    }

    return $true
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
    arduino-cli lib list 2>&1 | ForEach-Object { Write-Host "`t$_" }

    Write-Host "[DIAG] arduino-cli config dump:"
    arduino-cli config dump 2>&1 | ForEach-Object { Write-Host "`t$_" }

    foreach ($sketchConfigName in @("sketch.yaml", "sketch.json")) {
        $sketchConfigPath = Join-Path $SketchDir $sketchConfigName
        if (Test-Path $sketchConfigPath) {
            Write-Host "[DIAG] Found '$sketchConfigName' in sketch folder - contents:"
            Get-Content $sketchConfigPath | ForEach-Object { Write-Host "`t$_" }
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
    Write-Host "`tAttempting to install library '$libName' for missing header '$HeaderName'..."
    $installOutput = arduino-cli lib install $libName 2>&1
    $installOutput | ForEach-Object { Write-Host "`t$_" }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] arduino-cli reported a failure installing '$libName'"
        return $false
    }

    if (-not (Test-LibraryInstalled -LibraryName $libName)) {
        Write-Host "[WARNING] arduino-cli reported success, but '$libName' isn't appearing in 'arduino-cli lib list'"
        Write-Host "[WARNING] Diagnostics:"
        arduino-cli lib list 2>&1 | ForEach-Object { Write-Host "`t$_" }
        arduino-cli config dump 2>&1 | ForEach-Object { Write-Host "`t$_" }
        return $false
    }

    return $true
}

function Initialize-ArduinoCli {
    Write-Host "Verifying arduino-cli installation..."

    if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) {
        Write-Host "arduino-cli not found, attempting to install..."

        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id ArduinoSA.CLI -e --source winget `
                --accept-package-agreements `
                --accept-source-agreements *> $null
        }

        if (Test-Path "$env:ProgramFiles\Arduino CLI\arduino-cli.exe") {
            $env:PATH = "$env:ProgramFiles\Arduino CLI;$env:PATH"
        }

        if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) {
            return $false
        }
    }

    # initialize config only if it does not exist
    $arduinoConfig = Join-Path $env:LOCALAPPDATA "Arduino15\arduino-cli.yaml"
    if (-not (Test-Path $arduinoConfig)) {
        Write-Host "Arduino CLI not found; initializing..."
        arduino-cli config init *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to initialize Arduino CLI configuration"
            return $false
        }
    }

    # verify AVR core initialization locally
    $coreList = arduino-cli core list 2>$null

    if (-not ($coreList | Select-String -Pattern "^arduino:avr\s" -q)) {
        Write-Host "Arduino AVR core not found; updating index and installing..."

        arduino-cli core update-index *> $null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }

        arduino-cli core install arduino:avr *> $null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
    }

    # verify/install libraries specified in sketch.yaml
    $sketchDir = Join-Path $script:SCRIPT_DIR "behavioral_controller"
    $sketchYamlPath = Join-Path $sketchDir "sketch.yaml"
    if (-not (Test-Path $sketchYamlPath)) {
        Write-Host "[ERROR] sketch.yaml not found at $sketchYamlPath"
        return $false
    }

    $requiredLibraries = @()
    $inLibrariesSection = $false
    $librariesIndent = 0

    foreach ($line in Get-Content $sketchYamlPath) {
        if ($line -match '^(\s*)libraries:\s*$') {
            $inLibrariesSection = $true
            $librariesIndent = $matches[1].length
            continue
        }

        if ($inLibrariesSection) {
            if ($line -match '^\s*(#.*)?$') {
                continue
            }

            $currentIndent = ([regex]::Match($line, '^\s*')).Value.length
            if ($currentIndent -le $librariesIndent) {
                $inLibrariesSection = $false
            }
            elseif ($line -match '^\s*-\s+(.+?)\s*$') {
                $libraryEntry = $matches[1].Trim()

                if ($libraryEntry -notmatch '^dir:\s*') {
                    if ($libraryEntry -match '^dependency:\s*(.+)$') {
                        $libraryEntry = $matches[1].Trim()
                    }

                    $requiredLibraries += $libraryEntry
                }
            }
        }
    }

    $requiredLibraries = $requiredLibraries | Sort-Object -Unique

    if ($requiredLibraries.Count -gt 0) {

        $installedLibraryOutput = arduino-cli lib list --format json 2>$null

        if ($LASTEXITCODE -ne 0 -or -not $installedLibraryOutput) {
            Write-Host "[ERROR] Unable to query installed Arduino libraries"
            return $false
        }

        try {
            $installedLibraryJson = $installedLibraryOutput | ConvertFrom-Json
        }
        catch {
            Write-Host "[ERROR] Unable to parse Arduino library list"
            return $false
        }

        $missingLibraries = @()

        foreach ($requiredLibrary in $requiredLibraries) {
            if ($requiredLibrary -match '^(.*?)\s+\(([^)]+)\)\s*$') {
                $libraryName = $matches[1].Trim()
                $requiredVersion = $matches[2].Trim()
            }
            else {
                $libraryName = $requiredLibrary.Trim()
                $requiredVersion = $null
            }

            $installedMatch = $installedLibraryJson.installed_libraries |
                Where-Object { $_.library.name -eq $libraryName } |
                    Select-Object -First 1

            if (-not $installedMatch) {
                $missingLibraries += $requiredLibrary
                continue
            }

            if ($requiredVersion) {
                $installedVersion = $installedMatch.library.version

                if ($installedVersion -ne $requiredVersion) {
                    $missingLibraries += $requiredLibrary
                }
            }
        }

        if ($missingLibraries.Count -gt 0) {

            Write-Host "Required Arduino libraries missing or incorrect version:"
            foreach ($library in $missingLibraries) {
                Write-Host "`t$library"
            }

            Write-Host "Updating Arduino library index..."
            arduino-cli lib update-index *> $null

            if ($LASTEXITCODE -ne 0) {
                return $false
            }

            foreach ($library in $missingLibraries) {
                if ($library -match '^(.*?)\s+\(([^)]+)\)\s*$') {
                    $libraryName = $matches[1].Trim()
                    $libraryVersion = $matches[2].Trim()
                    $installSpec = "${libraryName}@${libraryVersion}"
                }
                else {
                    $libraryName = $library.Trim()
                    $installSpec = $libraryName
                }

                Write-Host "Installing $installSpec..."
                arduino-cli lib install $installSpec *> $null

                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[ERROR] Failed to install '$installSpec'"
                    return $false
                }
            }
        }
    }

    return $true
}

function Find-ArduinoBoard {
    try {
        $out = arduino-cli board list --format json 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $json = $out | ConvertFrom-Json
            foreach ($port in $json.ports) {
                foreach ($candidate in $script:CANDIDATE_BOARDS) {
                    if ($port.matching_boards.fqbn -contains $candidate.Fqbn) {
                        return [PSCustomObject]@{
                            Port = $port.address
                            Fqbn = $candidate.Fqbn
                            Profile = $candidate.Profile
                        }
                    }
                }
            }
        }
    }
    catch { }

    $textList = arduino-cli board list 2>$null
    foreach ($candidate in $script:CANDIDATE_BOARDS) {
        $line = $textList | Select-String -Pattern $candidate.Fqbn -SimpleMatch | Select-Object -First 1
        if ($line) {
            $firstToken = ($line.Line -split '\s+')[0]
            if ($firstToken) {
                return [PSCustomObject]@{ Port = $firstToken; Fqbn = $candidate.Fqbn; Profile = $candidate.Profile }
            }
        }
    }

    return $null
}

function Write-Sketch {
    param(
        [string]$SketchFolderName,
        [string]$ProfileName
    )

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
    if (-not $ProfileName) {
        Write-Host "[ERROR] No sketch.yaml profile resolved for the connected board"
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
        $compileOutput = & $script:ARDUINO_CLI compile --profile $ProfileName $sketchDir 2>&1
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
    $uploadOutput = & $script:ARDUINO_CLI upload --port $script:PORT --profile $ProfileName $sketchDir 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Sketch upload failed"
        $uploadOutput | ForEach-Object { Write-Host $_ }
        return $false
    }

    return $true
}

# ======================================================================
# MAIN
# ======================================================================
$script:ExitCode = 0

try {
    Clear-Host
    Write-Host "`nResolving script directory..."
    $script:SCRIPT_DIR = $PSScriptRoot
    Set-Location $script:SCRIPT_DIR

    Write-Host "Validating virtual environment..."
    $script:VENV_DIR = Join-Path $script:SCRIPT_DIR ".venv"
    $script:VENV_PYTHON = Join-Path $script:VENV_DIR "Scripts\python.exe"

    if (-not (Initialize-PythonVenv)) {
        Exit-Fatal "Python $($script:PYTHON_VERSION) virtual environment setup failed"
    }

    Write-Host "Making sure pip is up to date..."
    & $script:VENV_PYTHON -m pip install --upgrade pip -q --disable-pip-version-check

    if ($LASTEXITCODE -ne 0) {
        Exit-Fatal "pip upgrade failed"
    }

    Write-Host "Installing required packages..."
    $requirementsPath = Join-Path $script:SCRIPT_DIR ".\requirements.txt"
    if (-not (Test-Path $requirementsPath)) {
        Write-Host "[ERROR] requirements.txt not found at $requirementsPath"
        Exit-Fatal "requirements.txt missing"
    }

    $requirementsLines = Get-Content $requirementsPath |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith("#") }

    $displayByName = @{}
    foreach ($requirement in $requirementsLines) {
        if ($requirement -match '^([A-Za-z0-9_.\-]+)\s*==\s*(.+)$') {
            $packageName = $matches[1]
            $packageVersion = $matches[2]
            $displayByName[$packageName.ToLowerInvariant()] = "$packageName ($packageVersion)"
        }
        elseif ($requirement -match '^([A-Za-z0-9_.\-]+)') {
            $displayByName[$matches[1].ToLowerInvariant()] = $requirement
        }
    }

    & $script:VENV_PYTHON -m pip install -r $requirementsPath --disable-pip-version-check 2>&1 |
        ForEach-Object {
            if ($_ -match '^(?:Collecting|Requirement already satisfied:)\s+([A-Za-z0-9_.\-]+)') {
                $pkgKey = $matches[1].ToLowerInvariant()
                if ($displayByName.ContainsKey($pkgKey)) {
                    Write-Host "`t$($displayByName[$pkgKey])"
                }
            }
        }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Batched install failed; retrying package-by-package to isolate the failure..."

        foreach ($requirement in $requirementsLines) {
            & $script:VENV_PYTHON -m pip install $requirement -q --disable-pip-version-check
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Failed to install '$requirement'"
                Exit-Fatal "pip install failed"
            }
        }
    }

    Write-Host "Checking for PySpin installation..."
    if (-not (Install-PySpinWheel)) {
        Write-Host "[WARNING] PySpin isn't importable in this venv"
    }

    if (-not (Initialize-ArduinoCli)) {
            Exit-Fatal "arduino-cli installation failed or not found on PATH"
        }

    Write-Host "Searching for Arduino Mega/Uno..."
    $script:BOARD = Find-ArduinoBoard
    if (-not $script:BOARD) {
        Write-Host "[ERROR] No supported Arduino board detected"
        Exit-Fatal "Arduino not detected"
    }
    $script:PORT = $script:BOARD.Port
    Write-Host "`tFound $($script:BOARD.Fqbn) on $($script:PORT) - using profile '$($script:BOARD.Profile)'"

    if (-not (Write-Sketch -SketchFolderName "behavioral_controller" -ProfileName $script:BOARD.Profile)) {
        Exit-Fatal "Arduino compile/upload failed"
    }

    Write-Host "Running Python script..."
    Start-Sleep -Milliseconds 500
    Clear-Host
    & $script:VENV_PYTHON -m behavioral_master

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[WARNING] Python script exited with a non-zero exit code ($LASTEXITCODE)"
    }
    Start-Sleep -Seconds 1
    Write-Host ""
}
catch {
    Write-Host "[FATAL] $($_.Exception.Message)"
    $script:ExitCode = 1
}
finally {
    Write-Host "`nPress Enter to continue . . ."
    Read-Host | Out-Null
}

exit $script:ExitCode
