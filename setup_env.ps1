<#
setup_env.ps1

PowerShell helper to create and populate a virtual environment for this
project on Windows. It:
 - creates a venv at .venv
 - activates it for the duration of the script
 - upgrades pip and installs runtime requirements
 - optionally installs dev requirements

Usage (PowerShell):
  .\setup_env.ps1            # interactive prompt to install dev deps
  .\setup_env.ps1 -NoDev    # skip dev deps
#>
param(
    [switch]$NoDev
)

$ErrorActionPreference = 'Stop'
$venvPath = Join-Path -Path (Get-Location) -ChildPath '.venv'

# Create the venv if it doesn't already exist. Prefer 'python', fall back to 'py'.
if (-Not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
    Write-Host "Creating virtual environment at $venvPath" -ForegroundColor Cyan
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "Using 'python' to create the venv" -ForegroundColor Cyan
        & python -m venv $venvPath
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host "'python' not found; using 'py -3' to create the venv" -ForegroundColor Cyan
        & py -3 -m venv $venvPath
    } else {
        Write-Error "Neither 'python' nor 'py' was found in PATH. Please install Python and try again."
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists at $venvPath" -ForegroundColor DarkGray
}

$activate = Join-Path $venvPath 'Scripts\Activate.ps1'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

# Try to activate for user convenience, but don't fail if activation script is missing
if (Test-Path $activate) {
    try {
        Write-Host "Activating venv..." -ForegroundColor Cyan
        . $activate
    } catch {
        Write-Warning "Couldn't activate the venv (possibly due to execution policy). We'll continue by calling the venv's Python directly. To activate manually later, run: . $activate"
    }
} else {
    Write-Warning "Activation script not found at $activate. Continuing without activation; will use the venv's Python directly."
}

Write-Host "Upgrading pip and installing runtime dependencies..." -ForegroundColor Cyan
# Prefer to call the venv's python directly to ensure packages are installed into the created venv.
if (-Not (Test-Path $venvPython)) {
    # If the python executable isn't found at the expected path, fall back to the (possibly activated) 'python' in PATH
    $venvPython = 'python'
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

if (-Not $NoDev) {
    $installDev = Read-Host "Install dev requirements (black/isort/pytest)? [Y/n]"
    if ($installDev -eq '' -or $installDev -match '^[Yy]') {
        Write-Host "Installing dev requirements..." -ForegroundColor Cyan
        & $venvPython -m pip install -r dev-requirements.txt
    } else {
        Write-Host "Skipping dev requirements install" -ForegroundColor Yellow
    }
} else {
    Write-Host "NoDev flag specified; skipping dev requirements" -ForegroundColor Yellow
}

# Final message with manual activation hint and execution policy tip
Write-Host "Setup complete." -ForegroundColor Green
if (Test-Path $activate) {
    Write-Host "To activate the venv in this shell run:" -ForegroundColor Green
    Write-Host ". ./.venv/Scripts/Activate.ps1" -ForegroundColor Green
} else {
    Write-Host "Note: Activation script was not found. You can still use the environment via:$([environment]::NewLine)  $venvPython <script.py>" -ForegroundColor Yellow
}
Write-Host "If PowerShell blocks activation, you can temporarily allow it for this session with:" -ForegroundColor DarkGray
Write-Host "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor DarkGray
