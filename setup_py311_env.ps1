# Setup Python 3.11 environment for DeepFace
# Prerequisites: Install Python 3.11 from https://www.python.org/downloads/

# Check if Python 3.11 is available
$py311Paths = @(
    "C:\Python311\python.exe",
    "C:\Program Files\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)

$pythonExe = $null
foreach ($path in $py311Paths) {
    if (Test-Path $path) {
        $pythonExe = $path
        Write-Host "Found Python 3.11 at: $path" -ForegroundColor Green
        break
    }
}

if (-not $pythonExe) {
    Write-Host "Python 3.11 not found. Please install from:" -ForegroundColor Red
    Write-Host "https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or specify path manually:" -ForegroundColor Yellow
    $customPath = Read-Host "Enter Python 3.11 path (or press Enter to exit)"
    if ($customPath -and (Test-Path $customPath)) {
        $pythonExe = $customPath
    } else {
        exit 1
    }
}

# Verify version
Write-Host "Verifying Python version..." -ForegroundColor Cyan
& $pythonExe --version

# Create virtual environment
Write-Host "`nCreating virtual environment (.venv311)..." -ForegroundColor Cyan
& $pythonExe -m venv .venv311

# Activate and install packages
Write-Host "`nActivating environment and installing packages..." -ForegroundColor Cyan
Write-Host "This may take several minutes as TensorFlow is large..." -ForegroundColor Yellow

$activateScript = ".\.venv311\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    # Install DeepFace (will pull TensorFlow)
    pip install deepface
    
    Write-Host "`n✓ Setup complete!" -ForegroundColor Green
    Write-Host "`nTo use this environment:" -ForegroundColor Cyan
    Write-Host "  .\.venv311\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "`nTo run backend with real predictions:" -ForegroundColor Cyan
    Write-Host "  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor White
} else {
    Write-Host "Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
