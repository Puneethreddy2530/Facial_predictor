# Automated Python 3.11 installer for DeepFace

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Python 3.11 + DeepFace Installer"  -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Download Python 3.11.9
$pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$installerPath = "$env:TEMP\python-3.11.9-amd64.exe"

Write-Host "Downloading Python 3.11.9..." -ForegroundColor Yellow
Write-Host "URL: $pythonUrl" -ForegroundColor Gray
Write-Host ""

try {
    Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "Download complete!" -ForegroundColor Green
    Write-Host ""
    
    # Install Python silently
    Write-Host "Installing Python 3.11 to C:\Python311..." -ForegroundColor Yellow
    Write-Host "This may take 2-3 minutes..." -ForegroundColor Gray
    Write-Host ""
    
    $installArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_test=0",
        "TargetDir=C:\Python311"
    )
    
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
    
    Write-Host "Python 3.11 installed!" -ForegroundColor Green
    Write-Host ""
    
    # Verify installation
    if (Test-Path "C:\Python311\python.exe") {
        Write-Host "Verifying installation..." -ForegroundColor Yellow
        & C:\Python311\python.exe --version
        Write-Host ""
        
        # Create virtual environment
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        & C:\Python311\python.exe -m venv .venv311
        Write-Host "Done!" -ForegroundColor Green
        Write-Host ""
        
        # Install DeepFace and dependencies
        Write-Host "Installing DeepFace + dependencies..." -ForegroundColor Yellow
        Write-Host "This will take 5-10 minutes (downloading ~2GB)..." -ForegroundColor Gray
        Write-Host ""
        
        & .\.venv311\Scripts\pip.exe install --upgrade pip
        & .\.venv311\Scripts\pip.exe install deepface tf-keras tensorflow opencv-python fastapi uvicorn python-multipart pillow numpy
        
        Write-Host ""
        Write-Host "========================================"  -ForegroundColor Cyan
        Write-Host "  INSTALLATION COMPLETE!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Python 3.11 installed to: C:\Python311" -ForegroundColor Cyan
        Write-Host "Virtual environment: .venv311" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next: I will update backend to use DeepFace" -ForegroundColor Yellow
        Write-Host ""
        
    } else {
        Write-Host "ERROR: Installation failed!" -ForegroundColor Red
        Write-Host "Please install manually from: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install manually:" -ForegroundColor Yellow
    Write-Host "1. Download: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -ForegroundColor White
    Write-Host "2. Run installer, choose 'Install for all users'" -ForegroundColor White
    Write-Host "3. Set install path to: C:\Python311" -ForegroundColor White
    Write-Host "4. Run this script again" -ForegroundColor White
}

Write-Host ""
