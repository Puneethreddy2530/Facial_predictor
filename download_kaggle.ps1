# Download UTKFace from Kaggle using PowerShell
# More reliable for large files

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  UTKFace Dataset Downloader" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

# Set environment
$env:KAGGLE_API_TOKEN = 'KGAT_4fe35e813ba13d5c295451579b52c31f'

Write-Host "Downloading UTKFace dataset..." -ForegroundColor Yellow
Write-Host "   Size: ~400MB (23708 real face images)" -ForegroundColor Gray
Write-Host "   This may take 5-10 minutes..." -ForegroundColor Gray
Write-Host ""

try {
    # Use curl (built into Windows 10+) for more reliable download
    $url = "https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new"
    $output = "datasets\utkface-new.zip"
    
    # Create datasets directory
    New-Item -ItemType Directory -Force -Path "datasets" | Out-Null
    
    Write-Host "Downloading..." -ForegroundColor Cyan
    
    # Download using Invoke-WebRequest with resume support
    $headers = @{
        "Authorization" = "Bearer KGAT_4fe35e813ba13d5c295451579b52c31f"
    }
    
    Invoke-WebRequest -Uri $url -OutFile $output -Headers $headers -TimeoutSec 600 -MaximumRetryCount 5 -RetryIntervalSec 10
    
    Write-Host "`n✓ Download complete!" -ForegroundColor Green
    Write-Host "📦 Extracting..." -ForegroundColor Yellow
    
    # Extract
    Expand-Archive -Path $output -DestinationPath "datasets\" -Force
    
    # Check result
    $files = Get-ChildItem "datasets\UTKFace" -ErrorAction SilentlyContinue
    if ($files) {
        Write-Host "✅ Success! Extracted $($files.Count) files" -ForegroundColor Green
        Write-Host "📁 Location: datasets\UTKFace\" -ForegroundColor Cyan
    } else {
        Write-Host "⚠ Warning: UTKFace folder not found after extraction" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "`n❌ Download failed: $_" -ForegroundColor Red
    Write-Host "`nPlease download manually:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://www.kaggle.com/datasets/jangedoo/utkface-new" -ForegroundColor White
    Write-Host "2. Click 'Download' button" -ForegroundColor White
    Write-Host "3. Extract to: $PWD\datasets\" -ForegroundColor White
}

Write-Host "`n========================================`n" -ForegroundColor Cyan
