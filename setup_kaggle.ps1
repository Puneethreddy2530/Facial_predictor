Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kaggle API Setup for Dataset Download" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "STEP 1: Get your Kaggle API credentials" -ForegroundColor Yellow
Write-Host "1. Go to: https://www.kaggle.com/settings/account" -ForegroundColor White
Write-Host "2. Scroll to 'API' section" -ForegroundColor White
Write-Host "3. Click 'Create New Token'" -ForegroundColor White
Write-Host "4. Download kaggle.json file" -ForegroundColor White
Write-Host ""

Write-Host "STEP 2: Copy credentials" -ForegroundColor Yellow
$kaggleDir = "$env:USERPROFILE\.kaggle"
if (-not (Test-Path $kaggleDir)) {
    New-Item -ItemType Directory -Path $kaggleDir -Force | Out-Null
    Write-Host "✓ Created $kaggleDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Please paste the path to your downloaded kaggle.json file:" -ForegroundColor Cyan
Write-Host "(e.g., C:\Users\punee\Downloads\kaggle.json)" -ForegroundColor Gray
$sourcePath = Read-Host "Path"

if (Test-Path $sourcePath) {
    Copy-Item $sourcePath "$kaggleDir\kaggle.json" -Force
    Write-Host "✓ Kaggle credentials installed!" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "STEP 3: Downloading UTKFace dataset..." -ForegroundColor Yellow
    Write-Host "Dataset: ~500MB, 23,708 face images" -ForegroundColor Gray
    Write-Host ""
    
    # Download dataset
    Set-Location "C:\Users\punee\OneDrive\Desktop\Facial_prediction"
    python -c "import kaggle; kaggle.api.dataset_download_files('jangedoo/utkface-new', path='datasets/', unzip=True)"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Dataset downloaded successfully!" -ForegroundColor Green
        Write-Host "Location: datasets/utkface-new/" -ForegroundColor White
        Write-Host ""
        Write-Host "NEXT: Run training with real data" -ForegroundColor Yellow
        Write-Host "Command: python training\train_sklearn_model.py" -ForegroundColor Cyan
    } else {
        Write-Host "⚠ Download failed. You may need to accept dataset terms on Kaggle website first." -ForegroundColor Red
        Write-Host "Visit: https://www.kaggle.com/datasets/jangedoo/utkface-new" -ForegroundColor Cyan
    }
} else {
    Write-Host "⚠ File not found. Please download kaggle.json first." -ForegroundColor Red
    Write-Host "Visit: https://www.kaggle.com/settings/account" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
