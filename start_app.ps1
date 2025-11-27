# Facial Prediction App Launcher
# Starts backend and frontend servers

Write-Host "=== Facial Prediction App Launcher ===" -ForegroundColor Cyan
Write-Host ""

# Set mock mode
$env:MOCK_PREDICTION = '1'
$env:PYTHONPATH = "."

# Check if backend is already running
$backendRunning = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($backendRunning) {
    Write-Host "Backend already running on port 8000" -ForegroundColor Yellow
} else {
    Write-Host "Starting backend server (mock mode)..." -ForegroundColor Green
    $backendJob = Start-Job -ScriptBlock {
        param($pythonPath, $workDir)
        Set-Location $workDir
        $env:MOCK_PREDICTION = '1'
        $env:PYTHONPATH = "."
        & $pythonPath -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
    } -ArgumentList "C:/Python314/python.exe", $PWD.Path
    
    Write-Host "Waiting for backend to start..." -ForegroundColor Cyan
    Start-Sleep -Seconds 3
}

# Check if frontend is already running
$frontendRunning = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($frontendRunning) {
    Write-Host "Frontend already running on port 3000" -ForegroundColor Yellow
} else {
    Write-Host "Starting frontend server..." -ForegroundColor Green
    $frontendJob = Start-Job -ScriptBlock {
        param($pythonPath, $workDir)
        Set-Location "$workDir\frontend"
        & $pythonPath -m http.server 3000
    } -ArgumentList "C:/Python314/python.exe", $PWD.Path
    
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "=== App is running! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Frontend:  http://localhost:3000" -ForegroundColor Cyan
Write-Host "Backend:   http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Mode: MOCK (returns sample data)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop servers" -ForegroundColor Gray
Write-Host ""

# Open browser
Write-Host "Opening browser..." -ForegroundColor Cyan
Start-Process "http://localhost:3000"

# Keep script running and show logs
Write-Host "=== Server Logs ===" -ForegroundColor Cyan
try {
    while ($true) {
        Start-Sleep -Seconds 5
        # Check if servers are still running
        $backend = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
        $frontend = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
        
        if (-not $backend) {
            Write-Host "Backend stopped!" -ForegroundColor Red
            break
        }
        if (-not $frontend) {
            Write-Host "Frontend stopped!" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host "`nStopping servers..." -ForegroundColor Yellow
    if ($backendJob) { Stop-Job $backendJob; Remove-Job $backendJob }
    if ($frontendJob) { Stop-Job $frontendJob; Remove-Job $frontendJob }
    Write-Host "Servers stopped." -ForegroundColor Green
}
