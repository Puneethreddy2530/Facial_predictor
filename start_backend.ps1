# Quick starter - Run backend only
$env:MOCK_PREDICTION = '1'
$env:PYTHONPATH = '.'

Write-Host "Starting backend on http://localhost:8001..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

C:/Python314/python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
