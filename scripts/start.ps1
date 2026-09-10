param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run .\scripts\setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'frontend\dist\index.html')) { throw 'Build the frontend first: cd frontend; npm run build' }
$serverArgs = @('-E', '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', "$Port")
if (Test-Path -LiteralPath '.env') { $serverArgs += @('--env-file', '.env') }
Write-Host "Neuro / Lab: http://127.0.0.1:$Port" -ForegroundColor Cyan
& $pythonPath @serverArgs
if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error.' }
