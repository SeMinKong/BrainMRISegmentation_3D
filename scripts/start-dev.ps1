# Keep this terminal running. Ctrl+C stops both the web and API dev servers.
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run .\scripts\setup.ps1 first.' }
$serverArgs = @('-E', '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8000', '--reload')
if (Test-Path -LiteralPath '.env') { $serverArgs += @('--env-file', '.env') }
$apiProcess = Start-Process -FilePath $pythonPath -ArgumentList $serverArgs -WorkingDirectory $projectPath -WindowStyle Hidden -PassThru
try {
  Set-Location -LiteralPath (Join-Path $projectPath 'frontend')
  npm.cmd run dev
} finally {
  # Only terminate the process tree created above, including Uvicorn's reloader child.
  if (-not $apiProcess.HasExited) { taskkill.exe /PID $apiProcess.Id /T /F | Out-Null }
}
