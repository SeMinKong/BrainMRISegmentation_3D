param([switch]$WithML, [switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python 3.11+ is required.' }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Node.js 22.12+ or 24 LTS is required.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
  python -E -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw 'Could not create the project virtual environment.' }
}
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
& $pythonPath -E -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
if ($WithML) {
  if ($CpuOnly) {
    & $pythonPath -E -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
  }
  & $pythonPath -E -m pip install -e '.[ml]'
  if ($LASTEXITCODE -ne 0) { throw 'ML dependency installation failed.' }
}
Push-Location -LiteralPath (Join-Path $projectPath 'frontend')
try {
  npm.cmd ci
  if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
  npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Setup complete. Start with .\scripts\start.ps1' -ForegroundColor Green
