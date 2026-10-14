$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path "$ScriptDir\.venv\Scripts\python.exe")) {
    Write-Host "[!] Virtual environment not found. Setting up with uv..." -ForegroundColor Yellow
    uv venv .venv --python 3.8
    uv pip install -r requirements.txt
}

Write-Host "[*] Starting Liverify on http://127.0.0.1:9000 ..." -ForegroundColor Green
& "$ScriptDir\.venv\Scripts\python.exe" "$ScriptDir\server.py"
