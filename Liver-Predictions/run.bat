@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [!] Creating virtual environment with uv...
    uv venv .venv --python 3.8
    uv pip install -r requirements.txt
)
echo [*] Starting Liverify on http://127.0.0.1:9000 ...
.venv\Scripts\python.exe server.py
