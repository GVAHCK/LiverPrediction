@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Setting up virtual environment with uv...
    uv venv .venv --python 3.8
    uv pip install -r requirements.txt
)

echo ==========================================================
echo   LIVERIFY: COMPLETE REPRODUCIBILITY PIPELINE RUNNER      
echo ==========================================================

echo.
echo [1/7] Running Phase 1: Data Integrity ^& Hashing...
.venv\Scripts\python.exe src/data_audit.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/7] Running Phase 2: Generating Leak-Free Splits...
.venv\Scripts\python.exe src/cv.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/7] Running Phase 4: Imbalance ^& Calibration Factorial...
.venv\Scripts\python.exe src/imbalance_calibration.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [4/7] Running Phase 5: Clinical Decision Layer ^& Sex Audit...
.venv\Scripts\python.exe src/clinical_decision.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [5/7] Running Phase 6: Statistics, Ablations ^& Error Profiling...
.venv\Scripts\python.exe src/statistical_analysis.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [6/7] Running Phase 7: Exporting Production Bundle ^& Latency Benchmark...
.venv\Scripts\python.exe src/export_production_model.py
.venv\Scripts\python.exe src/benchmark_latency.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [7/7] Running Pytest Suite (33 Tests)...
.venv\Scripts\python.exe -m pytest tests/ -v
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==========================================================
echo   SUCCESS: All pipelines and tests executed cleanly!      
echo ==========================================================
