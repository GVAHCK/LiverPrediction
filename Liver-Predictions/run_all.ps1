param (
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = "$ScriptDir\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[!] Virtual environment not found. Setting up with uv..." -ForegroundColor Yellow
    uv venv .venv --python 3.11
    uv pip install -r requirements-lock.txt
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  REPRODUCIBILITY PIPELINE RUNNER (Target: $Target)       " -ForegroundColor Cyan
Write-Host "  Referral-Budgeted, Sex-Audited Selective Classification " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

function Run-Step($Name, $Command) {
    Write-Host "`n[*] Executing step: $Name..." -ForegroundColor Green
    & $PythonExe @Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $Name with exit code $LASTEXITCODE"
    }
}

switch ($Target.ToLower()) {
    "data" {
        Run-Step "Data Audit & Cleaning" @("-m", "src.data_audit")
    }
    "audit" {
        Run-Step "Data Audit & Integrity" @("-m", "src.data_audit")
    }
    "nested" {
        Run-Step "Nested Cross-Validation (Phase 10)" @("-m", "src.nested_selection")
    }
    "policy" {
        Run-Step "Selective Policy & Baselines (Phase 11)" @("-m", "src.selective_policy")
    }
    "stats" {
        Run-Step "Patient-Level Bootstrap Statistics (Phase 12)" @("-m", "src.patient_statistics")
    }
    "robust" {
        Run-Step "Robustness & HCV Replication (Phase 13)" @("-m", "src.robustness_experiments")
    }
    "hcv" {
        Run-Step "UCI HCV Methodological Replication" @("-m", "src.robustness_experiments")
    }
    "figs" {
        Run-Step "Interpretability & Vector Figures (Phase 14)" @("-m", "src.interpretability_analysis")
    }
    "paper" {
        Run-Step "Auto-Generate LaTeX Tables" @("-m", "src.generate_paper_tables")
        Run-Step "Verify Paper Numbers Against Reports JSON" @("-m", "src.verify_paper_numbers")
    }
    "app" {
        Run-Step "Export Production Prototype Model v3.0.0" @("-m", "src.export_production_model")
    }
    "test" {
        Run-Step "Run Pytest Suite" @("-m", "pytest", "-v", "tests/")
    }
    "verify" {
        Run-Step "Verify Paper Numbers Against Reports JSON" @("-m", "src.verify_paper_numbers")
    }
    "all" {
        Run-Step "1/10 Data Audit & Cleaning" @("-m", "src.data_audit")
        Run-Step "2/10 Nested Cross-Validation" @("-m", "src.nested_selection")
        Run-Step "3/10 Selective Policy & Baselines" @("-m", "src.selective_policy")
        Run-Step "4/10 Patient-Level Bootstrap Statistics" @("-m", "src.patient_statistics")
        Run-Step "5/10 Robustness & HCV Replication" @("-m", "src.robustness_experiments")
        Run-Step "6/10 Interpretability & Vector Figures" @("-m", "src.interpretability_analysis")
        Run-Step "7/10 Auto-Generate LaTeX Tables" @("-m", "src.generate_paper_tables")
        Run-Step "8/10 Export Production Model v3.0.0" @("-m", "src.export_production_model")
        Run-Step "9/10 Run Pytest Suite" @("-m", "pytest", "-v", "tests/")
        Run-Step "10/10 Verify Paper Numbers Against Reports JSON" @("-m", "src.verify_paper_numbers")
        Write-Host "`n==========================================================" -ForegroundColor Green
        Write-Host "  SUCCESS: All pipeline steps and verifications PASSED!    " -ForegroundColor Green
        Write-Host "==========================================================" -ForegroundColor Green
    }
    default {
        Write-Error "Unknown target: $Target. Valid targets: data, audit, nested, policy, stats, robust, hcv, figs, paper, app, test, verify, all"
    }
}
