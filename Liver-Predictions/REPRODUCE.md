# Reproduction Guide: Automating Liver Disease Prediction

This document provides exact, deterministic commands to reproduce every phase of the audit, benchmark, calibration factorial, statistical tests, model packaging, and web server execution.

All experiments use fixed seed **42** and adhere strictly to leak-free cross-validation protocols.

---

## 1. Environment Setup

### System Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** 3.8.x (CPython 64-bit)
- **Fast Package Manager:** [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Using uv (fastest)
uv venv .venv --python 3.8

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Pinned Dependencies
```bash
uv pip install -r requirements.txt
```

---

## 2. Deterministic Pipeline Execution

Run each stage in sequence to regenerate all processed artifacts, cross-validation folds, model benchmarks, and reports:

### Phase 1: Data Audit & Target Recoding
Audits raw ILPD CSV against cryptographic hash, documents missing values and duplicates, and recodes raw labels (`1 -> 1` Disease, `2 -> 0` Non-Disease).
```bash
python src/data_audit.py
pytest tests/test_data_integrity.py -v
```
- **Output:** `data/processed/ilpd_clean.parquet`, `reports/01_data_audit.md`

---

### Phase 2: Leak-Free Evaluation Harness & Splits Generation
Generates deterministic 5x3 Repeated Stratified K-Fold (stratified on `sex + label`) and duplicate-grouped sensitivity splits.
```bash
python src/cv.py
pytest tests/test_harness.py -v
```
- **Output:** `splits/outer_splits_5x3.json`, `splits/grouped_splits_5.json`

---

### Phase 3: Equal-Budget Model Benchmarking
Benchmarks 7 model families across all 15 outer folds using equal 30-trial inner cross-validation for hyperparameter tuning.
```bash
python src/train_models.py
```
- **Output:** `reports/oof_predictions.parquet`, `reports/03_models_benchmark.md`

---

### Phase 4: Imbalance & Calibration Factorial
Evaluates a 4x3 factorial design (4 resampling methods $\times$ 3 calibration methods) across 15 outer folds with nested inner probability calibration.
```bash
python src/imbalance_calibration.py
```
- **Output:** `reports/04_imbalance_calibration.md` (Selects SMOTE + Platt Sigmoid Calibration)

---

### Phase 5: Clinical Decision Layer & Sex Audit
Tuning of decision threshold on inner folds ($t^* = 0.51$), 3-tier clinical triage evaluation, and demographic auditing across biological sex.
```bash
python src/clinical_decision.py
```
- **Output:** `reports/05_clinical_decision.md`

---

### Phase 6: Statistical Significance, Ablations & Error Profiling
Computes paired statistical significance tests (DeLong / Paired t-test), quantifies global leakage optimism gap (+0.0007 AUROC inflation), and profiles false negative cases.
```bash
python src/statistics_ablations.py
```
- **Output:** `reports/06_statistics_ablations.md`

---

### Phase 7: Production Model Export & Latency Benchmarks
Packages the leak-free calibrated production pipeline into `app/production_model.joblib` and benchmarks API response latency over 100 requests.
```bash
python src/export_production_model.py
python src/benchmark_latency.py
```
- **Output:** `app/production_model.joblib`, `reports/07_latency_benchmark.md`

---

## 3. Automated Test Suite

Execute the complete pytest verification suite (33 tests covering data integrity, leak-free pipelines, metrics math, model bundle schemas, input validation, and API contracts):

```bash
pytest tests/ -v
```

Expected output:
```
============================= 33 passed in 1.50s ==============================
```

---

## 4. Running the Web Application Locally

### Method A: One-Click Startup Scripts
- **PowerShell:** `.\run.ps1`
- **Command Prompt:** `run.bat`

### Method B: Manual Startup
```bash
python server.py
```

The application will start on `http://127.0.0.1:9000/`.

### Verified Endpoints:
- `GET  /` : Brand homepage
- `GET  /check-up` : Interactive clinical risk assessment form
- `POST /check-up` : Form submission & JSON API (accepts patient biomarkers; returns calibrated probability, risk band, recommendations, and disclaimer)
- `GET  /health` : Health check and model readiness probe (`model_loaded`, `model_version`, `data_hash`)
- `GET  /contact` : Feedback and inquiries
