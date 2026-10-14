# Liverify — Retrospective Liver Disease Prediction & Clinical Decision Support System

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Tests](https://img.shields.io/badge/pytest-33%20passed-brightgreen.svg)](tests/)
[![Model Version](https://img.shields.io/badge/model--version-2.0.0-indigo.svg)](app/production_model.joblib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Liverify is a reproducible, leak-free machine learning application and clinical decision support system built on the **Indian Liver Patient Dataset (ILPD)**. The project features a rigorous 5x3 Repeated Stratified Cross-Validation harness, probabilistic calibration (Platt Sigmoid), 3-tier clinical risk triage (`LOW_RISK`, `UNCERTAIN`, `HIGH_RISK`), sex-stratified disparity auditing, and a hardened production **Flask** service.

---

## Key Highlights & Audit Findings

The project underwent an exhaustive methodological audit resolving **24 cataloged data leakage, evaluation, and production flaws (L01–L24)**:

- **100% Leak-Free Preprocessing:** All missingness imputation (median + indicators), log transforms, feature scaling, and SMOTE resampling strictly execute inside `imblearn` / `scikit-learn` pipelines fitted on training folds only.
- **Probabilistic Calibration:** Platt Sigmoid calibration yields a well-calibrated Brier score of **0.1722** and Cox calibration slope of **1.054** (close to ideal 1.0).
- **High-Sensitivity Clinical Screening:** High-sensitivity threshold ($t^* = 0.51$) achieves **88.8% recall on holdout disease cases**, mitigating hazardous false negatives.
- **3-Way Decision Triage:** Replaces brittle single binary cutoffs with actionable risk tiers:
  - **Low Risk ($P < 0.35$):** 91.7% Negative Predictive Value (NPV).
  - **Uncertain ($0.35 \le P \le 0.75$):** Automated abstention; routes ambiguous borderline cases to clinical evaluation and imaging.
  - **High Risk ($P > 0.75$):** 86.5% Positive Predictive Value (PPV).
  - **Accuracy on Decisive Cases:** **80.0%**.
- **Demographic Disparity Audit:** Exposed a **3.6x higher false negative rate in female patients (25.7% FNR)** compared to males (7.1% FNR) due to ILPD dataset demographic imbalance (75.6% male cohort) and deceptive normal median enzymes in female pathology.
- **Fast Production Inference:** Sub-15ms median response time (p50: 14.5 ms, p95: 16.9 ms) with zero startup retraining.

Detailed reports are available in [`reports/`](reports/), [`MODEL_CARD.md`](MODEL_CARD.md), and [`REPRODUCE.md`](REPRODUCE.md).

---

## Performance Benchmark

All metrics evaluated across 15 outer folds (5x3 Repeated Stratified K-Fold, N=8,745 evaluations) with equal 30-trial inner CV hyperparameter tuning:

| Model Architecture | AUROC (Mean $\pm$ Std) | 95% Conf. Interval | Brier Score | Recall (Disease) |
|:---|:---:|:---:|:---:|:---:|
| **Calibrated Logistic Regression + SMOTE (Production v2.0.0)** | **0.7548 $\pm$ 0.0175** | **[0.7450, 0.7647]** | **0.1722** | **88.8%** |
| Tuned K-Nearest Neighbors | 0.7520 $\pm$ 0.0210 | [0.7402, 0.7638] | 0.1845 | 86.2% |
| Logistic Regression (Uncalibrated Baseline) | 0.7473 $\pm$ 0.0223 | [0.7347, 0.7599] | 0.1780 | 85.5% |
| Tuned Random Forest Classifier | 0.7419 $\pm$ 0.0207 | [0.7302, 0.7536] | 0.1788 | 83.1% |
| HistGradientBoosting Classifier | 0.7225 $\pm$ 0.0195 | [0.7115, 0.7335] | 0.1892 | 81.0% |
| Legacy KNN ($k=22$, No Log Transform) | 0.7185 $\pm$ 0.0238 | [0.7050, 0.7320] | 0.1980 | 79.8% |
| Decision Tree Classifier | 0.6905 $\pm$ 0.0274 | [0.6750, 0.7060] | 0.2210 | 72.5% |
| Zero-Rule Majority Baseline | 0.5000 $\pm$ 0.0000 | [0.5000, 0.5000] | 0.2864 | 100.0% |

*Note: Paired DeLong / t-test between Logistic Regression and Random Forest indicates no statistically significant difference ($\Delta \text{AUROC} = -0.0098, p = 0.794$). Logistic Regression was chosen for production due to superior calibration, interpretability, and low inference latency.*

---

## Project Structure

```
Liver-Predictions/
├── app/
│   └── production_model.joblib     # Packaged model v2.0.0 (calibrated pipeline, metadata, thresholds)
├── data/
│   ├── raw/
│   │   └── indian_liver_patient.csv # Raw ILPD dataset (SHA256 verified)
│   └── processed/
│       └── ilpd_clean.parquet      # Recoded target (1=disease, 0=non-disease), verified clean
├── legacy/
│   └── legacy_server.py            # Preserved legacy server for historical comparison
├── notebooks/                      # Exploratory data analysis notebooks
├── reports/
│   ├── 00_audit.md                 # Complete flaw catalog (L01–L24)
│   ├── 01_data_audit.md            # Data integrity, duplicate audit, and target distribution
│   ├── 03_models_benchmark.md      # Equal-budget benchmark of 7 models across 15 folds
│   ├── 04_imbalance_calibration.md # Resampling and probability calibration factorial
│   ├── 05_clinical_decision.md     # Threshold optimization and sex disparity audit
│   ├── 06_statistics_ablations.md  # Paired statistical tests and error profiling
│   └── 07_latency_benchmark.md     # p50 / p95 response time benchmarks
├── splits/
│   ├── outer_splits_5x3.json       # Deterministic 5x3 Repeated Stratified K-Fold splits
│   └── grouped_splits_5.json       # Duplicate-grouped sensitivity splits
├── src/
│   ├── cv.py                       # Cross-validation harness & leak-free pipeline factory
│   ├── data_audit.py               # Data audit and hashing script
│   ├── train_models.py             # Equal-budget model evaluation script
│   ├── imbalance_calibration.py    # Resampling x calibration factorial script
│   ├── clinical_decision.py        # Decision threshold and sex audit script
│   ├── statistics_ablations.py     # Significance tests and error profiling script
│   ├── export_production_model.py  # Production bundle packaging script
│   └── benchmark_latency.py        # Endpoint latency benchmark script
├── static/                         # Static CSS, JS, and image assets
├── templates/
│   ├── base.html                   # Base layout template
│   ├── check.html                  # Risk assessment form, calibrated report & print view
│   ├── contact.html                # Contact page
│   └── home.html                   # Modern landing page
├── tests/
│   ├── test_data_integrity.py      # Unit tests for hashing, recoding, and data schema
│   ├── test_harness.py             # Tests for split isolation and leak-free pipelines
│   ├── test_app.py                 # Server route and UI integration tests
│   └── test_production_app.py      # Production bundle, validation, and API contract tests
├── MODEL_CARD.md                   # Formal Model Card (Mitchell et al.)
├── REPRODUCE.md                    # Step-by-step reproduction guide
├── requirements.txt                # Pinned dependencies
├── run.ps1                         # PowerShell quickstart script
├── run.bat                         # Windows CMD quickstart script
└── server.py                       # Production Flask server (loads model v2.0.0 directly)
```

---

## Quickstart & Installation (Windows / Linux / macOS)

### 1. Automated Quickstart (Recommended)
From PowerShell:
```powershell
.\run.ps1
```
From Command Prompt:
```cmd
run.bat
```

### 2. Manual Installation
1. **Create and activate a virtual environment (Python 3.8)**:
   ```bash
   uv venv .venv --python 3.8
   .venv\Scripts\Activate.ps1   # Windows PowerShell
   # source .venv/bin/activate  # Linux/macOS
   ```
2. **Install pinned dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
3. **Run the production server**:
   ```bash
   python server.py
   ```
4. **Access the application**: Open [http://127.0.0.1:9000](http://127.0.0.1:9000) in your browser.

---

## API Documentation (`POST /check-up`)

The application supports both JSON API requests and interactive HTML form submissions.

### Request Payload Example
```json
{
  "age": 52,
  "gender": "Male",
  "tb": 2.8,
  "db": 1.2,
  "ap": 290,
  "aa": 55,
  "asa": 68,
  "tp": 6.8,
  "ab": 3.2,
  "ag": 0.88
}
```

### Response Example
```json
{
  "status": "success",
  "result": 1,
  "prediction": 1,
  "probability": 0.7842,
  "risk_band": "HIGH_RISK",
  "recommendation": "Elevated probability of liver disease indicated. Prompt consultation with a gastroenterologist or hepatologist is recommended.",
  "model_version": "2.0.0",
  "data_hash": "9d90de6469ed034c209a23f3d0c78133e2f4b991e2b72ac86a8a9c09368238f6",
  "disclaimer": "Notice: This prediction is an experimental statistical estimate based on retrospective machine learning analysis of the Indian Liver Patient Dataset (ILPD)..."
}
```

---

## Running Automated Tests

Run the full automated test suite (33 tests):

```bash
pytest tests/ -v
```

All tests pass deterministically in under 2 seconds.

---

## Clinical Disclaimer & Medical Limitations

> [!WARNING]
> **RESEARCH & DEMONSTRATION USE ONLY**  
> 1. **Statistical Estimate:** Predictions generated by this system are statistical probabilities derived from retrospective analysis of historical tabular data (ILPD). They do **NOT** constitute medical diagnosis, clinical prognosis, or treatment recommendations.
> 2. **Demographic Disparity Notice:** Due to cohort limitations in the underlying dataset, this model exhibits a **significantly higher false negative rate in female patients (25.7%)**. Normal enzyme values in female patients should never be interpreted as clinical absence of disease.
> 3. **Consult a Physician:** Always seek the direct evaluation of a board-certified physician or hepatologist for medical decision-making, physical examinations, and comprehensive diagnostic workups.
