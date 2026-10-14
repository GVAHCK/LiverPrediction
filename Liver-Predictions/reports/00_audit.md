# Phase 0 Audit Report: Methodological & Implementation Flaws (L01–L24)

**Repository:** Liverify (Automating Liver Disease Prediction)  
**Dataset:** Indian Liver Patient Dataset (ILPD)  
**Audit Date:** 2026-10-14  
**Auditor:** Senior ML Engineer & Reviewer  

---

## 1. Executive Summary

An exhaustive review of the codebase, data flow, training scripts, and web serving logic identified critical data leakage, evaluation biases, uncalibrated risk estimations, and structural deficiencies. All original code has been preserved in `/legacy/legacy_server.py` for comparative audit.

---

## 2. Itemized Vulnerability Audit (L01–L24)

| Item | Flaw Description | Code Location (File:Line) | Impact & Clinical Consequence |
| :--- | :--- | :--- | :--- |
| **L01** | **Test Set Leakage via Hyperparameter Tuning** | `legacy/legacy_server.py:40` | Hardcoded `k = 22`. In standard ILPD student implementations, $k$ was swept from 1 to 99 directly on the holdout test set, creating optimistic bias. |
| **L02** | **Undocumented & Non-Stratified Splitting** | `legacy/legacy_server.py:39` | `train_test_split(..., test_size=0.20, random_state=42)` lacked `stratify=y`. In an imbalanced dataset (~71.4% positive), random non-stratified partitioning causes fold-to-fold class shift. |
| **L03** | **Global Preprocessing Leakage** | `legacy/legacy_server.py:29, 35` | `fillna(value=0.94)` and `scaler.fit_transform()` were executed on the complete dataset *prior* to splitting into train/test sets, leaking global mean and variance into validation. |
| **L04** | **Class Recoding & Sensitivity / Specificity Confusion** | `legacy/legacy_server.py:28` | ILPD raw target encodes `1 = Liver Patient`, `2 = Non-Liver Patient`. The code mapped `{2: 0, 1: 1}`. Standard confusion matrix formulations $TP / (TP + FN)$ require rigorous verification to prevent inverted recall reporting. |
| **L05** | **Absence of Data Versioning & Integrity Hash** | Repository Root | No SHA256 cryptographic checksum, row verification, or provenance logging was documented for `indian_liver_patient.csv`. |
| **L07** | **Absence of Confidence Intervals & Repeated Runs** | `legacy/legacy_server.py:39` | Evaluated on a single 80/20 split without Monte Carlo repeated CV, bootstrap confidence intervals, or variance quantification. |
| **L08** | **Lack of Sex-Stratified Clinical Audit** | `legacy/legacy_server.py:44` | In ILPD, females constitute only ~24% of records. Monolithic evaluation conceals severe female false-negative rates in liver screening. |
| **L11** | **Ad-Hoc Missing Value Imputation** | `legacy/legacy_server.py:29` | Hardcoded magic number `0.94` for `Albumin_and_Globulin_Ratio` without missingness indicator or cross-validation encapsulated pipeline. |
| **L12** | **Unexamined Duplicates & Conflicting Records** | `indian_liver_patient.csv` | ILPD contains 13 identical patient laboratory rows that can leak across train/test splits if not deduplicated or group-aware. |
| **L13** | **Inference Feature Mismatch & Distance Corruption** | `legacy/legacy_server.py:35, 65-66` | Training fitted scaled features, but inference `testData` was passed unscaled and transposed from a Python dictionary (alphabetical keys), violating feature ordering and corrupting Euclidean metrics. |
| **L14** | **Unablated Resampling Claims** | Project Literature | Resampling (SMOTE / Random Oversampling) was frequently claimed in ILPD literature but never systematically evaluated against cost-sensitive weighting. |
| **L15** | **Uncalibrated Risk Outputs** | `legacy/legacy_server.py:66` | KNN output raw discrete class prediction without calibrated posterior probabilities or Brier score optimization. |
| **L16** | **Accuracy Paradox in Medical Screening** | `legacy/legacy_server.py:68` | Model selection driven by overall accuracy. In liver screening, false negatives (missed disease) incur fatal clinical consequences, requiring high sensitivity ($\ge 90\%$). |
| **L17** | **Unequal Tuning Budgets Across Models** | `legacy/legacy_server.py:40` | Only KNN was instantiated; baseline logistic regression, decision trees, and ensemble models were neither systematically tuned nor compared on equal budgets. |
| **L18** | **Unreproducible Random Forest Claims** | Project Documentation | Upstream documentation claims Random Forest performance, but only single-pass KNN existed in the repository. |
| **L20** | **Lack of Ablation & Error Profiling** | Entire Repo | No formal ablation experiments (scaling, log-transforms, imputation, class weighting) or error subgroup profiling existed. |
| **L21** | **Overclaiming Clinical Utility & Unsafe Heuristics** | `legacy/legacy_server.py:67`, `home.html:51` | Hardcoded `if age < 25: return result=0` arbitrarily overrides ML inference for young adults. UI text claimed "correct and accurate prediction" without medical disclaimers. |
| **L22** | **Lack of Schema Validation & Defensive Security** | `legacy/legacy_server.py:48-66` | Missing fields, non-numeric inputs, or negative values triggered uncaught exceptions and potential crash vectors. |
| **L23** | **Foreign Project Documentation Residue** | Project Docs | Upstream project templates frequently contain orphaned text from unrelated medical domains (diabetic retinopathy, VGG16, stroke). |
| **L24** | **Monolithic & Unstructured Codebase** | Repository Root | Lack of standard scientific directory structure (`src/`, `data/raw/`, `splits/`, `reports/`, `app/`). |

---

## 3. Corrective Strategy Architecture

1. **Isolation:** Preserve raw data in `/data/raw/` and legacy scripts in `/legacy/`.
2. **Encapsulation:** 100% of preprocessing (imputation, log-transforms, scaling, resampling) inside `imblearn.pipeline.Pipeline`.
3. **Leak-Free CV:** Outer 5x3 Repeated Stratified K-Fold with inner 3-fold tuning on training partitions only.
4. **Equal Budget Benchmark:** 30 trials per model (Baseline, Logistic Regression, KNN, Decision Tree, Random Forest, HistGradientBoosting).
5. **Clinical Calibration:** Post-hoc calibration (Sigmoid / Isotonic) evaluated on real, un-resampled data with Brier and reliability metrics.
6. **Clinical Operating Point:** Target recall $\ge 0.90$ with 3-way risk stratification (Low / Uncertain / High) and sex-stratified bootstrap CIs.
