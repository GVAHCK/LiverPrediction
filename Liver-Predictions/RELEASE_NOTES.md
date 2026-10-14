# Release v3.0.0: Referral-Budgeted, Sex-Audited Selective Classification

**Release Date:** October 2026  
**DOI:** Pending Zenodo Archive  
**License:** MIT License  
**Environment:** Python 3.11.15 (Pinned via `requirements-lock.txt`)  

---

## Overview
This release delivers the complete, leak-free, reproducible research package and research prototype for:
> **"Referral-Budgeted, Sex-Audited Selective Classification for Small-Sample Liver Laboratory Data"**

It remediates all known methodological issues (E1–E12) identified in prior versions, implementing a leak-free nested cross-validation framework, a referral-budgeted sex-audited selective triage policy, patient-level bootstrap statistics, and a secondary methodological replication on the UCI HCV dataset ($N=615$, CC BY 4.0).

---

## Key Highlights & Findings
1. **Zero Data Leakage:** All preprocessing, resampling, model selection (48 combinations), calibration, and operational thresholds are tuned strictly on inner training folds across a $5\times 3$ Repeated Stratified outer cross-validation harness.
2. **True Holdout Performance vs Trivial Baselines:** 
   - Outer AUROC = **0.7472** [0.7382, 0.7562] (vs. 0.5000 chance)
   - Average Precision = **0.8909** [0.8830, 0.8989] (vs. 0.7136 base rate)
   - Brier Score = **0.1740** [0.1711, 0.1770] (vs. 0.2044 prevalence predictor)
   - Calibration Slope = **1.044** (ideal = 1.000)
3. **Screening Trade-off Reality:** Achieving 91.2% screening recall forces binary specificity to 23.0% (vs. 0.0% for all-positive classifier), establishing the necessity of clinical referral triage.
4. **Primary Hypothesis ($H_1$) Outcome:** The pre-registered primary comparison at $B=0.20$ was **NOT supported** ($\Delta = -0.47\%$, $p=0.8680$), as small female sample size ($n=142$) precluded statistical separation.
5. **Exploratory Disparity Mitigation ($B=0.30$):** In secondary exploratory analysis, sex-audited triage reduced worst-group missed disease from 15.5% (no abstention) to **5.0%**.
6. **Statistical Rigor:** Eliminated all fold-level tests; 2,000x patient-level paired bootstrap tests replace legacy fold DeLong and $t$-tests.
7. **Interpretability & Error Profiling:** False negative profiling revealed that 39.4% of false negatives occur in females, who present with deceptively normal median ALT (16.0 U/L) and bilirubin (0.70 mg/dL).
8. **Secondary Methodological Replication:** Evaluated on the UCI HCV cohort ($N=615$) across Arm A (exclude suspect donors, AUROC: 0.9928) and Arm B (include suspect donors, AUROC: 0.9788).
9. **Production Serving Latency:** Shipped research prototype v3.0.0 with HTTP p50 of 17.81 ms and pure ML inference p50 of 16.79 ms.
10. **100% Numerical Traceability:** Every number in the publication manuscript (`paper/paper.tex` and `paper/tables/*.tex`) is auto-generated from `/reports/*.json` and verified by `src/verify_paper_numbers.py`.

---

## One-Command Quickstart
```bash
# Clone repository
git clone https://github.com/SAKETH070706/CODE_STORM.git
cd CODE_STORM

# Set up environment and run complete reproduction pipeline
make all
```
Or on Windows PowerShell:
```powershell
.\run_all.ps1 -Target all
```
