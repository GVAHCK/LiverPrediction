# Phase 14 — Interpretability, False Negative Error Profiling, and Vector Figures

**Date:** 2026-10-14  
**Cohort:** Indian Liver Patient Dataset (ILPD, N=583)  
**Output Figures:** `paper/figures/fig1_architecture.pdf` ... `fig5_false_negative_profile.pdf` (+300 DPI PNGs)  

---

## 1. Standardized Model Coefficients (Log-Odds & Odds Ratios)

| Biomarker Feature | Standardized Log-Odds Coef | Odds Ratio (per 1 SD increase) | Directional Association |
|:---|:---:|:---:|:---|
| `Alamine_Aminotransferase (log)` | **+0.5253** | **1.691** | Elevates Risk |
| `Direct_Bilirubin (log)` | **+0.4455** | **1.561** | Elevates Risk |
| `Total_Protiens (std)` | **+0.3941** | **1.483** | Elevates Risk |
| `Aspartate_Aminotransferase (log)` | **+0.3317** | **1.393** | Elevates Risk |
| `Alkaline_Phosphotase (log)` | **+0.3259** | **1.385** | Elevates Risk |
| `Age (std)` | **+0.3081** | **1.361** | Elevates Risk |
| `Total_Bilirubin (log)` | **+0.2785** | **1.321** | Elevates Risk |
| `A/G Ratio Missing Flag` | **+0.0339** | **1.034** | Elevates Risk |
| `A/G Ratio (std)` | **+0.0127** | **1.013** | Elevates Risk |
| `is_male (binary)` | **-0.2389** | **0.787** | Lowers Risk / Protective |
| `Albumin (std)` | **-0.4187** | **0.658** | Lowers Risk / Protective |

---

## 2. False Negative Error Profile & Female Pathology Disparity

Analysis of the **33 False Negative patient cases** (13 females, 20 males):

| Cohort Group | Patient Count | Median Total Bilirubin (mg/dL) | Median ALT (U/L) | Median AST (U/L) | Median Albumin (g/dL) | Median Age |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Female False Negatives (FN)** | **13 (39.4%)** | **0.70** (Normal) | **16.0** (Normal) | **19.0** (Normal) | **3.50** | **29** |
| **Female True Positives (TP)** | 79 | 0.90 (Elevated) | 30.0 (Elevated) | 40.0 (Elevated) | 3.20 | 46 |
| **Male False Negatives (FN)** | 20 | 0.80 | 20.0 | 26.0 | 3.55 | 31 |
| **Male True Positives (TP)** | 304 | 1.80 | 46.0 | 60.5 | 3.00 | 48 |

### Clinical Insight:
Female false negatives systematically present with **normal median bilirubin (0.70 mg/dL, standard reference: 0.1–1.2 mg/dL)** and **normal transaminases (ALT: 16.0 U/L, reference: 7–35 U/L)**. In female patients, hepatic pathology often presents without acute enzymatic flares, misleading linear and tree models trained on a male-dominated cohort (75.6% male).

---

## 3. Publication Figure Index

All figures are compiled in vector format (`.pdf`) and 300 DPI bitmap (`.png`) using an accessible Okabe-Ito colorblind-safe palette:

1. [`paper/figures/fig1_architecture.pdf`](file:///c:/Users/venka/Downloads/Liver-Predictions/Liver-Predictions/paper/figures/fig1_architecture.pdf) — System decision flow and selective triage logic.
2. [`paper/figures/fig2_calibration_curves.pdf`](file:///c:/Users/venka/Downloads/Liver-Predictions/Liver-Predictions/paper/figures/fig2_calibration_curves.pdf) — Sex-stratified reliability curves and predicted probability histograms.
3. [`paper/figures/fig3_risk_coverage.pdf`](file:///c:/Users/venka/Downloads/Liver-Predictions/Liver-Predictions/paper/figures/fig3_risk_coverage.pdf) — Worst-group missed disease rate vs. cohort coverage tradeoff.
4. [`paper/figures/fig4_feature_importance.pdf`](file:///c:/Users/venka/Downloads/Liver-Predictions/Liver-Predictions/paper/figures/fig4_feature_importance.pdf) — Standardized logistic regression feature weights and odds ratios.
5. [`paper/figures/fig5_false_negative_profile.pdf`](file:///c:/Users/venka/Downloads/Liver-Predictions/Liver-Predictions/paper/figures/fig5_false_negative_profile.pdf) — Comparative boxplots showing deceptive normal enzyme levels in female false negatives.
