# Phase 12 — Patient-Level Paired Bootstrap Statistics & Multiple Testing

**Date:** 2026-10-14  
**Method:** 2,000 Patient-Level Paired Bootstrap Resamples (Repeat-Aware Aggregation over N=583 Patients)  
**Correction:** Holm-Bonferroni Family-Wise Error Rate Control on Secondary Comparisons  
**Compliance:** Zero Fold-Level DeLong or Paired t-Tests (Rule 6 Fully Satisfied)  

---

## 1. Patient-Level Performance Metrics (95% Bootstrap CIs)

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
|:---|:---:|:---:|
| **AUROC** | **0.7499** | [0.7102, 0.7874] |
| **Average Precision (AP)** | **0.8915** | [0.8592, 0.9177] |
| **Brier Score** | **0.1721** | [0.1576, 0.1867] |
| **Recall (Sensitivity on Disease)** | **92.1%** | [89.4%, 94.5%] |
| **Specificity (Non-Disease)** | **21.5%** | [15.3%, 28.5%] |
| **Female False Negative Rate (FNR)** | **14.0%** | [7.1%, 21.7%] |
| **Male False Negative Rate (FNR)** | **6.2%** | [3.8%, 8.8%] |
| **Sex FNR Disparity ($	ext{FNR}_{	ext{female}} - 	ext{FNR}_{	ext{male}}$)** | **+7.8%** | [+0.5%, +15.8%] |

---

## 2. Primary Confirmatory Endpoint Comparison ($B=0.20$)

> **Hypothesis:** Group-Audited Selective Policy significantly reduces the worst-group missed-disease rate compared to standard Global Confidence Abstention at matched referral budget $B=0.20$.

- **Comparison:** Group-Audited Policy vs. Global Confidence Abstention
- **$\Delta$ Worst-Group Missed Rate (Baseline $-$ Policy):** **-0.47%**
- **95% Bootstrap Confidence Interval:** **[-4.88%, +3.33%]**
- **Bootstrap Two-Sided $p$-Value:** **$p = 0.8680$**

---

## 3. Secondary Exploratory Comparisons (Holm-Bonferroni Corrected)

Positive $\Delta$ indicates the proposed policy achieves a lower (superior) worst-group missed-disease rate on accepted cases:

| Baseline Comparison | $\Delta$ Worst Miss Rate (Baseline $-$ Proposed) | 95% Bootstrap CI | Raw $p$-value | Holm-Adjusted $p$-value |
|:---|:---:|:---:|:---:|:---:|
| `decision_no_abs` | **+0.48%** | [-1.07%, +2.84%] | 0.7550 | **1.0000** |
| `decision_random` | **-0.04%** | [-4.68%, +4.47%] | 0.9960 | **1.0000** |
| `decision_global_two_threshold` | **-12.09%** | [-19.58%, -5.82%] | 0.0000 | **0.0000** |
| `decision_group_lambda0` | **-0.92%** | [-3.30%, +0.00%] | 0.7990 | **1.0000** |
| `decision_cost_sensitive` | **-0.13%** | [-0.56%, +0.00%] | 0.7400 | **1.0000** |

---

## 4. Elimination of Invalid Fold-Level Tests (Fixes E3)

1. **DeLong and Fold-Level t-Tests Removed:**  
   In compliance with Rule 6 and methodological best practices, all p-values derived from paired t-tests or DeLong's test across non-independent, overlapping cross-validation folds have been permanently removed.
2. **Replacement of L03 Leakage Gap Metric:**  
   The empirical difference between the globally leaked preprocessor and the leak-free pipeline is **+0.0007 AUROC** with a patient-level bootstrap 95% CI of **[-0.0002, +0.0016]**. Because the confidence interval encompasses zero, the previously claimed $p=0.003$ was confirmed to be a statistical artifact of treating overlapping folds as independent samples.
