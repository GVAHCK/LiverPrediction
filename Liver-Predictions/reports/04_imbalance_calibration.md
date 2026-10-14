# Phase 4: Imbalance & Calibration Factorial Analysis

**Dataset:** ILPD (71.4% Disease, 28.6% Non-Disease)  
**Evaluation:** 15 Outer Test Partitions ($5 \times 3$ Repeated Stratified CV)  
**Base Estimator:** Regularized Logistic Regression ($L_2$, $C=0.1$)  

---

## 1. Factorial Results Table ($4 \times 3$)

| Imbalance Handling | Calibration Method | AUROC (mean ± std) | Average Precision | Brier Score (lower is better) | Log Loss | Combined Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SMOTE` | `sigmoid` | 0.7548 ± 0.0175 | 0.8948 | **0.1722** | 0.5070 | #2 |
| `none` | `sigmoid` | 0.7540 ± 0.0167 | 0.8942 | **0.1725** | 0.5077 | #5 |
| `class_weight` | `sigmoid` | 0.7534 ± 0.0170 | 0.8940 | **0.1725** | 0.5077 | #5 |
| `RandomOverSampler` | `sigmoid` | 0.7529 ± 0.0198 | 0.8936 | **0.1725** | 0.5078 | #10 |
| `none` | `isotonic` | 0.7533 ± 0.0192 | 0.8895 | **0.1734** | 0.5466 | #11 |
| `RandomOverSampler` | `isotonic` | 0.7530 ± 0.0243 | 0.8892 | **0.1736** | 0.5405 | #13 |
| `class_weight` | `isotonic` | 0.7521 ± 0.0227 | 0.8890 | **0.1729** | 0.5378 | #14 |
| `none` | `none` | 0.7518 ± 0.0176 | 0.8930 | **0.1728** | 0.5081 | #15 |
| `SMOTE` | `isotonic` | 0.7529 ± 0.0196 | 0.8894 | **0.1740** | 0.5487 | #16 |
| `SMOTE` | `none` | 0.7519 ± 0.0195 | 0.8925 | **0.2023** | 0.5781 | #20 |
| `RandomOverSampler` | `none` | 0.7511 ± 0.0220 | 0.8923 | **0.2023** | 0.5788 | #22 |
| `class_weight` | `none` | 0.7517 ± 0.0180 | 0.8928 | **0.2027** | 0.5802 | #23 |

---

## 2. Selection of the Winning Pipeline

- **Selected Winner:** `SMOTE` with `sigmoid` calibration.
  - **AUROC:** 0.7548 ± 0.0175
  - **Brier Score:** 0.1722
  - **Log Loss:** 0.5070
  
### Key Observations:
1. **Resampling vs. Natural Prevalence:**
   - Aggressive synthetic oversampling (SMOTE) artificially shifts probability estimates towards 0.5, degrading the Brier score on true population prevalence unless recalibrated.
   - Standard training without synthetic distortion (`none` or `class_weight`) achieves superior probability sharpness.
2. **Impact of Post-Hoc Probability Calibration:**
   - Sigmoid (Platt) calibration significantly reduces the Brier score and Log Loss while preserving rank discrimination (AUROC).
   - Isotonic regression slightly overfits given the moderate dataset size ($N=583$), making parametric Sigmoid calibration the more stable choice.
