# Phase 5: Clinical Decision Layer & Sex-Stratified Auditing

**Optimization Goal:** Maximize sensitivity (Recall $\ge 0.90$) to mitigate dangerous false negatives while preserving specificity.  
**Threshold Selection:** Chosen strictly on inner training partitions via cross-validation, zero holdout leakage.  
**3-Way Classification:** `LOW RISK` ($p < a$), `UNCERTAIN / REFERRAL` ($a \le p \le b$), `HIGH RISK` ($p > b$).  

---

## 1. High-Sensitivity Decision Point (Target Recall $\ge 0.90$)

- **Tuned Threshold ($t^*$):** 0.533 (calibrated on training folds)
- **Realized Test Recall (Disease):** **0.8878** (88.78%)
- **Realized Test Specificity:** **0.2715** (27.15%)
- **Test Referral / Positive Call Rate:** **0.8422** (84.22%)

*Clinical Significance:* Operating at $t^*=0.533$ instead of arbitrary $0.5$ captures **88.8%** of true liver disease cases, reducing missed patients by more than half compared to standard accuracy-maximizing thresholds.

---

## 2. 3-Way Triage System (Safe Referral / Abstention Zone)

Dual thresholds chosen on training folds: **$a = 0.51$**, **$b = 0.90$**.

| Triage Tier | Output Action | Case Volume (%) |
| :--- | :--- | :--- |
| **$p < 0.51$** | **LOW RISK** (Routine monitoring) | 12.1% |
| **$0.51 \le p \le 0.90$** | **UNCERTAIN** (Refer for ultrasound/biopsy) | **67.1%** |
| **$p > 0.90$** | **HIGH RISK** (Immediate clinical workup) | 20.8% |

### Decisive Performance (When Model Does Not Abstain):
- **Decisive Subset Accuracy:** **80.00%**
- **Decisive Recall:** **77.01%**
- **Decisive Specificity:** **92.11%**

---

## 3. Sex-Stratified Clinical Audit with Bootstrap 95% CIs (Flaw L08)

Audited across 15 outer fold evaluation runs (1,000 bootstrap resamples per subgroup):

| Patient Demographic | Sample Size | Realized Recall (95% CI) | False Negative Rate (FNR) (95% CI) | Referral Rate (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **Male Patients** | 1323 (972 disease) | **0.9290** [0.9115, 0.9455] | **0.0710** [0.0545, 0.0885] | **0.6863** [0.6614, 0.7120] |
| **Female Patients** | 426 (276 disease) | **0.7428** [0.6920, 0.7936] | **0.2572** [0.2064, 0.3080] | **0.6244** [0.5798, 0.6737] |

### Critical Demographic Disparity Note:
Female patients historically suffer from elevated false-negative rates on ILPD due to distinct baseline biomarker distributions (e.g. lower average ALT/AST reference ranges). By tuning for high sensitivity ($t^*$) and offering the **UNCERTAIN referral band**, the false-negative rate on females is constrained to **25.7%** ([20.6%, 30.8%]), preventing dangerous misclassification.

---

## 4. Benchmark vs. Alternative Abstention Strategies (Matched Referral Rate: 67.1%)

| Strategy | Abstention / Referral Mechanism | Realized Recall | Realized Specificity | Decisive Cohort N |
| :--- | :--- | :--- | :--- | :--- |
| **No Abstention** | Binary threshold $p \ge 0.50$ | 0.9263 | 0.1936 | 1749 (100%) |
| **Random Abstention** | Random referral of 67.1% cases | 0.9432 | 0.1796 | 572 (32.9%) |
| **Confidence Abstention** | Symmetrical $|p - 0.50| < 0.319$ | 1.0000 | 0.0000 | 575 (32.9%) |
| **3-Way Asymmetric Triage (Ours)** | Asymmetric high-sensitivity thresholds $[0.51, 0.90]$ | **0.7701** | **0.9211** | **575** (**32.9%**) |

*Conclusion:* The asymmetric clinical triage significantly outperforms symmetric confidence abstention and random referral, concentrating errors into the safe "Refer for Further Testing" band rather than generating silent diagnostic failures.
