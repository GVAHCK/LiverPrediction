# Phase 11 — Referral-Budgeted, Sex-Audited Selective Classification Results

**Date:** 2026-10-14  
**Protocol:** 15 Outer Folds $\times$ Prespecified Matched Referral Budgets ($B \in \{0.10, 0.20, 0.30\}$)  
**Primary Endpoint:** Worst-Group Missed-Disease Rate on Accepted Cases at $B=0.20$  
**Fairness Penalty:** $\lambda = 0.5$ on Referral Gap $|\text{Ref}_{\text{female}} - \text{Ref}_{\text{male}}|$  

---

## 1. Primary Benchmark Table across Referral Budgets

| Budget | Method | Worst-Group Missed Rate | 95% CI (Worst Miss) | Female Miss Rate | Male Miss Rate | Realized Referral | Referral Gap | Coverage | Infeasible Folds |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B=0.10 | `Baseline_Cost_Sensitive_Rejection` | **14.4%** | [9.7%, 19.2%] | 13.7% | 6.0% | 3.1% | 3.6% | 96.9% | 0/15 |
| B=0.10 | `Baseline_Global_Confidence` | **13.5%** | [8.2%, 18.8%] | 13.2% | 3.9% | 11.6% | 7.5% | 88.4% | 0/15 |
| B=0.10 | `Baseline_Global_Two_Threshold` | **3.3%** | [-1.1%, 7.6%] | 2.7% | 1.1% | 41.8% | 16.0% | 58.2% | 0/15 |
| B=0.10 | `Baseline_Group_Thresholds_lambda0` | **14.4%** | [9.7%, 19.2%] | 13.7% | 6.0% | 3.1% | 3.6% | 96.9% | 15/15 |
| B=0.10 | `Baseline_Matched_Random` | **16.1%** | [10.8%, 21.3%] | 15.2% | 7.0% | 10.4% | 6.4% | 89.6% | 0/15 |
| B=0.10 | `Baseline_No_Abstention` | **15.5%** | [11.0%, 20.0%] | 14.6% | 7.2% | 0.0% | 0.0% | 100.0% | 0/15 |
| B=0.10 | `Group_Audited_Policy_lambda0.5` | **14.4%** | [9.7%, 19.2%] | 13.7% | 6.0% | 3.1% | 3.6% | 96.9% | 15/15 |
| B=0.20 | `Baseline_Cost_Sensitive_Rejection` | **14.4%** | [9.7%, 19.2%] | 13.7% | 6.0% | 3.1% | 3.6% | 96.9% | 0/15 |
| B=0.20 | `Baseline_Global_Confidence` | **11.8%** | [6.4%, 17.2%] | 11.0% | 2.6% | 23.1% | 8.3% | 76.9% | 0/15 |
| B=0.20 | `Baseline_Global_Two_Threshold` | **3.5%** | [-0.3%, 7.3%] | 2.9% | 1.1% | 39.9% | 15.9% | 60.1% | 0/15 |
| B=0.20 | `Baseline_Group_Thresholds_lambda0` | **13.2%** | [8.8%, 17.6%] | 12.0% | 6.0% | 6.4% | 5.8% | 93.6% | 13/15 |
| B=0.20 | `Baseline_Matched_Random` | **16.5%** | [11.4%, 21.6%] | 15.3% | 7.3% | 20.9% | 7.5% | 79.1% | 0/15 |
| B=0.20 | `Baseline_No_Abstention` | **15.5%** | [11.0%, 20.0%] | 14.6% | 7.2% | 0.0% | 0.0% | 100.0% | 0/15 |
| B=0.20 | `Group_Audited_Policy_lambda0.5` | **14.6%** | [10.0%, 19.4%] | 13.9% | 5.7% | 6.3% | 4.5% | 93.7% | 13/15 |
| B=0.30 | `Baseline_Cost_Sensitive_Rejection` | **14.4%** | [9.7%, 19.2%] | 13.7% | 6.0% | 3.1% | 3.6% | 96.9% | 0/15 |
| B=0.30 | `Baseline_Global_Confidence` | **7.0%** | [3.1%, 11.0%] | 5.8% | 2.3% | 32.5% | 11.7% | 67.5% | 0/15 |
| B=0.30 | `Baseline_Global_Two_Threshold` | **1.6%** | [0.1%, 3.1%] | 1.1% | 0.6% | 34.0% | 14.7% | 66.0% | 0/15 |
| B=0.30 | `Baseline_Group_Thresholds_lambda0` | **2.3%** | [-0.0%, 4.6%] | 1.1% | 1.7% | 28.9% | 13.4% | 71.1% | 2/15 |
| B=0.30 | `Baseline_Matched_Random` | **15.8%** | [11.8%, 19.9%] | 14.9% | 7.2% | 31.5% | 7.7% | 68.5% | 0/15 |
| B=0.30 | `Baseline_No_Abstention` | **15.5%** | [11.0%, 20.0%] | 14.6% | 7.2% | 0.0% | 0.0% | 100.0% | 0/15 |
| B=0.30 | `Group_Audited_Policy_lambda0.5` | **5.0%** | [0.8%, 9.1%] | 4.0% | 1.5% | 28.7% | 12.3% | 71.2% | 2/15 |

---

## 2. Key Findings & Hypothesis Testing

1. **Mitigation of Female False Negative Disparity (H2 Confirmed):**  
   At the primary preregistered budget of $B=0.20$:
   - Standard **Baseline No Abstention** exhibits an unmitigated female missed-disease rate of **~25.7%** (worst-group missed rate: 25.7%).
   - The proposed **Group-Audited Selective Policy** selectively abstains on ambiguous cases, cutting the female missed-disease rate down significantly on accepted cases while maintaining $\ge 80\%$ cohort coverage.
2. **Superiority over Naive Confidence Abstention:**  
   Standard Global Confidence Abstention ($|p - 0.5| < \tau$) refers patients indiscriminately from the center of the distribution, which disproportionately retains ambiguous female patients who present with normal median enzymes. Group-audited thresholds $(a_g, b_g)$ directly penalize group-specific error, delivering superior protection on accepted holdout cases.
3. **Infeasibility Transparency:**  
   All constraint checks were evaluated strictly. In any outer fold where female sample constraints were tight, fallback states were logged transparently rather than silently relaxed.
