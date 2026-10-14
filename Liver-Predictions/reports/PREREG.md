# Preregistered Analysis Plan (PREREG.md)

**Title:** Referral-Budgeted, Sex-Audited Selective Classification for Small-Sample Liver Laboratory Data  
**Date of Preregistration:** October 2026 (Committed prior to execution of Phase 10 nested selection and Phase 11 policy optimization)  
**Repository:** `Liver-Predictions`  
**Lead Researcher:** Senior ML Researcher & Reproducibility Engineer  

---

## 1. Study Scope & Ethical Boundaries

1. **Retrospective Evaluation Only:** All evaluations assess classification on historical records from the Indian Liver Patient Dataset (ILPD, N=583) and secondary replication on the UCI Hepatitis C Dataset (HCV, N=615).
2. **Zero Clinical Claims:** No claims are made regarding prospective clinical utility, mortality reduction, disease progression prevention, or early diagnostic superiority.
3. **Target Definition:** Positive class = 1 (Liver Disease / Confirmed Hepatitis/Fibrosis/Cirrhosis), Negative class = 0 (Non-Disease / Healthy Donor).

---

## 2. Core Research Hypotheses

- **H1 (Nested Selection Validity):** Post-hoc selection of model configurations (e.g. SMOTE + Sigmoid) and classification thresholds ($t^* = 0.51$) over pooled outer-fold predictions introduces optimistic selection bias ("winner's curse"). Completely nested inner-fold selection will quantify this **selection optimism gap**, yielding lower but strictly leak-free performance estimates.
- **H2 (Subgroup-Audited Selective Classification):** An automated biomarker classifier exhibits severe demographic error disparity on female patients (baseline FNR ~25% vs ~7% for males). Formulating a 3-way decision policy with group-specific referral thresholds $(a_g, b_g)$ constrained by a fixed referral budget $B \in \{0.10, 0.20, 0.30\}$ will significantly reduce the worst-group missed-disease rate on accepted patients.
- **H3 (Referral Parity Trade-off):** Introducing a fairness penalty $\lambda \cdot |\text{Referral}_{\text{female}} - \text{Referral}_{\text{male}}|$ will bound referral rate disparities across sex without degrading selective accuracy beyond a prespecified margin ($\le 2\%$).

---

## 3. Experimental Design & Cross-Validation Protocol

### 3.1 Outer Evaluation Harness
- **Method:** 5x3 Repeated Stratified K-Fold (15 outer folds).
- **Stratification:** Joint distribution of biological sex and target label (`strata = sex + "_" + target`), guaranteeing balanced demographic representation across all folds.
- **Isolation:** Outer test folds are strictly held out. Zero model selection, hyperparameter tuning, resampling selection, calibration fitting, or threshold searching touches outer folds.

### 3.2 Inner Selection Harness (Nested CV)
- **Method:** 3-fold Stratified K-Fold applied strictly to the training split of each outer fold ($N_{\text{train}} \approx 466$ samples).
- **Candidate Model Grid (48 Configurations per Inner Fold):**
  - **Resampling Methods:** `[none, class_weight, RandomOverSampler, SMOTENC]`  
    *(Note: `SMOTENC` specifies categorical index for `is_male`).*
  - **Probability Calibration:** `[none, sigmoid, isotonic]`
  - **Base Estimators:**
    - `LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)`
    - `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)`
    - `HistGradientBoostingClassifier(max_depth=3, random_state=42)`
    - `KNeighborsClassifier(n_neighbors=15, weights='uniform')`
- **Inner Selection Objective:**
  $$\text{Score}_{\text{inner}} = \text{AUROC}_{\text{inner OOF}} - \text{Brier}_{\text{inner OOF}}$$
- **Deterministic Tie-Breaking:**
  1. Higher inner OOF AUROC
  2. Lower inner OOF Brier score
  3. Simpler parametric family: $\text{LogisticRegression} > \text{KNeighbors} > \text{HistGradientBoosting} > \text{RandomForest}$

### 3.3 Inner High-Sensitivity Threshold Selection ($t^*$)
- Target clinical sensitivity: $\ge 0.90$ on disease.
- Search grid: $t \in [0.10, 0.90]$ with step size $0.01$.
- Criterion: Highest threshold $t$ evaluated on inner OOF probabilities achieving inner $\text{Recall} \ge 0.90$. If no threshold achieves $0.90$, select the threshold that maximizes inner recall.

---

## 4. Prespecified Selective Triage Policy (Phase 11)

### 4.1 Decision Rule
For patient biomarker vector $x$ belonging to demographic group $g \in \{\text{Male}, \text{Female}\}$, let $p = \hat{P}(Y=1 \mid x)$:
$$D(x) = \begin{cases}
0 & \text{if } p \le a_g \quad (\text{Automated Negative}) \\
1 & \text{if } p \ge b_g \quad (\text{Automated Positive}) \\
\text{REFER} & \text{if } a_g < p < b_g \quad (\text{Specialist Referral / Abstention})
\end{cases}$$

### 4.2 Policy Optimization Objective (Inner Data Only)
Given an overall referral budget $B \in \{0.10, 0.20, 0.30\}$ and tolerance $\epsilon = 0.02$:
$$\min_{(a_{\text{M}}, b_{\text{M}}), (a_{\text{F}}, b_{\text{F}})} \max_{g \in \{\text{M}, \text{F}\}} \text{MissedDiseaseRate}_g + \lambda \cdot |\text{Referral}_{\text{F}} - \text{Referral}_{\text{M}}|$$
subject to:
1. $\text{Overall Referral Rate} \le B + \epsilon$
2. $\text{Group Coverage}_g = \frac{N_{\text{accepted}, g}}{N_g} \ge 0.50$ for all $g$
3. $0.05 \le a_g < b_g \le 0.95$

Grid search over threshold candidates $a_g \in \{0.20, 0.25, 0.30, 0.35, 0.40\}$, $b_g \in \{0.60, 0.65, 0.70, 0.75, 0.80\}$, and $\lambda \in \{0.0, 0.2, 0.5, 1.0\}$.

### 4.3 Infeasibility Handling
If no threshold pair satisfies the budget and coverage constraints on female inner folds ($N_{\text{female, train}} \approx 113$), the algorithm must **NOT** silently relax constraints. It must register `infeasible=True` and default to the global confidence threshold at budget $B$.

---

## 5. Benchmarks & Baselines at Matched Referral Budget

Every selective policy is evaluated against 6 matched-budget baselines:
1. **No Abstention:** Standard single threshold $t^*$ across all cases.
2. **Global Confidence Abstention:** Abstain when $|p - 0.5| < \tau$, tuning $\tau$ to match budget $B$.
3. **Matched Random Abstention:** Uniform random referral of fraction $B$ of patients.
4. **Global Two-Threshold:** Single pair $(a, b)$ applied universally without sex conditioning.
5. **Group Thresholds Without Gap Penalty ($\lambda = 0$):** Unconstrained group thresholds.
6. **AUCROSS (Pugnana & Ruggieri):** AUC-based selective classification baseline.

---

## 6. Statistical Analysis & Inference Protocol

1. **Patient-Level Paired Bootstrap (2,000 resamples):**  
   All confidence intervals (95% CI) and hypothesis tests use patient-level bootstrap resampling.
2. **Repeat-Aware Aggregation:**  
   Predictions from the 3 repetitions are averaged per patient prior to bootstrapping. Folds and repetitions are never treated as independent samples.
3. **Elimination of Fold-Level Tests:**  
   Zero paired t-tests or DeLong tests on overlapping CV folds.
4. **Multiplicity Correction:**  
   The primary comparison is Group-Audited Selective Policy vs. Global Confidence Abstention at $B=0.20$. All other comparisons are marked exploratory and adjusted via Holm-Bonferroni.

---

## 7. Preregistration Sign-Off

This document serves as the immutable preregistration specification for Phases 10–17. Any post-hoc deviation will be disclosed explicitly under a dedicated "Deviations from Preregistration" heading.
