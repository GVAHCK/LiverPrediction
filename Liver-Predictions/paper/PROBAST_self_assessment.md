# PROBAST+AI Self-Assessment (Prediction Model Risk of Bias Assessment Tool)

This assessment evaluates potential sources of bias across the four core domains of the **PROBAST+AI framework** (*Prediction model Risk Of Bias ASsessment Tool for Artificial Intelligence*, Ann Intern Med 2024).

---

## Overall Assessment Summary

| Domain | Risk of Bias | Applicability Concerns | Key Rationale |
|:---|:---:|:---:|:---|
| **1. Participants** | **HIGH** | **HIGH** | Retrospective single-center cohort collected in Andhra Pradesh, India; subject to referral and tertiary care presentation bias. |
| **2. Predictors** | **LOW** | **LOW** | Predictors are objective, standardized serum enzyme and liver biomarker measurements recorded without knowledge of model development. |
| **3. Outcome** | **HIGH** | **HIGH** | The ILPD outcome represents retrospective clinical diagnostic coding without universal gold-standard histology (liver biopsy) or prospective follow-up. |
| **4. Analysis** | **LOW** | **LOW** | Complete nested cross-validation (5x3 outer, 3-fold inner) eliminates selection leakage; patient-level paired bootstrap replaces invalid fold tests; all metrics pre-registered. |
| **OVERALL SUMMARY** | **HIGH RISK OF BIAS** | **HIGH APPLICABILITY CONCERN** | **Classified honestly as High Risk of Bias** primarily due to retrospective small-sample participant recruitment and single-center outcome labeling. |

---

## Detailed Domain Evaluation

### Domain 1: Participants
- **1.1 Were appropriate data sources used, e.g., cohort, RCT, or nested case-control?**  
  *No.* The Indian Liver Patient Dataset (ILPD) is a cross-sectional convenience sample collected from a single regional medical center in Andhra Pradesh.
- **1.2 Were all inclusions and exclusions appropriate?**  
  *Unclear.* The original dataset documentation does not record exclusion criteria for co-morbidities (e.g., HIV, renal failure) or medication usage.
- **Risk Assessment:** **HIGH RISK**. Convenience sampling in a clinical setting induces presentation bias toward more severe disease, distorting natural population prevalence.

---

### Domain 2: Predictors
- **2.1 Were predictors defined and assessed in the same way for all participants?**  
  *Yes.* All 10 biomarker features (Age, Sex, Bilirubins, Enzymes, Proteins) followed standard clinical laboratory photometric and enzymatic assays.
- **2.2 Were predictor assessments made without knowledge of outcome data?**  
  *Yes.* Routine laboratory tests are performed blinded to downstream retrospective diagnostic classification.
- **2.3 Are all predictors available at the intended time of model use?**  
  *Yes.* All 10 biomarkers constitute a standard liver function panel available simultaneously during routine outpatient blood draws.
- **Risk Assessment:** **LOW RISK**.

---

### Domain 3: Outcome
- **3.1 Was the outcome determined appropriately?**  
  *Unclear/High.* The binary label denotes "liver disease" vs "non-disease," derived from retrospective hospital records. Histological confirmation via needle biopsy was not universally documented.
- **3.2 Was a prespecified or standard outcome definition used?**  
  *Unclear.* Outcome captures diverse etiologies (cirrhosis, hepatitis, steatosis) under a single non-differentiated binary label.
- **3.3 Were predictors excluded from the outcome definition?**  
  *No.* In clinical hepatology practice, clinicians rely partly on liver enzymes to establish clinical diagnoses (incorporation bias).
- **Risk Assessment:** **HIGH RISK**. Incorporation bias is an inherent limitation of retrospective clinical diagnostic datasets lacking independent tissue biopsy verification.

---

### Domain 4: Analysis
- **4.1 Were there a reasonable number of participants with the outcome?**  
  *Marginal.* Total cohort $N=583$ (416 positive, 167 negative). While total events ($N=416$) exceed conventional 10-events-per-variable thresholds, female subgroup sample size ($n=142$, 92 disease) is sparse.
- **4.2 Was continuous predictor categorization avoided?**  
  *Yes.* All biomarkers were preserved on their continuous physiological scales.
- **4.3 Were all participants included in the analysis without arbitrary exclusions?**  
  *Yes.* All 583 records analyzed. A/G ratio missingness ($n=4$) handled within pipeline cross-validation folds.
- **4.4 Was data leakage strictly avoided across folds?**  
  *Yes.* Zero selection or preprocessing on evaluation data. Resampling, imputation, scaling, algorithm selection, calibration, and threshold selection were executed strictly inside inner training folds.
- **4.5 Were calibration and discrimination assessed appropriately?**  
  *Yes.* Assessed via outer holdout AUROC, Average Precision, Brier score, and calibration slope/intercept.
- **4.6 Were confidence intervals computed using valid statistical methods?**  
  *Yes.* Replaced flawed fold-level t-tests and DeLong tests with 2,000x patient-level paired bootstrap resampling.
- **Risk Assessment:** **LOW RISK**. The analytical pipeline adheres to the most stringent reproducibility standards.

---

## Conclusion & Ethical Statement
In accordance with Hard Rule 4 ("Report drops honestly. A worse but valid number beats a better leaky one"), we explicitly classify the overall study as **High Risk of Bias** due to participant sampling and outcome definition constraints. The algorithms developed here serve strictly as **finite-sample methodological proofs of concept** for referral-budgeted selective triage, and must not be interpreted as validated medical diagnostic devices.
