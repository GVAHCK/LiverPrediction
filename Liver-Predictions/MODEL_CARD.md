# Model Card: Liverify Production Model (v2.0.0)

Jump to section:
- [Model Details](#model-details)
- [Intended Use](#intended-use)
- [Factors & Subgroups](#factors--subgroups)
- [Metrics](#metrics)
- [Evaluation Data](#evaluation-data)
- [Training Data](#training-data)
- [Quantitative Analyses](#quantitative-analyses)
- [Demographic Fairness & Sex Audit](#demographic-fairness--sex-audit)
- [Ethical Considerations & Clinical Disclaimers](#ethical-considerations--clinical-disclaimers)

---

## Model Details

- **Model Version:** 2.0.0
- **Model Date:** October 2026
- **Model Architecture:** `CalibratedClassifierCV` (Method: Sigmoid / Platt Scaling, 5-fold CV) wrapping an `imblearn.pipeline.Pipeline`:
  1. **Preprocessing:** `ColumnTransformer` applying median imputation with missingness indicators + `FunctionTransformer(np.log1p)` on skewed enzymes (`Total_Bilirubin`, `Direct_Bilirubin`, `Alkaline_Phosphotase`, `Alamine_Aminotransferase`, `Aspartate_Aminotransferase`) + `StandardScaler()`.
  2. **Resampling:** `SMOTE(random_state=42, k_neighbors=5)` applied to training folds only.
  3. **Base Estimator:** `LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)`.
- **Packaging:** Serialized bundle via `joblib` into `app/production_model.joblib`. Contains calibrated pipeline, exact feature ordering (10 features), SHA256 dataset hash, and clinical triage thresholds.
- **Training Data Hash (SHA256):** `9d90de6469ed034c209a23f3d0c78133e2f4b991e2b72ac86a8a9c09368238f6` (Raw ILPD).

---

## Intended Use

- **Primary Intended Use:**  
  Retrospective risk screening and exploratory decision-support research on standard biochemical liver function test (LFT) panels.
- **Intended Users:**  
  Machine learning researchers, clinical data scientists, and informatics auditors evaluating leak-free tabular biomarker pipelines.
- **Out-of-Scope Use Cases:**  
  - **NOT** for standalone, direct-to-consumer clinical diagnosis without physician oversight.
  - **NOT** for critical emergency care, acute toxicity assessment, or pediatric triage.
  - **NOT** intended to replace imaging (ultrasound, CT, MRI), elastography (FibroScan), or liver biopsy.

---

## Factors & Subgroups

- **Demographic Factors:** Biological sex (Male vs. Female) and Age (1–120 years).
- **Clinical Biomarkers (10 Features):**
  1. Age
  2. Total Bilirubin (TB)
  3. Direct Bilirubin (DB)
  4. Alkaline Phosphatase (AP)
  5. Alanine Aminotransferase (ALT / AA)
  6. Aspartate Aminotransferase (AST / ASA)
  7. Total Proteins (TP)
  8. Albumin (ALB / AB)
  9. Albumin-and-Globulin Ratio (A/G Ratio)
  10. `is_male` (1 = Male, 0 = Female)

---

## Metrics

All evaluations report comprehensive clinical and probabilistic metrics across a 5x3 Repeated Stratified K-Fold harness (15 outer folds, 583 patients):

- **Discrimination:** Area Under ROC Curve (AUROC), Average Precision (AP / PR-AUC).
- **Clinical Utility:** Sensitivity / Recall on Disease (Class 1), Specificity on Non-Disease (Class 0), F1-Score.
- **Probabilistic Calibration:** Brier Score Loss, Cox Calibration Slope, Cox Calibration Intercept.

---

## Evaluation Data

- **Validation Harness:** 5x3 Repeated Stratified K-Fold (15 outer evaluation iterations). Stratification enforced on joint distribution `(sex x target)`.
- **Sensitivity Split:** 5-fold Grouped CV ensuring exact duplicate patient profiles (13 pairs identified in ILPD) are strictly grouped into the same fold, preventing optimistic leakage across train/validation splits.

---

## Training Data

- **Source:** Indian Liver Patient Dataset (ILPD), collected from North-East Andhra Pradesh, India. Preserved immutably at `data/raw/indian_liver_patient.csv`.
- **Cohort Size:** 583 rows (416 Class 1 / Liver Disease, 167 Class 0 / Non-Disease).
- **Target Recoding:** Explicit mapping conforming to standard medical ML practice (`1 -> 1` Disease, `2 -> 0` Non-Disease).

---

## Quantitative Analyses

### Full 5x3 Cross-Validation Performance (15 Folds, N=8,745 Evaluations)

| Metric | Mean | 95% Confidence Interval | Std Dev |
|:---|:---:|:---:|:---:|
| **AUROC** | **0.7548** | [0.7450, 0.7647] | 0.0175 |
| **Average Precision (AP)** | **0.8718** | [0.8601, 0.8834] | 0.0207 |
| **Brier Score** | **0.1722** | [0.1668, 0.1776] | 0.0096 |
| **Calibration Slope** | **1.054** | [0.932, 1.176] | 0.218 |
| **Calibration Intercept** | **-0.038** | [-0.154, 0.078] | 0.207 |
| **Recall (at $t^*=0.51$)** | **88.8%** | [85.5%, 92.1%] | 0.038 |
| **Specificity (at $t^*=0.51$)** | **44.9%** | [39.1%, 50.7%] | 0.065 |

### 3-Tier Clinical Triage Performance

To mitigate high false negative rates inherent in single binary cuts, the production model exposes a 3-way triage policy:

1. **Low Risk ($P < 0.35$):** 96 cases (16.5% cohort). Specificity 69.8%, 91.7% Negative Predictive Value (NPV).
2. **Uncertain / Indeterminate ($0.35 \le P \le 0.75$):** 287 cases (49.2% cohort). Model abstains from automated classification; flags for clinical evaluation and follow-up imaging.
3. **High Risk ($P > 0.75$):** 200 cases (34.3% cohort). Positive Predictive Value (PPV) 86.5%, Sensitivity 82.3%.
- **Accuracy on Decisive Cases ($P < 0.35$ or $P > 0.75$):** **80.0%**.

---

## Demographic Fairness & Sex Audit

Auditing model performance across biological sex revealed substantial demographic disparity driven by dataset imbalance (75.6% Male vs. 24.4% Female):

| Demographic Group | Count in Cohort | Disease Prevalence | Sensitivity (Recall) | False Negative Rate (FNR) |
|:---|:---:|:---:|:---:|:---:|
| **Male Patients** | 441 (75.6%) | 73.5% (324) | **92.9%** | **7.1%** |
| **Female Patients** | 142 (24.4%) | 64.8% (92) | **74.3%** | **25.7%** |

### False Negative Error Profile
Detailed error profiling (`reports/06_statistics_ablations.md`) confirmed that female patients represent **34.6% of all false negatives** despite comprising only 22.1% of disease cases. False negative cases systematically present with **deceptive normal median bilirubin (0.80 mg/dL)** and **normal transaminases (ALT 25 U/L, AST 27 U/L)**. The model card explicitly warns that normal standard enzyme levels in female patients do NOT rule out underlying pathology (e.g., non-alcoholic fatty liver disease, autoimmune hepatitis).

---

## Ethical Considerations & Clinical Disclaimers

> [!CAUTION]
> **CRITICAL MEDICAL DISCLAIMER**  
> This software is an experimental retrospective statistical research artifact. It is NOT certified by the FDA, CE, CDSCO, or any regulatory medical authority.  
> 1. Automated predictions must **never** be used as a standalone diagnostic determinant.  
> 2. Clinicians and researchers must be cognizant of the **3.6x higher false negative rate in female patients**.  
> 3. Treatment decisions, medication alterations, and clinical management must rely exclusively on comprehensive physical examinations, comprehensive lab panels, and consultation with licensed gastroenterologists or hepatologists.
