# Phase 1: Data Integrity & Provenance Audit

**File Analyzed:** `data/raw/indian_liver_patient.csv`  
**SHA-256 Checksum:** `9d90de6469ed034c209a23f3d0c78133e2f4b991e2b72ac86a8a9c09368238f6`  
**Dimensions:** 583 rows, 11 columns  

---

## 1. Provenance & Cryptographic Integrity

- **Filename:** `indian_liver_patient.csv`
- **SHA256:** `9d90de6469ed034c209a23f3d0c78133e2f4b991e2b72ac86a8a9c09368238f6`
- **Total Records:** 583
- **Features:** 10 predictor features + 1 target (`Dataset`)

---

## 2. Missing Value Profile

| Column | Missing Count | Percentage (%) |
| :--- | :--- | :--- |
| `Albumin_and_Globulin_Ratio` | 4 | 0.69% |

*Note: All other 10 columns have 0 missing values.*

---

## 3. Duplicate & Conflicting Records Analysis

- **Exact Duplicate Rows (redundant instances):** 13 redundant rows (26 rows involved in identical pairs/clusters).
- **Conflicting Label Records (identical labs, contradictory diagnosis):** 0 rows.
- **Deduplicated Patient Count:** 570 unique clinical instances.
- **Handling Protocol:** 
  1. Main pipeline retains complete raw instances but groups duplicate clusters together in sensitivity splits to prevent leakage.
  2. Dedicated sensitivity benchmark run on strictly deduplicated data (`/data/processed/ilpd_dedup.parquet`).

---

## 4. Target Recoding & Class Balance (Hard Rule 4)

- **Raw Labels in ILPD:** `1` = Liver Disease Patient, `2` = Non-Liver Patient (Healthy Control).
- **Explicit Recoding:**
  - `Dataset == 1` $\rightarrow$ `target = 1` (Positive / Liver Disease)
  - `Dataset == 2` $\rightarrow$ `target = 0` (Negative / Non-Disease)
- **Recoded Class Counts:**
  - **Class 1 (Disease):** 416 (71.36%)
  - **Class 0 (Non-Disease):** 167 (28.64%)
  - **Imbalance Ratio:** 2.49:1

---

## 5. Sex-Stratified Breakdown

```
target    0    1  All
sex                  
Female   50   92  142
Male    117  324  441
All     167  416  583
```

- **Male Patients:** 441 (75.6%)
  - Disease Prevalence in Males: 73.5%
- **Female Patients:** 142 (24.4%)
  - Disease Prevalence in Females: 64.8%

*Critical Observation (L08):* Female patients represent less than a quarter of the dataset. Models evaluated on accuracy alone risk extreme false-negative rates on females.

---

## 6. Generated Artifacts

- Cleaned Dataset with explicit target: `data/processed/ilpd_clean.parquet`
- Deduplicated Dataset: `data/processed/ilpd_dedup.parquet`
