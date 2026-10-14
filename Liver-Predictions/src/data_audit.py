import os
import hashlib
import json
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "indian_liver_patient.csv"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def recode_target(val: int) -> int:
    """Explicit recode: 1 -> 1 (Liver Disease), 2 -> 0 (Non-Disease)."""
    if val == 1:
        return 1
    elif val == 2:
        return 0
    else:
        raise ValueError(f"Unexpected target label: {val}")

def audit_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    sha256 = compute_sha256(RAW_DATA_PATH)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    n_rows, n_cols = df_raw.shape

    # Missing values
    missing_counts = df_raw.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].to_dict()

    # Duplicates check (all columns)
    exact_duplicates_mask = df_raw.duplicated(keep=False)
    n_exact_duplicate_rows = df_raw.duplicated(keep='first').sum()
    n_exact_duplicate_total_rows = exact_duplicates_mask.sum()

    # Conflicting duplicates (same features, different label)
    feature_cols = [c for c in df_raw.columns if c != 'Dataset']
    conflicting_mask = df_raw.duplicated(subset=feature_cols, keep=False) & ~df_raw.duplicated(keep=False)
    n_conflicting_rows = conflicting_mask.sum()

    # Target class distribution (raw)
    raw_class_counts = df_raw['Dataset'].value_counts().to_dict()

    # Recode target (Rule 4: 1 -> 1, 2 -> 0)
    df_clean = df_raw.copy()
    df_clean['target'] = df_clean['Dataset'].apply(recode_target)
    
    # Sex distribution & sex x target breakdown
    # Encode sex as binary 1=Male, 0=Female; keep raw 'sex' string for auditing only
    df_clean['sex'] = df_clean['Gender']  # String column for stratification/auditing
    df_clean['is_male'] = (df_clean['Gender'].str.strip().str.lower() == 'male').astype(int)

    recoded_class_counts = df_clean['target'].value_counts().to_dict()
    sex_target_crosstab = pd.crosstab(df_clean['sex'], df_clean['target'], margins=True)

    # Save processed dataset
    clean_csv_path = os.path.join(PROCESSED_DIR, "ilpd_clean.csv")
    clean_parquet_path = os.path.join(PROCESSED_DIR, "ilpd_clean.parquet")
    df_clean.to_csv(clean_csv_path, index=False)
    df_clean.to_parquet(clean_parquet_path, index=False)

    # Deduplicated version for sensitivity analysis
    df_dedup = df_clean.drop_duplicates(subset=feature_cols, keep='first').copy()
    dedup_parquet_path = os.path.join(PROCESSED_DIR, "ilpd_dedup.parquet")
    df_dedup.to_parquet(dedup_parquet_path, index=False)

    # Write Markdown Report
    report_content = f"""# Phase 1: Data Integrity & Provenance Audit

**File Analyzed:** `data/raw/indian_liver_patient.csv`  
**SHA-256 Checksum:** `{sha256}`  
**Dimensions:** {n_rows} rows, {n_cols} columns  

---

## 1. Provenance & Cryptographic Integrity

- **Filename:** `indian_liver_patient.csv`
- **SHA256:** `{sha256}`
- **Total Records:** {n_rows}
- **Features:** {n_cols - 1} predictor features + 1 target (`Dataset`)

---

## 2. Missing Value Profile

| Column | Missing Count | Percentage (%) |
| :--- | :--- | :--- |
"""
    for col, count in missing_cols.items():
        pct = (count / n_rows) * 100
        report_content += f"| `{col}` | {count} | {pct:.2f}% |\n"

    report_content += f"""
*Note: All other {n_cols - len(missing_cols)} columns have 0 missing values.*

---

## 3. Duplicate & Conflicting Records Analysis

- **Exact Duplicate Rows (redundant instances):** {n_exact_duplicate_rows} redundant rows ({n_exact_duplicate_total_rows} rows involved in identical pairs/clusters).
- **Conflicting Label Records (identical labs, contradictory diagnosis):** {n_conflicting_rows} rows.
- **Deduplicated Patient Count:** {len(df_dedup)} unique clinical instances.
- **Handling Protocol:** 
  1. Main pipeline retains complete raw instances but groups duplicate clusters together in sensitivity splits to prevent leakage.
  2. Dedicated sensitivity benchmark run on strictly deduplicated data (`/data/processed/ilpd_dedup.parquet`).

---

## 4. Target Recoding & Class Balance (Hard Rule 4)

- **Raw Labels in ILPD:** `1` = Liver Disease Patient, `2` = Non-Liver Patient (Healthy Control).
- **Explicit Recoding:**
  - `Dataset == 1` $\\rightarrow$ `target = 1` (Positive / Liver Disease)
  - `Dataset == 2` $\\rightarrow$ `target = 0` (Negative / Non-Disease)
- **Recoded Class Counts:**
  - **Class 1 (Disease):** {recoded_class_counts[1]} ({recoded_class_counts[1]/n_rows*100:.2f}%)
  - **Class 0 (Non-Disease):** {recoded_class_counts[0]} ({recoded_class_counts[0]/n_rows*100:.2f}%)
  - **Imbalance Ratio:** {recoded_class_counts[1] / recoded_class_counts[0]:.2f}:1

---

## 5. Sex-Stratified Breakdown

```
{sex_target_crosstab.to_string()}
```

- **Male Patients:** {(df_clean['sex'] == 'Male').sum()} ({(df_clean['sex'] == 'Male').sum() / n_rows * 100:.1f}%)
  - Disease Prevalence in Males: {(df_clean[df_clean['sex'] == 'Male']['target'] == 1).mean() * 100:.1f}%
- **Female Patients:** {(df_clean['sex'] == 'Female').sum()} ({(df_clean['sex'] == 'Female').sum() / n_rows * 100:.1f}%)
  - Disease Prevalence in Females: {(df_clean[df_clean['sex'] == 'Female']['target'] == 1).mean() * 100:.1f}%

*Critical Observation (L08):* Female patients represent less than a quarter of the dataset. Models evaluated on accuracy alone risk extreme false-negative rates on females.

---

## 6. Generated Artifacts

- Cleaned Dataset with explicit target: `data/processed/ilpd_clean.parquet`
- Deduplicated Dataset: `data/processed/ilpd_dedup.parquet`
"""

    report_path = os.path.join(REPORTS_DIR, "01_data_audit.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Data audit completed.")
    print(f"SHA-256: {sha256}")
    print(f"Rows: {n_rows}, Exact duplicate pairs: {n_exact_duplicate_rows}, Conflicting: {n_conflicting_rows}")
    print(f"Class counts: {recoded_class_counts}")
    return sha256, n_rows, recoded_class_counts, n_exact_duplicate_rows

if __name__ == "__main__":
    audit_dataset()
