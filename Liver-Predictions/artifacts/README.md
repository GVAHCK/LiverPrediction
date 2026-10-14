# Artifacts & Checksums Registry

This directory contains persistent, immutable evaluation artifacts generated during the study:
"Referral-Budgeted, Sex-Audited Selective Classification for Small-Sample Liver Laboratory Data."

## 1. Directory Structure

- `splits/`:
  - `outer_splits_5x3.json`: 5-fold $\times$ 3 repeats stratified cross-validation train/test patient indices (Seed 42).
  - `grouped_splits_5.json`: 5-fold grouped cross-validation splits ensuring exact duplicate vectors never cross folds.
- `predictions/`:
  - `10_nested_outer_predictions.parquet`: Outer-fold holdout predictions ($N=15$ folds) containing fold ID, patient index, ground truth label, recorded sex, predicted probability, and threshold decision.
  - `11_policy_predictions.parquet`: Holdout predictions under the sex-audited policy and all 6 matched-budget baselines across referral budgets ($B \in \{0.10, 0.20, 0.30\}$).
- `models/`:
  - `production_model.joblib`: Pre-trained, calibrated model bundle (v3.0.0) shipped for software demonstration and serving (strictly marked as serving prototype; zero reported metrics derived from it).
- `HASHES.sha256`: SHA-256 cryptographic checksums for all stored artifacts.

## 2. Integrity Verification
To verify the integrity of all artifacts against their declared hashes:
```bash
sha256sum -c HASHES.sha256
```
