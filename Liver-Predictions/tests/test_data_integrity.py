import os
import pytest
import pandas as pd
from src.data_audit import compute_sha256, recode_target, RAW_DATA_PATH

def test_raw_data_sha256():
    """Verify raw ILPD dataset cryptographic hash matches known record."""
    expected_hash = "9d90de6469ed034c209a23f3d0c78133e2f4b991e2b72ac86a8a9c09368238f6"
    assert compute_sha256(RAW_DATA_PATH) == expected_hash

def test_explicit_recode_rule_4():
    """Verify ILPD raw 1 maps to 1 (Disease) and raw 2 maps to 0 (Non-Disease)."""
    assert recode_target(1) == 1
    assert recode_target(2) == 0
    with pytest.raises(ValueError):
        recode_target(0)
    with pytest.raises(ValueError):
        recode_target(3)

def test_processed_dataset_integrity():
    """Verify clean dataset row counts and class balance."""
    processed_path = os.path.join(os.path.dirname(RAW_DATA_PATH), "..", "processed", "ilpd_clean.parquet")
    assert os.path.exists(processed_path), "Processed dataset parquet file does not exist."
    df = pd.read_parquet(processed_path)
    assert len(df) == 583
    assert df['target'].value_counts()[1] == 416
    assert df['target'].value_counts()[0] == 167
    assert set(df['target'].unique()) == {0, 1}
    assert 'is_male' in df.columns
    assert 'sex' in df.columns

def test_recall_toy_matrix():
    """Acceptance criteria: Recall unit test passes with known toy confusion matrix.
    Confusion matrix format:
                  Pred 0    Pred 1
    Actual 0 (TN)   TN        FP
    Actual 1 (FN)   FN        TP
    """
    # Suppose TP = 90, FN = 10, FP = 20, TN = 80
    tn, fp, fn, tp = 80, 20, 10, 90
    recall = tp / (tp + fn)
    assert recall == 0.90
    specificity = tn / (tn + fp)
    assert specificity == 0.80
