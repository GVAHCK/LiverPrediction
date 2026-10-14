import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.cv import (
    SPLITS_DIR,
    PROCESSED_DATA_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    compute_metrics,
    create_model_pipeline
)

def test_splits_no_data_leakage():
    """Verify zero overlap between train and test indices in every fold."""
    outer_file = os.path.join(SPLITS_DIR, "outer_splits_5x3.json")
    with open(outer_file, "r") as f:
        splits = json.load(f)

    assert len(splits) == 15
    for split in splits:
        train_idx = set(split['train_indices'])
        test_idx = set(split['test_indices'])
        assert len(train_idx.intersection(test_idx)) == 0, "Train and test overlap detected!"
        assert len(train_idx) + len(test_idx) == 583

def test_pipeline_leak_free_execution():
    """Verify Pipeline fits exclusively on train set and transforms test set without error."""
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df['target']

    train_idx = list(range(400))
    test_idx = list(range(400, 583))

    pipe = create_model_pipeline(LogisticRegression())
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

    probs = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    assert len(probs) == len(test_idx)
    assert not np.isnan(probs).any()

def test_compute_metrics_comprehensive():
    """Verify all 7 required metrics (Rule 3) are reported."""
    y_true = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.85, 0.3, 0.4, 0.95, 0.15])
    m = compute_metrics(y_true, y_prob)

    required_keys = ['auroc', 'ap', 'recall', 'specificity', 'f1', 'brier', 'cal_slope', 'cal_intercept']
    for k in required_keys:
        assert k in m, f"Missing metric {k}"
        assert not np.isnan(m[k]), f"Metric {k} is NaN"

    assert m['recall'] == 1.0
    assert m['specificity'] == 1.0
    assert m['auroc'] == 1.0
