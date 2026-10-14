import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ilpd_clean.parquet")

NUMERIC_FEATURES = [
    'Age',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase',
    'Total_Protiens',
    'Albumin',
    'Albumin_and_Globulin_Ratio'
]

SKEWED_FEATURES = [
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase'
]

UNSKEWED_NUMERIC = [
    'Age',
    'Total_Protiens',
    'Albumin',
    'Albumin_and_Globulin_Ratio'
]

CATEGORICAL_FEATURES = ['is_male']

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes AUROC, AP, Recall, Specificity, F1, Brier score,
    and Calibration Slope & Intercept (Cox calibration).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).clip(1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    auroc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Calibration slope & intercept via log-odds regression
    # logit(p) = ln(p / (1-p))
    log_odds = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    try:
        # Fit unregularized or very lightly penalized logistic regression
        cal_lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        cal_lr.fit(log_odds, y_true)
        cal_slope = float(cal_lr.coef_[0][0])
        cal_intercept = float(cal_lr.intercept_[0])
    except Exception:
        cal_slope = np.nan
        cal_intercept = np.nan

    return {
        'auroc': float(auroc),
        'ap': float(ap),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'brier': float(brier),
        'cal_slope': float(cal_slope),
        'cal_intercept': float(cal_intercept),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

def build_preprocessing_pipeline(apply_log1p=True, scale=True):
    """
    Constructs a ColumnTransformer that handles:
    1. Median imputation (with missing indicator flag for A/G ratio)
    2. Optional log1p transformation on skewed lab markers
    3. Optional standard scaling
    4. Categorical pass-through of binary sex feature
    """
    if apply_log1p:
        skewed_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('log1p', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', StandardScaler() if scale else 'passthrough')
        ])
    else:
        skewed_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler() if scale else 'passthrough')
        ])

    unskewed_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler() if scale else 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed', skewed_pipeline, SKEWED_FEATURES),
            ('unskewed', unskewed_pipeline, UNSKEWED_NUMERIC),
            ('cat', 'passthrough', CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def create_model_pipeline(model, resampler=None, apply_log1p=True, scale=True):
    """Builds a full imblearn Pipeline wrapping preprocessing, optional resampling, and estimator."""
    steps = [('preprocessor', build_preprocessing_pipeline(apply_log1p=apply_log1p, scale=scale))]
    if resampler is not None:
        steps.append(('resampler', resampler))
    steps.append(('model', model))
    return ImbPipeline(steps)

def generate_and_save_splits(seed=42):
    """
    Generates:
    1. Outer 5x3 RepeatedStratifiedKFold stratified on combined target + sex.
    2. Grouped 5-Fold where duplicate feature rows are strictly clustered into the same fold.
    Saves splits to /splits/*.json.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_PATH)

    # Strata: sex_target (Male_1, Male_0, Female_1, Female_0)
    strata = df['sex'].astype(str) + "_" + df['target'].astype(str)

    # 1. 5x3 Repeated Stratified K-Fold
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    outer_splits = []
    for split_id, (train_idx, test_idx) in enumerate(rskf.split(df, strata)):
        outer_splits.append({
            'split_id': split_id,
            'repeat': split_id // 5,
            'fold': split_id % 5,
            'train_indices': train_idx.tolist(),
            'test_indices': test_idx.tolist()
        })

    outer_splits_file = os.path.join(SPLITS_DIR, "outer_splits_5x3.json")
    with open(outer_splits_file, "w") as f:
        json.dump(outer_splits, f)

    # 2. Duplicate cluster grouping (for sensitivity analysis)
    # Assign unique group id to exact laboratory profiles
    feature_tuples = df[NUMERIC_FEATURES + ['Gender']].astype(str).agg(','.join, axis=1)
    _, group_ids = np.unique(feature_tuples, return_inverse=True)
    
    # We group clusters into 5 folds
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    grouped_splits = []
    for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(df, df['target'], groups=group_ids)):
        grouped_splits.append({
            'fold_id': fold_id,
            'train_indices': train_idx.tolist(),
            'test_indices': test_idx.tolist()
        })

    grouped_splits_file = os.path.join(SPLITS_DIR, "grouped_splits_5.json")
    with open(grouped_splits_file, "w") as f:
        json.dump(grouped_splits, f)

    print(f"Generated {len(outer_splits)} outer repeated stratified splits -> {outer_splits_file}")
    print(f"Generated {len(grouped_splits)} grouped sensitivity splits -> {grouped_splits_file}")

if __name__ == "__main__":
    generate_and_save_splits(seed=42)
