import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

from src.cv import (
    PROCESSED_DATA_PATH,
    ALL_MODEL_FEATURES,
    build_preprocessing_pipeline
)
from src.data_audit import compute_sha256, RAW_DATA_PATH

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "app")
MODEL_OUTPUT_PATH = os.path.join(APP_DIR, "production_model.joblib")

def export_model(seed=42):
    """
    Exports the v3.0.0 research prototype model bundle.
    
    CRITICAL METHODOLOGICAL NOTICE (Fixes E7):
    This model is trained on the full cohort strictly for software serving and demonstration.
    NO scientific performance metrics, sensitivity estimates, or claims are derived from this artifact.
    All reported metrics are derived exclusively from nested cross-validation (reports/10_nested_results.json
    and reports/11_policy_results.json).
    """
    os.makedirs(APP_DIR, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = df[ALL_MODEL_FEATURES]
    y = df['target'].values

    print(f"[v3.0.0] Training production prototype bundle on {len(df)} samples...")

    # Winning nested architecture: Preprocessor -> SMOTENC -> LogisticRegression (C=0.1) -> Sigmoid Calibration
    base_pipeline = ImbPipeline([
        ('preprocessor', build_preprocessing_pipeline()),
        ('resampler', SMOTENC(categorical_features=[-1], random_state=seed)),
        ('model', LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=seed))
    ])

    calibrated_pipeline = CalibratedClassifierCV(estimator=base_pipeline, method='sigmoid', cv=5)
    calibrated_pipeline.fit(X, y)

    data_hash = compute_sha256(RAW_DATA_PATH)
    model_version = "3.0.0"

    bundle = {
        'model_version': model_version,
        'data_hash': data_hash,
        'created_at': datetime.utcnow().isoformat() + "Z",
        'model_type': 'CalibratedClassifierCV(Sigmoid, LogisticRegression L2, C=0.1, SMOTENC)',
        'feature_names': ALL_MODEL_FEATURES,
        'purpose': 'RESEARCH_PROTOTYPE_DEMONSTRATION_ONLY',
        'metrics_source': 'reports/10_nested_results.json and reports/11_policy_results.json (Nested CV Only)',
        'decision_thresholds': {
            'target_recall_threshold': 0.50, # High sensitivity screening threshold
            'low_risk_threshold': 0.35,
            'high_risk_threshold': 0.70,
            'global_thresholds': {
                'low_risk_threshold': 0.35,
                'high_risk_threshold': 0.70
            },
            'sex_aware_thresholds': {
                'male': {'a_m': 0.35, 'b_m': 0.70},
                'female': {'a_f': 0.35, 'b_f': 0.65}
            }
        },
        'clinical_warnings': {
            'tradeoff': "High sensitivity (91.2%) operates at binary specificity of 23.0%. Selective referral is recommended for borderline cases.",
            'female_disparity': "Female liver patients frequently present with normal median bilirubin (0.70 mg/dL) and ALT (16 U/L). Normal enzymes do not rule out pathology."
        },
        'pipeline': calibrated_pipeline
    }

    joblib.dump(bundle, MODEL_OUTPUT_PATH)
    print(f"Production model bundle exported -> {MODEL_OUTPUT_PATH}")
    print(f"Version: {model_version}, Data Hash: {data_hash}")
    return bundle

if __name__ == "__main__":
    export_model()
