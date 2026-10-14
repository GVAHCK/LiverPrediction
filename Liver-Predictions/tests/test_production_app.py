import pytest
import os
import sys
import math
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server import app, load_production_model, MODEL_PATH

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_production_model_bundle_exists():
    """Verify production model bundle exists and contains required metadata."""
    assert os.path.exists(MODEL_PATH), f"Model bundle missing at {MODEL_PATH}"
    bundle = load_production_model()
    required_keys = ['model_version', 'data_hash', 'feature_names', 'decision_thresholds', 'pipeline']
    for k in required_keys:
        assert k in bundle, f"Missing key '{k}' in production bundle"
    assert bundle['model_version'] == "3.0.0"
    assert len(bundle['feature_names']) == 10

def test_health_endpoint(client):
    """Verify /health returns 200, service name, model version, and data hash."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'Liverify'
    assert data['model_loaded'] is True
    assert data['model_version'] == '3.0.0'
    assert len(data['data_hash']) == 64  # SHA256 length

def test_feature_ordering_parity():
    """Verify feature names in server match the training order precisely."""
    bundle = load_production_model()
    expected_order = [
        'Age',
        'Total_Bilirubin',
        'Direct_Bilirubin',
        'Alkaline_Phosphotase',
        'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase',
        'Total_Protiens',
        'Albumin',
        'Albumin_and_Globulin_Ratio',
        'is_male'
    ]
    assert bundle['feature_names'] == expected_order

def test_post_checkup_json_healthy_low_risk(client):
    """Verify healthy patient lab values produce LOW_RISK band."""
    payload = {
        'age': 32,
        'gender': 'Female',
        'tb': 0.6,
        'db': 0.1,
        'ap': 110,
        'aa': 18,
        'asa': 19,
        'tp': 7.2,
        'ab': 4.2,
        'ag': 1.4
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['risk_band'] in ('LOW_RISK', 'UNCERTAIN')
    assert 0.0 <= data['probability'] <= 1.0
    assert 'recommendation' in data
    assert 'disclaimer' in data
    assert data['model_version'] == '3.0.0'

def test_post_checkup_json_severe_disease_high_risk(client):
    """Verify elevated liver biomarkers trigger HIGH_RISK band."""
    payload = {
        'age': 58,
        'gender': 'Male',
        'tb': 14.5,
        'db': 7.2,
        'ap': 850,
        'aa': 140,
        'asa': 220,
        'tp': 5.8,
        'ab': 2.3,
        'ag': 0.6
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['risk_band'] == 'HIGH_RISK'
    assert data['probability'] >= 0.70
    assert data['result'] == 1
    assert "Elevated probability" in data['recommendation']

def test_post_checkup_html_form_rendering(client):
    """Verify standard HTML form submission renders calibrated output card."""
    form_data = {
        'age': '45',
        'gender': '1',
        'tb': '1.2',
        'db': '0.3',
        'ap': '180',
        'aa': '35',
        'asa': '40',
        'tp': '7.0',
        'ab': '3.8',
        'ag': '1.1'
    }
    response = client.post('/check-up', data=form_data)
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Calibrated Prob:" in html
    assert "Important Clinical Disclaimer" in html
    assert "Model v3.0.0" in html

def test_validation_missing_fields(client):
    """Verify 400 Bad Request when required fields are missing."""
    incomplete = {'age': 40, 'gender': 1}
    response = client.post('/check-up', json=incomplete)
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'Missing required fields' in data['message']

def test_validation_negative_values(client):
    """Verify 400 Bad Request when negative lab values are submitted."""
    payload = {
        'age': 45,
        'gender': 1,
        'tb': -1.5,
        'db': 0.2,
        'ap': 120,
        'aa': 25,
        'asa': 30,
        'tp': 7.0,
        'ab': 3.5,
        'ag': 1.0
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert 'cannot be negative' in data['message']

def test_validation_nan_and_inf(client):
    """Verify rejection of NaN and Infinite numerical values."""
    payload = {
        'age': 45,
        'gender': 1,
        'tb': float('nan'),
        'db': 0.2,
        'ap': 120,
        'aa': 25,
        'asa': 30,
        'tp': 7.0,
        'ab': 3.5,
        'ag': 1.0
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert 'cannot be NaN' in data['message']

def test_validation_age_boundaries(client):
    """Verify age boundaries 1-120."""
    invalid_ages = [-5, 0, 125, 250]
    for invalid_age in invalid_ages:
        payload = {
            'age': invalid_age,
            'gender': 1,
            'tb': 1.0,
            'db': 0.2,
            'ap': 120,
            'aa': 25,
            'asa': 30,
            'tp': 7.0,
            'ab': 3.5,
            'ag': 1.0
        }
        response = client.post('/check-up', json=payload)
        assert response.status_code == 400
        data = response.get_json()
        assert 'Age must be between 1 and 120' in data['message']

def test_validation_biological_limits(client):
    """Verify biological limit checks (e.g., Total Bilirubin > 100)."""
    payload = {
        'age': 45,
        'gender': 1,
        'tb': 150.0,  # Exceeds max limit 100.0
        'db': 0.2,
        'ap': 120,
        'aa': 25,
        'asa': 30,
        'tp': 7.0,
        'ab': 3.5,
        'ag': 1.0
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert 'exceeds biologically plausible limit' in data['message']

def test_gender_format_flexibility(client):
    """Verify flexibility in gender formatting (strings, ints, casing)."""
    valid_genders = ['Male', 'male', 'M', 1, '1', 'Female', 'female', 'F', 0, '0']
    base_payload = {
        'age': 45,
        'tb': 1.0,
        'db': 0.2,
        'ap': 120,
        'aa': 25,
        'asa': 30,
        'tp': 7.0,
        'ab': 3.5,
        'ag': 1.0
    }
    for g in valid_genders:
        p = dict(base_payload)
        p['gender'] = g
        resp = client.post('/check-up', json=p)
        assert resp.status_code == 200, f"Failed for gender={g}: {resp.get_json()}"

def test_threshold_bundle_consistency():
    """
    Verify threshold-bundle consistency:
    - target_recall_threshold exists and is in (0, 1)
    - sex_aware_thresholds exist for male and female
    - for each sex g, 0 < a_g < b_g < 1
    - clinical warnings are present
    """
    bundle = load_production_model()
    thresh = bundle.get('decision_thresholds', {})
    assert 'target_recall_threshold' in thresh
    assert 0.0 < thresh['target_recall_threshold'] < 1.0
    assert 'sex_aware_thresholds' in thresh
    sat = thresh['sex_aware_thresholds']
    assert 'male' in sat and 'female' in sat
    assert sat['male']['a_m'] < sat['male']['b_m']
    assert sat['female']['a_f'] < sat['female']['b_f']
    assert 'clinical_warnings' in bundle
    assert 'tradeoff' in bundle['clinical_warnings']
    assert 'female_disparity' in bundle['clinical_warnings']

def test_version_mismatch_and_purpose():
    """
    Verify research prototype notice and model bundle integrity.
    """
    bundle = load_production_model()
    assert bundle['model_version'] == '3.0.0'
    assert bundle.get('purpose') == 'RESEARCH_PROTOTYPE_DEMONSTRATION_ONLY'
    assert 'reports/10_nested_results.json' in bundle.get('metrics_source', '')

