import pytest
import os
import sys

# Ensure project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server import app, train_or_get_model

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'Liverify'
    assert data['model_loaded'] is True

def test_home_page(client):
    """Test home page loads with brand title."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Liverify" in response.data
    assert b"personal Liver Assistant" in response.data

def test_contact_page(client):
    """Test contact page loads correctly."""
    response = client.get('/contact')
    assert response.status_code == 200
    assert b"Liverify" in response.data
    assert b"Feedback" in response.data

def test_check_up_get(client):
    """Test check-up page renders the test input form."""
    response = client.get('/check-up')
    assert response.status_code == 200
    assert b"Total Bilirubin" in response.data
    assert b"Alkaline Phosphatase" in response.data

def test_check_up_post_healthy_html(client):
    """Test check-up POST with healthy normal lab values."""
    payload = {
        'age': '24',
        'gender': '0',
        'tb': '0.5',
        'db': '0.1',
        'ap': '90',
        'aa': '15',
        'asa': '15',
        'tp': '7.5',
        'ab': '4.5',
        'ag': '1.5'
    }
    response = client.post('/check-up', data=payload)
    assert response.status_code == 200
    assert b"Low Risk of Liver Disease" in response.data

def test_check_up_post_diseased_html(client):
    """Test check-up POST with abnormal diseased lab values."""
    payload = {
        'age': '62',
        'gender': '1',
        'tb': '10.9',
        'db': '5.5',
        'ap': '699',
        'aa': '64',
        'asa': '100',
        'tp': '7.5',
        'ab': '3.2',
        'ag': '0.74'
    }
    response = client.post('/check-up', data=payload)
    assert response.status_code == 200
    assert b"Potential Liver Disease Detected" in response.data

def test_check_up_post_json_api(client):
    """Test check-up POST with JSON API request."""
    payload = {
        'age': 55,
        'gender': 1,
        'tb': 8.5,
        'db': 4.0,
        'ap': 550,
        'aa': 80,
        'asa': 95,
        'tp': 6.0,
        'ab': 2.5,
        'ag': 0.7
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['result'] == 1
    assert data['risk_band'] == 'HIGH_RISK'
    assert data['probability'] > 0.75

def test_eliminated_age_under_25_bias(client):
    """
    Verify resolution of L17 flaw: Young patients under 25 with severe liver
    pathology are NO LONGER masked or suppressed by an arbitrary hardcoded rule.
    """
    payload = {
        'age': '20',
        'gender': '1',
        'tb': '10.0',
        'db': '5.0',
        'ap': '500',
        'aa': '80',
        'asa': '90',
        'tp': '6.0',
        'ab': '2.5',
        'ag': '0.7'
    }
    response = client.post('/check-up', data=payload)
    assert response.status_code == 200
    assert b"Potential Liver Disease Detected" in response.data

def test_model_training_and_caching():
    """Verify model loading returns production bundle with calibrated pipeline."""
    bundle = train_or_get_model()
    assert 'pipeline' in bundle
    assert hasattr(bundle['pipeline'], 'predict_proba')
    assert len(bundle['feature_names']) == 10
    assert bundle['model_version'] == '3.0.0'

def test_missing_fields_validation(client):
    """Test validation fails when required fields are missing."""
    incomplete_payload = {
        'age': 40,
        'gender': 1
    }
    response = client.post('/check-up', json=incomplete_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'Missing required fields' in data['message']

def test_negative_values_validation(client):
    """Test validation fails when negative lab values are submitted."""
    negative_payload = {
        'age': 40,
        'gender': 1,
        'tb': -2.5,
        'db': 0.5,
        'ap': 120,
        'aa': 25,
        'asa': 20,
        'tp': 7.0,
        'ab': 4.0,
        'ag': 1.2
    }
    response = client.post('/check-up', json=negative_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'cannot be negative' in data['message']

def test_non_numeric_values_validation(client):
    """Test validation fails when non-numeric strings are submitted."""
    invalid_payload = {
        'age': 'forty',
        'gender': '1',
        'tb': 'invalid',
        'db': '0.5',
        'ap': '120',
        'aa': 25,
        'asa': 20,
        'tp': 7.0,
        'ab': 4.0,
        'ag': 1.2
    }
    response = client.post('/check-up', json=invalid_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'valid numeric numbers' in data['message']

def test_age_out_of_range_validation(client):
    """Test validation fails when age is outside 1-120."""
    invalid_age_payload = {
        'age': 150,
        'gender': 1,
        'tb': 1.0,
        'db': 0.3,
        'ap': 120,
        'aa': 25,
        'asa': 20,
        'tp': 7.0,
        'ab': 4.0,
        'ag': 1.2
    }
    response = client.post('/check-up', json=invalid_age_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'Age must be between 1 and 120' in data['message']

def test_medical_disclaimer_present(client):
    """Test response includes explicit medical disclaimer."""
    payload = {
        'age': 45,
        'gender': 1,
        'tb': 1.0,
        'db': 0.3,
        'ap': 120,
        'aa': 25,
        'asa': 20,
        'tp': 7.0,
        'ab': 4.0,
        'ag': 1.2
    }
    response = client.post('/check-up', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'disclaimer' in data
    assert 'clinical medical diagnosis' in data['disclaimer'].lower()
