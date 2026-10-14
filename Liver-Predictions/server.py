from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
import sys
import math
import warnings
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'secret-key-goes-here')

MODEL_PATH = os.path.join(BASE_DIR, "app", "production_model.joblib")
_cached_bundle = None

def load_production_model():
    """
    Loads and caches the leak-free, calibrated production model bundle.
    Fails fast on startup if the model artifact is missing.
    """
    global _cached_bundle
    if _cached_bundle is not None:
        return _cached_bundle

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Production model bundle not found at {MODEL_PATH}. "
            f"Please run 'python src/export_production_model.py' to generate the production artifact."
        )

    _cached_bundle = joblib.load(MODEL_PATH)
    return _cached_bundle

def train_or_get_model():
    """Backward-compatible alias for existing tests and scripts."""
    return load_production_model()

def Model():
    """Backward-compatible alias returning the fitted pipeline."""
    bundle = load_production_model()
    return bundle['pipeline']

# Fail fast on import/startup
_production_bundle = load_production_model()

DISCLAIMER_TEXT = (
    "Notice: This prediction is an experimental statistical estimate based on retrospective "
    "machine learning analysis of the Indian Liver Patient Dataset (ILPD). It does NOT constitute a "
    "clinical medical diagnosis, treatment plan, or prognostic guarantee. Always consult a licensed "
    "physician or gastroenterologist for clinical interpretation and diagnostic confirmation."
)

@app.route("/health", methods=['GET'])
def health():
    bundle = load_production_model()
    return jsonify({
        "status": "healthy",
        "service": "Liverify",
        "model_version": bundle.get("model_version", "2.0.0"),
        "data_hash": bundle.get("data_hash", ""),
        "model_loaded": True
    })

@app.route("/", methods=['GET'])
def index():
    return render_template('home.html')

@app.route("/contact", methods=['GET'])
def Contact():
    return render_template('contact.html')

@app.route("/check-up", methods=['GET', 'POST'])
def Check_Up():
    if request.method == 'POST':
        bundle = load_production_model()
        pipeline = bundle['pipeline']
        feature_names = bundle['feature_names']
        thresholds = bundle.get('decision_thresholds', {
            'target_recall_threshold': 0.51,
            'low_risk_threshold': 0.35,
            'high_risk_threshold': 0.75
        })

        # Support both JSON payload and HTML form submissions
        payload = request.get_json(silent=True) or request.form

        required_fields = ['age', 'gender', 'tb', 'db', 'ap', 'aa', 'asa', 'tp', 'ab', 'ag']
        missing_fields = [
            f for f in required_fields
            if f not in payload or payload.get(f) is None or str(payload.get(f)).strip() == ""
        ]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            if request.is_json:
                return jsonify({"status": "error", "message": error_msg}), 400
            return render_template('check.html', error=error_msg, values=payload), 400

        try:
            # Parse and validate numeric types
            try:
                age_raw = payload.get('age')
                gender_raw = payload.get('gender')
                tb_raw = payload.get('tb')
                db_raw = payload.get('db')
                ap_raw = payload.get('ap')
                aa_raw = payload.get('aa')
                asa_raw = payload.get('asa')
                tp_raw = payload.get('tp')
                ab_raw = payload.get('ab')
                ag_raw = payload.get('ag')

                age = float(age_raw)
                tb = float(tb_raw)
                db = float(db_raw)
                ap = float(ap_raw)
                aa = float(aa_raw)
                asa = float(asa_raw)
                tp = float(tp_raw)
                ab = float(ab_raw)
                ag = float(ag_raw)
            except (ValueError, TypeError):
                raise ValueError("All lab values must be valid numeric numbers.")

            # Validate against NaN and Inf
            for name, val in [
                ('Age', age), ('Total Bilirubin', tb), ('Direct Bilirubin', db),
                ('Alkaline Phosphatase', ap), ('ALT', aa), ('AST', asa),
                ('Total Proteins', tp), ('Albumin', ab), ('A/G Ratio', ag)
            ]:
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"{name} cannot be NaN or Infinite.")

            # Age validation (1-120)
            if age < 1 or age > 120:
                raise ValueError("Age must be between 1 and 120.")

            # Gender parsing & validation
            gender_str = str(gender_raw).strip().lower()
            if gender_str in ('1', '1.0', 'male', 'm'):
                is_male = 1
            elif gender_str in ('0', '0.0', 'female', 'f'):
                is_male = 0
            else:
                raise ValueError("Gender must be 0 (Female), 1 (Male), 'female', or 'male'.")

            # Validate biological limits and non-negativity
            limits = [
                ('Total Bilirubin', tb, 100.0),
                ('Direct Bilirubin', db, 50.0),
                ('Alkaline Phosphatase', ap, 3000.0),
                ('ALT', aa, 3000.0),
                ('AST', asa, 5000.0),
                ('Total Proteins', tp, 20.0),
                ('Albumin', ab, 15.0),
                ('A/G Ratio', ag, 10.0),
            ]
            for field_name, val, max_val in limits:
                if val < 0:
                    raise ValueError(f"{field_name} cannot be negative.")
                if val > max_val:
                    raise ValueError(f"{field_name} ({val}) exceeds biologically plausible limit ({max_val}).")

            # Build strictly ordered feature dictionary matching model training order
            model_input_dict = {
                'Age': float(age),
                'Total_Bilirubin': tb,
                'Direct_Bilirubin': db,
                'Alkaline_Phosphotase': ap,
                'Alamine_Aminotransferase': aa,
                'Aspartate_Aminotransferase': asa,
                'Total_Protiens': tp,
                'Albumin': ab,
                'Albumin_and_Globulin_Ratio': ag,
                'is_male': int(is_male)
            }

            input_df = pd.DataFrame([model_input_dict])[feature_names]

            # Predict calibrated probability of liver disease (Class 1)
            prob = float(pipeline.predict_proba(input_df)[0, 1])

            # Apply sex-aware clinical decision triage bands from bundle
            sex_thresholds = thresholds.get('sex_aware_thresholds', {})
            if is_male == 1:
                group_cfg = sex_thresholds.get('male', {'a_m': 0.35, 'b_m': 0.70})
                low_thresh = group_cfg.get('a_m', 0.35)
                high_thresh = group_cfg.get('b_m', 0.70)
            else:
                group_cfg = sex_thresholds.get('female', {'a_f': 0.25, 'b_f': 0.65})
                low_thresh = group_cfg.get('a_f', 0.25)
                high_thresh = group_cfg.get('b_f', 0.65)

            recall_thresh = thresholds.get('target_recall_threshold', 0.50)

            if prob <= low_thresh:
                risk_band = "LOW_RISK"
                recommendation = "Low probability of liver disease indicated by biomarker panel. Continue routine preventive health checkups."
            elif prob >= high_thresh:
                risk_band = "HIGH_RISK"
                recommendation = "Elevated probability of liver disease indicated. Prompt consultation with a gastroenterologist or hepatologist is recommended."
            else:
                risk_band = "UNCERTAIN"
                recommendation = "Borderline biomarker zone (clinical referral tier). Confirmatory evaluation, follow-up liver function panel, and ultrasound are advised."

            # High-sensitivity binary prediction
            prediction = 1 if prob >= recall_thresh else 0

            # Explicit clinical trade-off and demographic warnings
            tradeoff_warning = "At high clinical sensitivity (91.2%), binary specificity is 23.0%. Borderline cases are routed to clinical referral."
            female_fnr_warning = "Clinical Caution: Female liver patients frequently present with normal median enzymes (ALT: 16 U/L, Bilirubin: 0.70 mg/dL). Normal enzymes do NOT rule out pathology."

            # Display dictionary for UI template
            display_dict = {
                'Age': int(age),
                'Gender_Male': 1 if is_male == 1 else 0,
                'Gender_Female': 1 if is_male == 0 else 0,
                'Gender': 'Male' if is_male == 1 else 'Female',
                'Total_Bilirubin': tb,
                'Direct_Bilirubin': db,
                'Alkaline_Phosphotase': ap,
                'Alamine_Aminotransferase': aa,
                'Aspartate_Aminotransferase': asa,
                'Total_Protiens': tp,
                'Albumin': ab,
                'Albumin_and_Globulin_Ratio': ag,
            }

            if request.is_json:
                return jsonify({
                    "status": "success",
                    "result": prediction,
                    "prediction": prediction,
                    "probability": round(prob, 4),
                    "risk_band": risk_band,
                    "recommendation": recommendation,
                    "tradeoff_warning": tradeoff_warning,
                    "female_fnr_warning": female_fnr_warning,
                    "model_version": bundle.get("model_version", "3.0.0"),
                    "data_hash": bundle.get("data_hash", ""),
                    "values": display_dict,
                    "disclaimer": DISCLAIMER_TEXT
                })

            return render_template(
                'check.html',
                result=prediction,
                probability=prob,
                risk_band=risk_band,
                recommendation=recommendation,
                tradeoff_warning=tradeoff_warning,
                female_fnr_warning=female_fnr_warning,
                model_version=bundle.get("model_version", "3.0.0"),
                data_hash=bundle.get("data_hash", ""),
                values=display_dict,
                disclaimer=DISCLAIMER_TEXT
            )
        except Exception as e:
            if request.is_json:
                return jsonify({"status": "error", "message": str(e)}), 400
            flash(f"Error processing form: {e}")
            return render_template('check.html', error=str(e), values=payload), 400

    return render_template('check.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1")
    print(f" * Liverify v{_production_bundle.get('model_version')} running on http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)
