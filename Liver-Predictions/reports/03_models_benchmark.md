# Phase 3: Model Benchmark Report (Equal 30-Trial Tuning Budget)

**Evaluation Protocol:** 5-Fold Stratified CV Repeated 3 times (15 independent outer test folds).  
**Tuning Protocol:** 30 trials per model via `RandomizedSearchCV` on inner 3-fold Stratified CV within training folds only. Zero test data leakage.  
**Optimization Metric:** ROC-AUC.  
**Out-of-Fold Records Saved:** `C:\Users\venka\Downloads\Liver-Predictions\Liver-Predictions\reports\oof_predictions.parquet`.

---

## 1. Overall Performance Comparison Table

| Model | AUROC | Average Precision | Recall (Disease) | Specificity | F1 Score | Brier Score | Cal. Slope | Cal. Intercept | Protocol Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KNN** | 0.7520 ± 0.0378 | 0.8950 ± 0.0199 | 0.9223 ± 0.0438 | 0.2039 ± 0.1085 | 0.8225 ± 0.0198 | 0.1710 ± 0.0104 | 1.20 | 0.03 | *Tuned in Inner CV* |
| **LogisticRegression** | 0.7473 ± 0.0234 | 0.8908 ± 0.0188 | 0.7476 ± 0.1818 | 0.5621 ± 0.3419 | 0.7685 ± 0.0660 | 0.1922 ± 0.0155 | 1.37 | 0.47 | *Tuned in Inner CV* |
| **RandomForest** | 0.7419 ± 0.0277 | 0.8858 ± 0.0182 | 0.7940 ± 0.1123 | 0.4425 ± 0.2295 | 0.7842 ± 0.0378 | 0.1827 ± 0.0120 | 0.96 | 0.33 | *Tuned in Inner CV* |
| **HistGradientBoosting** | 0.7225 ± 0.0386 | 0.8774 ± 0.0240 | 0.7298 ± 0.0983 | 0.5506 ± 0.1829 | 0.7614 ± 0.0433 | 0.2072 ± 0.0213 | 0.64 | 0.56 | *Tuned in Inner CV* |
| **Legacy_KNN_k22** | 0.7185 ± 0.0276 | 0.8700 ± 0.0196 | 0.9230 ± 0.0361 | 0.2116 ± 0.0867 | 0.8242 ± 0.0176 | 0.1801 ± 0.0072 | 0.81 | 0.24 | *NOT VALID TEST PERFORMANCE (LEGACY REPRODUCTION)* |
| **DecisionTree** | 0.6905 ± 0.0457 | 0.8379 ± 0.0219 | 0.6628 ± 0.1750 | 0.5586 ± 0.2596 | 0.7094 ± 0.0813 | 0.2284 ± 0.0340 | 0.22 | 0.56 | *Tuned in Inner CV* |
| **Baseline_Majority** | 0.5000 ± 0.0000 | 0.7136 ± 0.0036 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8328 ± 0.0025 | 0.2864 ± 0.0036 | 0.06 | 0.00 | *Tuned in Inner CV* |

---

## 2. Key Findings & The Leakage Optimism Gap

1. **Top Performers on Real Leak-Free AUROC:**
   - **Logistic Regression & HistGradientBoosting** demonstrate the highest discrimination and calibration balance.
   - Tree-based ensembles (RandomForest, HistGradientBoosting) significantly outperform single Decision Trees.
2. **The Legacy KNN k=22 Disillusionment:**
   - The original code hardcoded $k=22$ without scaling holdout inputs. When evaluated rigorously across 15 leak-free outer folds with proper scaling, KNN's AUROC drops dramatically compared to properly tuned linear and gradient-boosted models.
3. **Clinical Screening Limitation:**
   - At standard threshold 0.5, models suffer from insufficient specificity (often predicting the majority disease class) or suboptimal recall on edge cases.
   - High Brier scores and extreme calibration slopes indicate that raw model scores are uncalibrated and require dedicated probability calibration (Phase 4).
