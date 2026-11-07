# Delhi Bus Travel-Time Delay Prediction — Model Risk & Governance Report

**Project**: Delhi Bus Travel-Time Prediction (Target: `travel_time_s`, Unit: seconds)
**Production Candidate**: `lightgbm_l1` (Tuned LightGBM with L1 loss, 255 leaves, learning rate 0.02217)
**Evaluation Split**: Held-out Month 3 (Days 77–92; $N = 153,705$ segment transitions)
**Repository Branch**: `phase2-testing-registry`
**Governance Standard**: Antigravity MLOps Phase 2 Automated Quality Gate & Model Registry

---

## Executive Summary

The `lightgbm_l1` model demonstrates strong aggregate predictive accuracy for stop-to-stop bus segment travel times in Delhi, achieving a **test MAE of 30.60 seconds** and beating the strongest real-time baseline (`recent_buses_baseline`, test MAE 38.22s) by **19.93%**—comfortably exceeding the project's 10.0% improvement target. Furthermore, **89.35%** of predictions on unseen test segments are accurate to within 60 seconds of actual travel time, with a small generalization gap of **4.32%** between validation (29.33s) and test.

However, production deployment introduces operational risks spanning real-time feature availability, extreme traffic congestion tails, route representation imbalances, and platform infrastructure failure modes. This document evaluates eight principal risk categories, grounding each in empirical project evidence and linking them directly to automated safeguards in `src/models/quality_gate.py` and `src/models/registry.py`.

---

## Risk Matrix Overview

| Risk ID | Category | Severity | Primary Project Evidence | Automated Quality Gate Link |
|---|---|---|---|---|
| **R-01** | Data Drift | High | 31-day test window exhibits daily MAE fluctuations from 24.1s to 38.6s | `max_test_mae <= 35.0s`, `within_60s_pct >= 85.0%` |
| **R-02** | Target Leakage | Critical | `own_prev_segment_tt_s` strictly precedes segment entry timestamp | Feature contract in `data/processed/feature_spec.json` |
| **R-03** | Overfitting | Medium | Val MAE 29.33s vs Test MAE 30.60s (Generalization gap: 4.32%) | `generalisation_gap_pct <= 10.0%` |
| **R-04** | Route & Hour Bias | High | Route 425 up MAE 43.66s vs Route 534 up MAE 24.55s; evening peak MAE 44.57s | `max_slice_mae_ratio <= 1.8`, `min_slice_samples >= 100` |
| **R-05** | Cold-Start Degradation | High | `has_prev_bus=0` MAE 51.20s vs `has_prev_bus=1` MAE 29.80s (+71.8% error) | `has_prev_bus` slice checks in quality gate |
| **R-06** | Long Travel-Time Tail | High | Segments $\ge 600$s (1.6% of test rows) exhibit MAE of 397.9s | `prediction_sanity` bounds $[1, 1800]$s, L1 objective |
| **R-07** | Artifact Integrity | Critical | Feature realignment overhead (49ms vs 16ms); wrong column order causes garbage outputs | Exact booster ordering `categoricals + numerics`, size $\le 100$MB |
| **R-08** | Registry / Deployment | Critical | W&B daemon socket hangs under sandbox; unauthorized production promotion | Candidate/Staging/Production separation, local fallback |

---

## Detailed Risk Assessments & Safeguards

### 1. Data Drift (R-01)
* **Description**: Shifts in underlying traffic dynamics, weather conditions, road closures, schedule overhauls, or fleet alterations over time.
* **Project Evidence**:
  - Across the 31-day evaluation period (days 62–92), daily MAE ranged from ~24s to ~39s (`docs/model_evaluation.md`). The first 8 test days averaged 31.60s while the second 8 averaged 29.81s, indicating no monotonic degradation over 30 days, but notable day-to-day variance.
  - The dataset covers only 3 months and lacks true calendar dates (e.g. holidays, festivals, monsoon rain events are invisible).
* **Severity**: **High**
* **Detection Method**: Online rolling MAE monitoring per route and daily Kolmogorov-Smirnov drift tests on continuous features (`elapsed_trip_s`, `headway_s`).
* **Mitigation**: Automated retraining triggers when 7-day rolling MAE degrades beyond 35.0s.
* **Responsible Monitoring Action**: Daily scheduled evaluation cron job logging drift metrics to W&B.

---

### 2. Data Leakage (R-02)
* **Description**: Features incorporating information from the future relative to the prediction timestamp, or targets leaking into predictors.
* **Project Evidence**:
  - The delay target is $y = \text{travel\_time\_s} = t_{\text{exit}} - t_{\text{entry}}$.
  - The feature `own_prev_segment_tt_s` captures the *same* bus on its *immediately preceding* segment ($t_{\text{entry}} - t_{\text{prev\_entry}}$), which finishes before or exactly at segment entry.
  - Target encoding (`te_vehicle`) uses 5-fold out-of-fold cross-validation during training and maps unseen vehicles to the global prior, preventing vehicle label leakage.
* **Severity**: **Critical**
* **Detection Method**: Strict validation against `data/processed/feature_spec.json`; unit tests asserting non-overlap between `target` and `features`.
* **Mitigation**: Pipeline featurization enforces causality checks: no timestamp $\ge t_{\text{entry}}$ can enter feature calculations.
* **Responsible Monitoring Action**: Pre-commit automated feature contract schema tests (`tests/test_model.py:test_feature_schema_specification`).

---

### 3. Overfitting & Generalization Failure (R-03)
* **Description**: The model memorizes training noise or segment-specific idiosyncrasies that fail to generalize to future weeks.
* **Project Evidence**:
  - Validation MAE (Month 2.5, days 62–76): **29.33s**.
  - Held-out Test MAE (Month 3, days 77–92): **30.60s**.
  - Empirical Generalization Gap: **4.32%**, well within the 10.0% ceiling.
  - Hyperparameter tuning: Bayesian sweep (`7rcgds9b`, 20 trials) refined validation MAE by only ~0.38% (29.43s to 29.32s), confirming that tree depth and leaf counts (`num_leaves=255`, `learning_rate=0.02217`) are regularized by L1 loss (`objective="l1"`, `reg_lambda=0.0265`).
* **Severity**: **Medium**
* **Detection Method**: Automated generalization gap ratio: $(|\text{Test MAE} - \text{Val MAE}| / \text{Val MAE}) \times 100$.
* **Mitigation**: Quality gate check `generalisation_gap <= 10.0%` halts promotion if test error diverges from validation.
* **Responsible Monitoring Action**: Quality Gate execution prior to model registry tagging.

---

### 4. Route and Time Slice Bias (R-04)
* **Description**: Significant localized performance degradation across specific routes, directions, or times of day, masked by strong global averages.
* **Project Evidence**:
  - **Route Disparity**: Route 534 (up: 24.55s, down: 31.13s) performs far better than Route 425 (up: 43.66s, down: 39.58s). Route 425 has fewer observations (14k vs 61k) and traverses more congested central corridors.
  - **Time-of-Day Disparity**: Evening rush hours (18:00–19:00) suffer MAE of 43.08s and 44.57s (ratio ~1.45 to overall MAE).
  - **Sample Sparsity**: Hour 4 has only $N = 1$ observation in the test split (MAE 73.72s).
* **Severity**: **High**
* **Detection Method**: Automated slice evaluation measuring slice MAE ratio to overall MAE, with low-sample filtering (`min_slice_samples = 100`).
* **Mitigation**: Slices with $N < 100$ are flagged `INSUFFICIENT_DATA` rather than passing or failing the gate. Real slices must maintain a ratio $\le 1.8$ to overall MAE.
* **Responsible Monitoring Action**: Quality gate slice checks (`hourly_slices` and `route_direction_slices` in `reports/quality_gate.json`).

---

### 5. Cold-Start Operational Failure (R-05)
* **Description**: Severe degradation when a bus begins its trip or enters a segment without preceding buses in the recent window.
* **Project Evidence**:
  - Rows with `has_prev_bus = 0` (5,788 test rows, 3.8% of volume) have an MAE of **51.20s**—71.8% higher than warm rows (`has_prev_bus = 1`, MAE **29.80s**).
  - When real-time bus telemetry drops, the model falls back to static historical features (`typical_segment_hour_s`, `typical_segment_s`).
* **Severity**: **High**
* **Detection Method**: Dedicated slice check in `src/models/quality_gate.py` asserting that cold-start MAE ratio does not exceed 1.8 of overall MAE (actual: $51.20 / 30.60 = 1.67$).
* **Mitigation**: Unit tests in `tests/test_model.py:test_cold_start_rows_prediction` verify that inference on cold-start segments never throws `NaN`, `Inf`, or crashes. Downstream dispatch systems are alerted with a confidence flag when `has_prev_bus == 0`.
* **Responsible Monitoring Action**: Serving-time logging of prediction confidence based on `has_prev_bus`.

---

### 6. Long Travel-Time / Extreme Outlier Tail (R-06)
* **Description**: Travel-time distributions are heavy-tailed; extreme traffic congestion or breakdown segments cause massive absolute errors.
* **Project Evidence**:
  - Segments taking $0\text{--}60\text{s}$ have MAE **10.3s**; segments taking $60\text{--}120\text{s}$ have MAE **16.0s**; segments taking $\ge 600\text{s}$ (1.6% of test records, 2,474 rows) have an MAE of **397.9s**.
  - Ridge regression failed catastrophically on test R2 (-18.64) due to exponentiation blow-up on long segments, whereas `lightgbm_l1` directly minimizes absolute error without variance explosions.
* **Severity**: **High**
* **Detection Method**: Prediction sanity bounds checking: all outputs must reside within $[1, 1800]$ seconds (`params.yaml: segments.min_segment_seconds` and `max_segment_seconds`).
* **Mitigation**: `clip_and_backtransform` in `src/models/evaluate.py` clamps outputs to physical bounds ($1\text{s} \le \hat{y} \le 1800\text{s}$). L1 objective provides robust linear penalties against extreme outliers.
* **Responsible Monitoring Action**: Real-time alerts when consecutive predictions hit the 1,800-second ceiling.

---

### 7. Model Artifact & Schema Integrity (R-07)
* **Description**: Pickle corruption, feature ordering mismatches, or schema drift causing garbage predictions or high inference latency in production.
* **Project Evidence**:
  - LightGBM was trained with feature order `categorical_features + numeric_features`.
  - When fed features in alphabetical or arbitrary order, LightGBM is forced to re-index and cast Pandas categories at runtime, tripling inference latency from **16.25 ms to 49.9 ms**.
* **Severity**: **Critical**
* **Detection Method**: Automated booster schema assertion:
  ```python
  assert lightgbm_model.booster_.feature_name() == (
      feature_spec["categorical_features"] + feature_spec["numeric_features"]
  )
  ```
  and artifact file size ceiling: $\text{size} \le 100.0\text{ MB}$ (actual: 66.8 MB).
* **Mitigation**: `load_model("production")` in `src/models/registry.py` bundles the exact feature specification and guarantees the correct feature ordering prior to inference.
* **Responsible Monitoring Action**: Automated model contract tests in `tests/test_model.py`.

---

### 8. Registry & Deployment Failure (R-08)
* **Description**: Unauthorized model deployment, missing cloud registry connection, or deployment of an unvalidated model artifact.
* **Project Evidence**:
  - Sandboxed execution environments block local IPC sockets, causing `wandb-core` to hang during `wandb.init()` unless offline/disabled fallbacks are present.
  - Automatically deploying a candidate model directly to production risks promoting broken weights.
* **Severity**: **Critical**
* **Detection Method**: Strict alias lifecycle separation:
  - `candidate`: Newly trained model.
  - `staging`: Promoted ONLY if `quality_gate.json` has `passed: true`.
  - `production`: Promoted ONLY via explicit human approval (`promote_to_production()`).
* **Mitigation**: `src/models/registry.py` enforces local JSON fallback (`reports/model_registry.json`) when W&B is offline, ensuring zero dependency on external network availability while strictly preventing automatic promotion to production.
* **Responsible Monitoring Action**: Audit trail of `approved_by` and timestamp logged in `reports/model_registry.json`.

---

## Quality Gate Verification Checklist

| Quality Gate Check | Threshold (params.yaml) | Empirical Value | Gate Status |
|---|---|---|---|
| Improvement over Typical Baseline | $\ge 10.0\%$ | **25.88%** | **PASS** |
| Improvement over Recent Buses Baseline | $\ge 10.0\%$ | **19.93%** | **PASS** |
| Maximum Test MAE | $\le 35.0\text{ s}$ | **30.60 s** | **PASS** |
| Generalisation Gap | $\le 10.0\%$ | **4.32%** | **PASS** |
| Within-60s Accuracy | $\ge 85.0\%$ | **89.35%** | **PASS** |
| Model Artifact File Size | $\le 100.0\text{ MB}$ | **66.80 MB** | **PASS** |
| Single-Row Warm Latency (Median) | $\le 50.0\text{ ms}$ | **15.17 ms** | **PASS** |
| Prediction Sanity | Finite, positive, $[1, 1800]$s | All 200 samples valid | **PASS** |
| Slices: Hourly Performance | Ratio $\le 1.8$, Min $N \ge 100$ | 18 hours evaluated, max ratio 1.46 | **PASS** |
| Slices: Low-Volume Slices | Mark $N < 100$ as INSUFFICIENT_DATA | Hour 4 ($N=1$) flagged cleanly | **PASS** |
| Slices: Cold-Start (`has_prev_bus=0`) | Ratio $\le 1.8$ | Ratio 1.67 | **PASS** |

**Final Recommendation**: Model `lightgbm_l1` satisfies all empirical quality thresholds and is approved for promotion to **`staging`**. Deployment to **`production`** remains gated behind explicit human review.
