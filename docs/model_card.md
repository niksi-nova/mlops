# Model Card: Delhi Bus Travel-Time Regressor (`lightgbm_l1`)

Following the standard model card reporting framework ([Mitchell et al., 2019](https://arxiv.org/abs/1810.03993)), this document details the performance characteristics, intended applications, data provenance, and operational boundaries of the production `lightgbm_l1` model.

---

## 1. Model Details

* **Model Identifier**: `lightgbm_l1`
* **Model Type**: Gradient Boosted Decision Trees (`LGBMRegressor`)
* **Objective Function**: L1 Loss (Mean Absolute Error)
* **Hyperparameters**:
  - `num_leaves`: 255
  - `learning_rate`: 0.02217
  - `n_estimators`: 3000 (with early stopping at round 100)
  - `subsample`: 0.879
  - `colsample_bytree`: 0.829
  - `reg_lambda`: 0.0265
  - `min_child_samples`: 50
* **Model File Size**: 66.80 MB (`models/lightgbm_l1.joblib`)
* **Developer Organization**: Delhi Bus Transit MLOps Team (RV College of Engineering & Nova)
* **Release Version**: `v1` (Local Pipeline) / `v2` (W&B Artifact: `delhi-lightgbm_l1:v2`)
* **Associated Collection**: `delhi-bus-delay-model:v0` (Alias: `production`)
* **Primary Metric**: Mean Absolute Error (MAE) in seconds

---

## 2. Intended Use & Target Users

### 2.1 Intended Use Cases
* **Real-Time Passenger Information Systems (PIS)**: Providing accurate, dynamic arrival and delay estimates for passengers waiting at bus stops.
* **Transit Operations & Dispatch**: Assisting municipal dispatchers in identifying transit congestion hotspots, schedule drift, and route bottlenecks in real time.
* **Headway Management**: Calculating recommended bus departure intervals based on predicted downstream segment congestion.

### 2.2 Out-of-Scope & Prohibited Applications
* **Navigation for Private Vehicles**: Bus travel times include passenger dwell times, boarding delays, and dedicated bus lane dynamics; using predictions for passenger cars will result in severe overestimation.
* **Emergency Dispatch**: Model predictions must not be used for emergency services (ambulances, police, fire response).
* **Unseen Route Generalization**: The model is trained and calibrated specifically on Delhi routes 425 and 534. It must not be deployed on uncalibrated transit corridors without prior validation and retraining.
* **Driver Performance Scoring**: The model must not be used to penalize or evaluate individual drivers, as travel times are overwhelmingly driven by traffic congestion and passenger volume rather than driving style.

---

## 3. Training & Evaluation Data

* **Dataset Identifier**: `delhi-data-v2` (Versioned via DVC on DagsHub)
* **Time Span**: 92 continuous days of public transit operations in Delhi.
* **Temporal Partitions**:
  - **Training Set**: Days 1–61 ($N = 491,856$ transitions)
  - **Validation Set**: Days 62–76 ($N = 122,964$ transitions)
  - **Test Set (Held-Out)**: Days 77–92 ($N = 153,705$ transitions)
* **Target Variable**: `travel_time_s` (Continuous duration in seconds to traverse a stop-to-stop segment).

### Feature Schema (25 Total Features)
* **Categorical (4)**: `route` (425, 534), `direction` (up, down), `segment_id`, `relative_weekday` (0–6).
* **Temporal (4)**: `hour` (0–23), `minute_of_day`, `hour_sin`, `hour_cos`.
* **Spatial & Position (4)**: `from_node`, `nodes_spanned`, `route_position`, `stops_into_trip`.
* **Traffic & Congestion Dynamics (13)**: `is_peak_am`, `is_peak_pm`, `prev_bus_tt_s`, `prev3_median_tt_s`, `headway_s`, `n_buses_recent`, `has_prev_bus`, `own_prev_segment_tt_s`, `own_prev3_median_tt_s`, `elapsed_trip_s`, `cum_distance_km`, `delay_recent_buses_s`, `delay_own_prev_segment_s`.

---

## 4. Aggregate Performance Metrics

Evaluated on the held-out Month 3 test partition ($N = 153,705$ segments):

| Metric | Candidate (`lightgbm_l1`) | Recent Buses Baseline | Typical Historical Baseline | Improvement over Best Baseline | Quality Gate Status |
|---|---|---|---|---|---|
| **Test MAE** | **30.60 s** | 38.22 s | 41.29 s | **+19.93%** | **PASS** ($\le 35.0$s) |
| **Validation MAE**| **29.33 s** | 37.73 s | 38.00 s | — | — |
| **Generalization Gap** | **4.32%** | — | — | — | **PASS** ($\le 10.0\%$) |
| **Within-60s Accuracy**| **89.35%** | 86.43% | 86.93% | +2.92% | **PASS** ($\ge 85.0\%$) |
| **Median Warm Latency** | **15.17 ms** | — | — | — | **PASS** ($\le 50.0$ ms) |

---

## 5. Slice-Based Fairness & Performance Analysis

Source: `reports/quality_gate.json`

### 5.1 Hourly Performance & Congestion Slices
* Evaluated across 18 operational daytime and evening hours (6:00 to 23:00).
* Off-peak morning hours (e.g. 06:00) demonstrate best-case MAE of **22.45 s**.
* Peak evening congestion (18:00–20:00) exhibits elevated MAE of **44.57 s**.
* Max hourly slice ratio: **1.46** (well within the $\le 1.8$ threshold).
* **Low-Volume Note**: Hour 4 recorded only 1 sample ($N=1$); it was cleanly flagged as `INSUFFICIENT_DATA` rather than skewing operational statistics.

### 5.2 Cold-Start Resilience (`has_prev_bus`)
* **Warm Segments (`has_prev_bus = 1`)**: $N = 147,917$ (96.2% of data), **MAE: 29.80 s**.
* **Cold-Start Segments (`has_prev_bus = 0`)**: $N = 5,788$ (3.8% of data), **MAE: 51.20 s**.
* Slice error ratio: **1.67** (satisfies the $\le 1.8$ quality threshold).

### 5.3 Multi-Node Segment Transitions
* Single-node standard segments (`nodes_spanned = 1`): $N = 146,862$, **MAE: 28.64 s**.
* Multi-node express/skipped transitions (`nodes_spanned > 1`): $N = 6,843$, **MAE: 72.63 s**.

### 5.4 Travel Time Distribution Tails
* Short segments (0–60s): **MAE: 10.34 s** ($N = 44,217$).
* Medium segments (60–120s): **MAE: 15.96 s** ($N = 54,882$).
* Congested segments (120–300s): **MAE: 34.80 s** ($N = 44,875$).
* Extreme congestion tail ($> 600$s): **MAE: 397.86 s** ($N = 2,474$, 1.6% of rows).
* Bound protection: All predictions are clipped to $[1.0, 1800.0]$ seconds to prevent negative or infinite values.

---

## 6. Known Limitations

1. **Route Coverage Boundary**: Calibration is restricted to routes 425 and 534 in South/Central Delhi. It cannot be applied to Northern or Eastern peripheral routes without retraining.
2. **Missing Spatial Coordinates**: The dataset lacks raw GPS latitude/longitude. Consequently, spatial interactions with cross-traffic, traffic signals, and road elevation cannot be explicitly engineered.
3. **Calendar Date Blindness**: The dataset spans day indices (1–92) without Gregorian dates. National holidays (Republic Day, Diwali, Eid), extreme rain events, or school closures cannot be explicitly represented.
4. **Data Drift Sensitivity**: Analysis of daily MAE over the 3-month window indicates daily performance variance between 23.6s and 38.6s, demonstrating sensitivity to changing seasonal traffic patterns.

---

## 7. Ethical & Privacy Considerations

* **Privacy Preservation**: No personal commuter data, ticketing histories, or facial images are collected or processed.
* **Corridor Equity**: Performance across route 425 (up MAE 43.66s) and route 534 (up MAE 24.55s) reflects underlying road infrastructure variance; monitoring ensures accuracy improvements on one route do not degrade service estimation on the other.
