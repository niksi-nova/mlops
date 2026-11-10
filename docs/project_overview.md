# Project Overview: AI-Assisted Public Transport Information System (Delhi Bus Travel-Time Prediction)

## Problem

Predict how long a Delhi bus will take to travel from one stop to the next
(`travel_time_s`, in seconds), and from that, a "delay vs typical" signal -
`delay_s = travel_time_s - (historical median travel time for that segment
and hour)` - so a passenger information system can show a realistic
stop-to-stop ETA and flag when a bus is running slower than usual, instead
of relying on a fixed timetable. Supervised regression; primary metric MAE
(seconds).

## Objectives

**Main objective:** Build an AI-assisted public transport information
system that predicts Delhi bus stop-to-stop travel time from real-time GPS
node-crossing data and recent-bus/historical-typical signals, giving
passengers realistic ETAs and a "running slower than usual" delay signal,
supported by an end-to-end MLOps pipeline that keeps the model
reproducible, trackable and reliable over time.

**Specific objectives:**
1. **(Phase 1)** Build and evaluate the travel-time model: acquire, clean,
   segment and validate the Delhi GPS data; engineer real-time
   (recent-bus) and historical (typical-segment) features with no leakage;
   train models from simple baselines to tuned gradient boosting,
   evaluated with MAE/RMSE/R² on a day-based split; manage the pipeline
   with DVC and track every run with Weights & Biases. Success criterion:
   **≥10% lower test MAE than the best non-ML baseline.**
2. **(Phase 2)** Ensure reproducibility and automate deployment: formal
   test coverage, a W&B Model Registry entry for the production model, a
   Dockerized FastAPI serving endpoint, and GitHub Actions CI/CD with
   rollback.
3. **(Phase 3)** Monitor, adapt and govern the model: Evidently drift
   monitoring, a live dashboard, a deliberate drift scenario triggering
   retraining and redeployment, and a model card.

## Risk identification

| ID | Risk | Type | Description |
|---|---|---|---|
| R1 | Data quality | Intrinsic | Skipped stop readings, node/time going backward, and midnight-crossing trips in the raw GPS data can silently corrupt segments |
| R2 | Data leakage / non-reproducibility | Intrinsic | The previous-bus feature can leak if a bus still mid-segment is used as "previous"; unversioned work cannot be reproduced |
| R3 | Data and concept drift | Extrinsic | Travel time rises month to month in this data; a model trained once on early months degrades as conditions shift |
| R4 | Upstream source / pipeline failure | Extrinsic | A single static, unlicensed research dataset, a live feed that would need an access key, and third-party tool risk (e.g. a tracking provider being discontinued) can all break the pipeline |

Scoring: Risk score = Likelihood (1-5) x Impact (1-5).
Levels: 1-4 Low, 5-9 Medium, 10-16 High, 20-25 Critical.

## Risk assessment before MLOps

| ID | Likelihood | Impact | Score | Level |
|---|---|---|---|---|
| R1 | 4 | 4 | 16 | High |
| R2 | 4 | 5 | 20 | Critical |
| R3 | 5 | 4 | 20 | Critical |
| R4 | 3 | 5 | 15 | High |
| Total | | | 71 | |

## Risk mitigation using MLOps (Delhi evidence)

| ID | Mitigation | Tools | Evidence in this repo |
|---|---|---|---|
| R1 | Pandera validation stage that stops the pipeline on bad data | Pandera inside the DVC pipeline | `src/data/validate.py`; 80% of trips skip at least one node reading and the raw data has midnight-crossing trips (`docs/data_inventory.md`, `reports/cleaning_report.json` - 38 trips midnight-corrected, 11 dropped for time going backward); this exact validation stage caught a real zero-length-segment bug (`nodes_spanned <= 0`) during development, before it reached training |
| R2 | Strict arrival-time rule for the previous-bus feature, leakage-guard unit tests, out-of-fold historical features, a fully versioned DVC pipeline, and W&B run/artifact lineage | DVC + Weights & Biases + PyTest | `src/features/build_features.py` (`compute_recent_bus_features`), `tests/test_features.py` (`test_prev_bus_ignores_earlier_bus_still_in_flight`), `dvc.yaml`, `params.yaml`, W&B lineage chain `delhi-feature-spec` → model → `final-evaluation-test` |
| R3 | Drift measured explicitly and monitoring planned | Evidently (Phase 3) + current drift evidence | Mean `travel_time_s` rose month to month (116.99s → 118.18s → 121.70s overall; evening peak hours 17-19 rose 137.88s → 144.22s → 153.42s each month); the static `typical_baseline` degraded **8.67%** from validation MAE (37.996s) to test MAE (41.290s) - exactly the kind of drift Phase 3 monitoring targets |
| R4 | DVC-versioned copy of the data on a remote (DagsHub), schema validation at ingest, and a model registry with rollback | DVC + DagsHub + Pandera + W&B Model Registry (Phase 2) | `delhi-data-v1`/`delhi-data-v2` tags and DVC remote push; the source dataset has no stated licence and is a static research snapshot (no live-feed access key); any future external tracking tool (the project already migrated once, from MLflow to W&B, after evaluating tooling risk) is swapped behind the same `src/models/tracking.py` interface |

## Risk assessment after MLOps (estimated residual risk after controls)

| ID | Likelihood | Impact | Score | Level |
|---|---|---|---|---|
| R1 | 2 | 3 | 6 | Medium |
| R2 | 1 | 4 | 4 | Low |
| R3 | 4 | 2 | 8 | Medium |
| R4 | 3 | 2 | 6 | Medium |
| Total | | | 24 | (-66%) |

Intrinsic risks (R1, R2) fall mainly in likelihood: MLOps prevents them.
Extrinsic risks (R3, R4) fall mainly in impact: MLOps contains them.
**MLOps prevents what we control and contains what we don't.**

## Tools

| Tool | Role |
|---|---|
| Git / GitHub | Version control, code history, collaboration |
| uv | Python dependency and environment management |
| DVC + DagsHub | Data/pipeline versioning (`delhi-data-v1`/`v2`), remote storage, pipeline DAG (`dvc.yaml`) |
| Pandera | Schema/data validation gate inside the pipeline (`src/data/validate.py`) |
| Weights & Biases | Experiment tracking, model/dataset artifacts and lineage, Bayesian hyperparameter sweeps |
| LightGBM / scikit-learn | Model training (gradient boosting, ridge, random forest, baselines) |
| PyTest | Unit tests, incl. leakage guards for the recent-bus features |
| Jupyter | Exploratory data analysis (`notebooks/01_eda.ipynb`) |

**Why W&B replaced MLflow:** the project originally tracked experiments
with MLflow; it was replaced with Weights & Biases for this phase to get
native Bayesian hyperparameter sweeps, artifact lineage (dataset → model →
evaluation, visible directly in the UI), and hosted tracking without
self-managing a tracking server.

## Results (Phase 1)

Seven models were trained and compared
(`reports/train_metrics.json` for validation, `reports/test_metrics.json`
for the held-out day_index 77-92 test split):

| Model | Validation MAE (s) | Test MAE (s) |
|---|---:|---:|
| global_median | 68.56 | 71.13 |
| typical_baseline | 38.00 | 41.29 |
| recent_buses_baseline | 37.73 | 38.22 |
| ridge | 37.51 | 49.67 |
| random_forest | 30.30 | 31.78 |
| lightgbm_l2 | 29.84 | 31.30 |
| **lightgbm_l1 (best model)** | **29.33** | **30.60** |

**Best model:** `lightgbm_l1` (LightGBM, L1/MAE objective, raw target,
hyperparameters tuned via a 20-run W&B Bayesian sweep).

**Success target:** ≥10% MAE improvement over the best non-ML baseline
(`recent_buses_baseline`), measured on test. **Actual: 19.93% improvement**
(recent_buses_baseline test MAE 38.22s → best model test MAE 30.60s) -
**target met** (`target_met: true` in `reports/test_metrics.json`).

Against the simpler `global_median` baseline, the improvement is larger:
**56.98%** (71.13s → 30.60s).

Full error breakdown by hour, route, segment, travel-time bucket and the
delay view is in `docs/model_evaluation.md`.
