# Project Overview: AI-Assisted Public Transport Information System (Transit Delay Prediction)

## Problem
When a TTC bus disruption is reported, predict how many minutes the delay
will last, so a passenger information system can show an expected delay
instead of only "delayed". Supervised regression; target `min_delay` (minutes);
primary metric MAE.

## Objectives
**Main objective:** Develop an AI-assisted public transport information system
that predicts the delay duration of transit disruptions from historical delay,
schedule, weather and calendar data, giving passengers reliable delay estimates
and helping operators identify recurring delay patterns, supported by an
end-to-end MLOps pipeline that keeps the model reproducible, deployable and
reliable over time.

**Specific objectives:**
1. Build and evaluate the delay prediction model: acquire, clean and enrich
   multi-year TTC delay data with weather and holidays; engineer features;
   train models from a naive baseline to gradient boosting, evaluated with MAE,
   RMSE and R² on a time-based split; target at least 15% MAE improvement over
   the route x incident baseline. (Phase 1)
2. Ensure reproducibility and automate deployment: DVC data versioning and
   pipeline, MLflow tracking, PyTest validation, MLflow Model Registry,
   Dockerized FastAPI service, GitHub Actions CI/CD with rollback. (Phase 2)
3. Monitor, adapt and govern the model: Evidently drift monitoring,
   Prometheus/Grafana dashboard, a deliberate drift scenario triggering
   retraining and redeployment, and a model card. (Phase 3)

## Risk identification
| ID | Risk | Type | Description |
|---|---|---|---|
| R1 | Data quality and label noise | Intrinsic | Manually logged delays contain missing values, duplicates, inconsistent names and impossible values |
| R2 | Data leakage and non-reproducibility | Intrinsic | Post-incident features (min_gap) and random splits inflate accuracy; unversioned work cannot be reproduced |
| R3 | Data and concept drift | Extrinsic | Travel patterns, routes, weather and ridership change, so a trained model degrades over time |
| R4 | Upstream source changes and pipeline/serving failure | Extrinsic | The open data portal or weather API changes format or goes offline, breaking ingestion or serving |

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

## Risk mitigation using MLOps
| ID | Mitigation | Tools | Evidence in this repo |
|---|---|---|---|
| R1 | Automated data validation stage that stops the pipeline on bad data | Pandera inside the DVC pipeline | src/data/validate.py, reports/validation_report.json |
| R2 | Versioned, reproducible pipeline with a fixed time-based split | DVC + MLflow | dvc.yaml, params.yaml, data-v1/data-v2 tags, leakage tests |
| R3 | Drift monitoring with automated retraining | Evidently + GitHub Actions | Planned for Phase 3; drift already measured (val vs test gap) |
| R4 | Automated rollback to the last validated version | MLflow Model Registry + CI/CD | Planned for Phase 2; schema differences already handled in clean.py |

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

## Results (Phase 1)

Six models were trained and compared (`reports/train_metrics.json` for
validation, `reports/test_metrics.json` for the held-out 2024 test split):

| Model | Validation MAE | Test MAE |
|---|---|---|
| naive_baseline (route x incident) | 8.114 | 8.903 |
| global_median | 9.773 | 10.413 |
| **lightgbm_l1 (best model)** | **7.408** | **8.302** |

**Best model:** `lightgbm_l1` (LightGBM, L1/MAE objective, raw target).

**Success target:** ≥15% MAE improvement over the route x incident naive
baseline, measured on test. **Actual: 6.75% improvement** (naive baseline
test MAE 8.903 → best model test MAE 8.302) — **target not met**
(`target_met: false` in `reports/test_metrics.json`).

Against the simpler `global_median` baseline, the improvement is larger:
**20.27%** (10.413 → 8.302).

**Validation-to-test gap:** validation MAE 7.408 → test MAE 8.302, a gap of
**0.893 minutes**. This is a real but moderate generalisation gap, consistent
with mild drift in the target distribution between 2023 (validation) and
2024 (test) rather than a sign of overfitting.

**Typical-case accuracy:** median absolute error of the best model on test
is **3.003 minutes** — for most incidents, the predicted delay is very
close to the actual one, even though the 15% aggregate-MAE improvement
target was not reached (full error breakdown by delay bucket, incident
type, hour and location is in `docs/model_evaluation.md`).
