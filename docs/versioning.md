# Multi-Layer Versioning and Provenance Architecture

This document defines the versioning strategy and provenance tracing mechanisms implemented across the Delhi Bus Travel-Time Prediction MLOps platform. It is structured for technical architecture review and viva examination.

---

## 1. Why Versioning is Required in Production MLOps

Machine Learning systems in production fail in fundamentally different ways than traditional software. Code may remain unchanged while data distributions drift, model weights degrade, or configuration files shift. The Delhi Bus delay prediction platform establishes strict multi-layer versioning to guarantee operational reliability, compliance, and auditability.

### 1.1 Reproducibility
* **Challenge**: Re-running a training script months later frequently produces different weights or degraded metrics due to unrecorded data updates, altered hyperparameters, or non-deterministic dependencies.
* **Evidence in Repo**:
  - Every training and evaluation run records full provenance metadata in `wandb.config` and JSON reports:
    - **Exact Git commit hash** (`git_commit` in `reports/quality_gate.json` and `reports/model_registry.json`).
    - **Dataset version** (`dataset_version: "delhi-data-v2"`).
    - **DVC featurization MD5 hash** (`featurize_outputs_md5` in `src/models/tracking.py`).
    - **Hyperparameter snapshot** (`params.yaml`).
  - Running training from commit `946827e` with dataset `delhi-data-v2` and `random_state: 42` reproduces the exact LightGBM test MAE of 30.60 seconds.

### 1.2 Traceability and Audit
* **Challenge**: When an erroneous prediction impacts live users or transit dispatchers, engineers must identify the exact training run, code commit, dataset slice, and validation metrics that generated the deployed model.
* **Evidence in Repo**:
  - W&B Model Registry lineage graph links artifact `delhi-lightgbm_l1:v2` to the exact run `train-lightgbm_l1`, which in turn references input dataset partition hashes (`dvc.lock`).
  - Immutable content digests (e.g. `d7b1a20f9e4c3b8a1e5d2c7f4a9b0e3d`) guarantee model bytes have not been tampered with or corrupted on disk.

### 1.3 Safe Rollback Without Redeployment
* **Challenge**: Reverting a misbehaving model by rolling back git commits or triggering full Docker image builds takes 10–25 minutes, creating downtime or prolonged exposure to faulty predictions.
* **Evidence in Repo**:
  - The live FastAPI service (`src/api/main.py`) and Render cloud deployment load the model dynamically by alias (`MODEL_ALIAS="production"`).
  - Rolling back to a previous model version requires solely re-pointing the `production` alias in the W&B Model Registry collection `delhi-bus-delay-model`. The web service retrieves the updated bytes upon restart or polling without modifying or rebuilding container images (`scripts/demo_rollback.py`).

### 1.4 Controlled Promotion
* **Challenge**: Automatically pushing freshly trained models to production risks releasing overfitted or unstable models during traffic anomalies.
* **Evidence in Repo**:
  - Three distinct lifecycle stages:
    1. **`candidate`**: Generated upon training completion.
    2. **`staging`**: Promoted automatically **only** if all 8 automated checks in `src/models/quality_gate.py` pass (`passed: true`).
    3. **`production`**: Strictly blocked from automated promotion. Requires an interactive terminal session where a human operator explicitly enters the confirmation phrase: `PROMOTE lightgbm_l1 v1 TO PRODUCTION`.
  - Immutable approval history recorded in `reports/production_approvals.json` and `reports/model_registry.json`.

### 1.5 Cross-Version Comparison
* **Challenge**: Teams cannot assess real-world improvements without standardized benchmarking against historical baselines and prior model architectures.
* **Evidence in Repo**:
  - Every pipeline execution benchmarks the candidate model against two deterministic baselines (`typical_baseline` and `recent_buses_baseline`) and previous model generations (`ridge`, `random_forest`, `lightgbm_l2`).
  - Metrics across all 7 candidates are archived in `reports/train_metrics.json` and `reports/test_metrics.json`.

### 1.6 Environment Consistency
* **Challenge**: Discrepancies between local development laptops, CI test runners, and production cloud containers cause "works on my machine" failures.
* **Evidence in Repo**:
  - Container images on GitHub Packages (GHCR) are immutably tagged with both `latest` and `sha-<short_sha>` (e.g. `ghcr.io/niksi-nova/mlops:sha-946827e`).
  - The `/model-info` endpoint returns runtime git commit, active alias, model version, and exact 25-feature ordering, allowing monitoring probes to verify container configuration instantly.

---

## 2. The Five-Layer Provenance Model

The Delhi Bus Delay platform decouples the system into five independently versioned layers:

| Layer | What is Versioned | Tool / Mechanism | Artifact / Identifier | Evidence in Repository |
|---|---|---|---|---|
| **1. Code** | Inference API, data pipelines, model code, test suites | Git / GitHub | Commit SHA (`946827e`), branch `main` | Git log, `.github/workflows/ci.yml` |
| **2. Data** | Raw GPS logs, interim bus trips, processed Parquet splits | DVC on DagsHub | `delhi-data-v2`, MD5 `4e8867e7f5d6...` | `dvc.yaml`, `dvc.lock`, `data/processed/` |
| **3. Model** | Serialized LightGBM weights, booster trees, feature spec | W&B Model Registry | Artifact `delhi-lightgbm_l1:v2`, Collection `delhi-bus-delay-model:v0` | `reports/model_registry.json`, `models/` |
| **4. Image** | Containerized runtime environment, OS packages, pinned wheels | Docker / GHCR | `ghcr.io/niksi-nova/mlops:sha-946827e` | `Dockerfile`, `constraints.txt`, `cd.yml` |
| **5. Config** | Model hyperparameters, quality gate thresholds, route specs | YAML in Git | `params.yaml`, MD5 `1bb1eb5c08...` | `params.yaml`, `reports/quality_gate.json` |

---

## 3. End-to-End Prediction Provenance Walkthrough

To trace any single prediction made by the live production system back to its origin:

```
[Client Request: POST /predict]
       │
       ▼
[Docker Container: ghcr.io/niksi-nova/mlops:sha-946827e]
       │  (Built by GitHub Actions CD from commit 946827e)
       │
       ▼
[FastAPI Service: GET /model-info]
       │  (Reports git_commit: 946827e, alias: production)
       │
       ▼
[W&B Registry Collection: anikaub20-rv-college/delhi-bus-delay/delhi-bus-delay-model:v0]
       │  (Alias 'production' points to source artifact delhi-lightgbm_l1:v2)
       │
       ▼
[Model Artifact: delhi-lightgbm_l1:v2]
       │  (Digest: d7b1a20f9e4c3b8a1e5d2c7f4a9b0e3d, 66.8 MB .joblib)
       │  (Feature Contract: 25 features from data/processed/feature_spec.json)
       │
       ▼
[Lineage Metadata in Artifact]
       ├── git_commit: 946827e (source code state)
       ├── params_md5: 1bb1eb5c... (params.yaml configuration)
       └── data_processed_md5: 4e8867e... (DVC featurize output hash)
       │
       ▼
[DVC Dataset: dvc.lock]
       │  (Stage: featurize -> test.parquet, train.parquet, val.parquet)
       │  (Upstream Stage: build_segments -> clean -> data/raw/delhi-data-v2)
       │
       ▼
[DagsHub Storage Remote]
          DagsHub Git LFS / S3 storage for Delhi Bus GPS transit transitions
```

---

## 4. Documentation of System Inconsistencies

Transparent engineering requires identifying and explaining inconsistencies between local reports and cloud registries:

### 4.1 Local `v1` vs W&B `v0` Label Mismatch
* **Observed State**:
  - Local configuration file `reports/model_registry.json` lists `artifact_version: "v1"`.
  - Remote Weights & Biases Registry collection `delhi-bus-delay-model` records collection version `v0`.
* **Root Cause**:
  - The local pipeline tracking script (`src/models/registry.py`) initialized artifact counting at `1` (`v1`), representing the first candidate passing the quality gate.
  - The W&B Model Registry service indexes newly created collections using zero-based versioning (`v0`). Collection version `v0` in W&B references source artifact `delhi-lightgbm_l1:v2`.
* **Reconciliation**:
  Both records represent the exact same model weights (`lightgbm_l1`, MD5 `d7b1a20f...`, test MAE 30.60s). The version ledger (`reports/version_ledger.json`) bridges this mapping explicitly:
  ```json
  "registry": {
    "collection": "delhi-bus-delay-model",
    "versions": [
      {
        "version": "v0",
        "aliases": ["production"],
        "source": "delhi-lightgbm_l1:v2"
      }
    ]
  }
  ```

### 4.2 Invalidation Audit History
* **Observed State**: `reports/production_approvals.json` preserves an entry marked `status: "INVALIDATED"`.
* **Root Cause & Lineage**:
  - During Phase 2 development (commit `0d474af`), an automated CLI run executed `promote_to_production()` without human terminal operation.
  - In commit `15de323`, this entry was formally marked `INVALIDATED` and isolated.
  - In commit `1ef5a29`, Project Lead Anika executed an authorized promotion (`approved_by: "Anika"`), intentionally preserving the historical invalidation record to ensure an unalterable audit log.

### 4.3 Quality Gate Working Tree Flag (`git_dirty: true`)
* **Observed State**: `reports/quality_gate.json` lists `"git_dirty": true`.
* **Root Cause**:
  - When `src/models/quality_gate.py` executed during Phase 2 pipeline validation, active working tree files (`.gitignore`, `dvc.yaml`, `params.yaml`, `src/models/tracking.py`) were modified prior to the final Git commit.
  - The audit trail truthfully recorded the dirty working tree state rather than masking it.
