# Phase 3 demonstration

The Delhi model predicts the duration of a stop-to-stop bus segment in seconds. The following demonstrations use the same held-out model and release records as the prediction API.

| Requirement | What to run or open | Evidence and interpretation |
| --- | --- | --- |
| Induce and detect drift | `uv run python scripts/run_drift_demo.py`, then open the [HTML replay](../reports/drift_report.html) | [Window measurements](../reports/drift_report.json) cover all 25 inputs. Compare each SIMULATED scenario with the real, unmodified control. [Monitoring method and calibration](drift_monitoring.md) |
| Show and explain model versioning | `uv run python scripts/build_version_ledger.py` or open **Version ledger** on the dashboard | [Five-layer ledger](../reports/version_ledger.json) connects code, DVC data, model, image and configuration for reproducibility and rollback. [Versioning guide](versioning.md) explains the separate artifact and collection version numbers. |
| Show the MLOps pipeline | Open the [pipeline dashboard](https://delhi-bus-delay-api.onrender.com/static/dashboard.html). To run locally, use the command below and visit `/static/dashboard.html`. | [Site-data builder](../scripts/build_site_data.py) packages the evidence in the serving image. Compare validation and test MAE for seven models, inspect the eight [quality checks](../reports/quality_gate.json), and query the running model. |
| Explain and justify governance | `uv run python scripts/check_governance.py` or open **Governance controls** on the dashboard | Each [control](../reports/governance_status.json) names its risk and evidence. The [approval log](../reports/production_approvals.json) preserves the invalidated entry alongside the later human approval. See the [governance guide](governance.md) and [model card](model_card.md). |

## Reproduce locally

With the repository’s DVC remote configured, run:

```sh
uv sync
uv run dvc pull data/processed/train.parquet data/processed/val.parquet data/processed/test.parquet data/processed/feature_spec.json models/lightgbm_l1.joblib models/model_index.json
uv run python scripts/run_drift_demo.py
uv run python -m scripts.build_site_data
MODEL_PATH=models/lightgbm_l1.joblib uv run uvicorn src.api.main:app
```

## What the evidence supports

- The W&B replay created one `monitoring` run with notifications off. Its `status_table` contains 40 windows across five scenarios. [Monitoring run](https://forge.coreweave.com/wandb/anikaub20-rv-college/delhi-bus-delay/runs/4t0vffa3)
- The control passed. Peak traffic, road works and GPS outage first alerted in w4. Gradual drift first alerted in w6. Seven validation windows set the 14.72% warning and 22.94% alert limits. The replay uses real test days with seeded perturbations. Continuous production monitoring and delayed ground-truth collection remain planned.
- The dashboard packages a live W&B snapshot taken on 28 November 2026. Source artifacts `v0`, `v1` and `v2` have distinct digests. Registry collection `delhi-bus-delay-model:v0` links to `delhi-lightgbm_l1:v2` and carries `candidate`, `staging`, `latest` and `production`. W&B omits test MAE, code revision and dataset version from the metadata for source versions `v0` and `v1`. The local approval record uses its own version labels.
- Configuration evidence shows how CI/CD and deployment are wired. Only the live panel contacts the serving process. The runtime reports the model lineage commit from the registry record. GitHub Actions records the deployed image commit. A free-tier instance can take about a minute to wake.
- The governance report checks repository evidence. Branch protection requires administrator confirmation. The release policy requires quality checks and human approval. A drift alert can recommend investigation or candidate retraining.
