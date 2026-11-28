# Phase 3 demo and viva

Before presenting, open the [drift replay](../reports/drift_report.html), [version ledger](../reports/version_ledger.json), [governance report](../reports/governance_status.json) and [public dashboard](https://delhi-bus-delay-api.onrender.com/static/dashboard.html). Check the public service with `uv run python scripts/check_live.py`. Keep the local dashboard open or save its screenshots in case the free instance is slow to wake.

## 0:00–2:30 · Drift

**Open:** With the DVC splits and model available, run `uv run python scripts/run_drift_demo.py --out /tmp/delhi-demo/drift_report.json` and open the neighboring HTML report. If replay is still running, show the committed HTML and compare the clean control with gradual drift. For the W&B view, open `status_table` on the finished run. It contains eight windows for each of the five scenarios. [Monitoring run](https://forge.coreweave.com/wandb/anikaub20-rv-college/delhi-bus-delay/runs/4t0vffa3)

**Say:** “These are SIMULATED perturbations on real test days. The unmodified control stayed OK, and gradual drift first alerted in w6. Road works produced a single alert, below the two consecutive alerts needed to recommend retraining.” [Window results](../reports/drift_report.json)

## 2:30–4:30 · Versioning

**Open:** Show the **Version ledger** dashboard section. If the dashboard is unavailable, open [reports/version_ledger.json](../reports/version_ledger.json) and show `layers`, `model_versions` and `registry`.

**Say:** “Code, data, model, image and configuration have separate identifiers so we can reconstruct a release or choose what to roll back. The live W&B ledger shows three distinct artifact digests. The registry collection’s v0 carries the production alias and links to source artifact delhi-lightgbm_l1:v2.” [Versioning guide](versioning.md)

## 4:30–7:30 · Pipeline dashboard

**Open:** Visit the [public dashboard](https://delhi-bus-delay-api.onrender.com/static/dashboard.html). Walk the pipeline strip, compare models, select a drift scenario, then show **Live service**. If Render is unavailable, run `MODEL_PATH=models/lightgbm_l1.joblib uv run uvicorn src.api.main:app` with the DVC model and feature spec available, then visit [the local dashboard](http://127.0.0.1:8000/static/dashboard.html). If the model or server is unavailable, use prepared dashboard screenshots and the committed reports.

**Say:** “The dashboard shows evidence from ingestion through monitoring. The held-out MAE is 30.60 s, and the candidate passed all eight gate checks. The charts use checked-in reports, while the live panel asks the serving process whether the model is loaded.” [Quality gate results](../reports/quality_gate.json)

## 7:30–10:00 · Governance

**Open:** Show **Governance controls**, then the [approval history](../reports/production_approvals.json). If the dashboard is unavailable, use the committed governance report and approval history side by side.

**Say:** “A passing gate lets the team consider a candidate. Serving it requires explicit human approval, so the audit keeps the invalidated approval alongside the authorized approval. G09 branch protection is MANUAL, and continuous drift monitoring is PLANNED.” [Governance report](../reports/governance_status.json)

## Viva questions

1. **Did production drift?** This is a seeded replay on real test days. The dataset also shifts naturally. Monthly mean segment times were **117.0, 118.2 and 121.7 s**, rounded from the saved notebook output. [EDA findings](eda_findings.md) · [EDA notebook](../notebooks/01_eda.ipynb)
2. **Did you tune the drift thresholds on the test replay?** Validation windows set the **14.72% warning** and **22.94% alert** limits. We applied the same limits to the test control and simulated scenarios. [Calibration method](drift_monitoring.md) · [Drift report](../reports/drift_report.json)
3. **Why exclude weekday and vehicle encoding from alerting?** A short window covers only part of the weekly cycle, and vehicle encoding is out-of-fold in training but uses a fixed training-fitted map on held-out data. Both `relative_weekday` and `te_vehicle` remain measured and visible in every window. [Drift report](../reports/drift_report.json)
4. **Why did road works alert without a retraining recommendation?** It produced one isolated alert. The policy requires two consecutive alert windows. A recommendation still requires a candidate to pass the quality gate and receive human approval. [Drift report](../reports/drift_report.json)
5. **Can PSI alone establish model degradation?** PSI measures distribution shifts. Prediction error requires matching observed travel times to predictions, and those outcomes may arrive later. [Monitoring guide](drift_monitoring.md)
6. **Why are there different version numbers?** The local approval refers to `v1`. W&B independently identifies the collection entry as `v0` and its linked source artifact as `delhi-lightgbm_l1:v2`. The live API confirmed the link and the collection’s `production` alias. [Approvals](../reports/production_approvals.json) · [Version ledger](../reports/version_ledger.json)
7. **What would you identify before rolling back?** Find the approved artifact and its feature spec, the matching image and code revision, and the dataset and config used for evaluation. Verify the running version and its health afterward. [Version ledger](../reports/version_ledger.json)
8. **Is the model's reported git commit the deployed image's commit?** The API reports model lineage from registry metadata. The CD job publishes the image tag for its own source revision. Check those identifiers separately. [Deployment notes](deployment.md)
9. **Why show both validation and test MAE?** The team selects and tunes on validation, then evaluates on held-out data. The gate records **29.33 s validation MAE**, **30.60 s test MAE** and a **4.32%** gap. [Quality gate](../reports/quality_gate.json)
10. **Does the benchmark include a Render cold start?** The reported **15.70 ms median** and **18.00 ms p95** measure warmed single-row inference. The smoke checker separately allows time for service startup. [Quality gate](../reports/quality_gate.json)
11. **Why keep an invalidated approval?** It records the unauthorized action and its invalidation. The later authorized entry supplies the release evidence while preserving the audit trail. [Approval history](../reports/production_approvals.json)
12. **Which governance controls still require work?** G09 branch protection requires administrator confirmation, and G10 continuous monitoring is planned. The status report records **8 PASS, 0 FAIL, 1 MANUAL and 1 PLANNED**. [Governance report](../reports/governance_status.json)
