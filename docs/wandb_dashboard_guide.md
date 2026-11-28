# Build the W&B dashboard in ten minutes

Use the workspace for `anikaub20-rv-college/delhi-bus-delay`. All numbered actions below are **W&B UI clicks**. This verification session had no W&B API key, so no monitoring run was submitted. The keys below come from the repository's logging code and await live readback.

1. **0:00–2:00 · Runs table.** Open the project → **Workspace** → expand the runs table. Use the column controls to show run name, job type, `test_mae_best_model` and `val_mae_best_model`. Sort `test_mae_best_model` ascending and filter to rows with that value. The evaluator writes that test metric to `final-evaluation-test`. Training and sweep runs typically expose validation metrics. A single evaluation row is a valid result.
2. **2:00–4:00 · Model comparison.** In Workspace, select **Add panels** → **Bar plot** and choose the `final-evaluation-test` run. Use the following summary fields as separate metrics. If the editor accepts only one, use a custom chart that reshapes the summary fields to model and MAE, or present the existing Models page comparison. The seven values come from one evaluation run. [Evaluation logger](../src/models/evaluate.py)
3. **4:00–6:00 · Sweep.** Open **Sweeps** → `7rcgds9b` → **Parameter importance**. Select `val_mae` as the objective if the panel prompts for a metric. Show **Parallel coordinates** alongside it to explain the relationship among settings. The sweep minimizes validation MAE. [Sweep configuration](../sweeps/lightgbm_l1_sweep.yaml) · [W&B sweep UI](https://docs.coreweave.com/products/wandb/sweeps/visualize-sweep-results)
4. **6:00–9:00 · Drift replay.** If a verified monitoring run is available, select `drift-demo-simulated` with job type `monitoring`. Add two **Line plot** panels with `window_idx` as X. For Y, select each scenario's `<scenario>/mae` in the first and `<scenario>/max_psi` in the second. Open the run's `status_table` and show the `scenario`, `window`, `n`, `mae`, `mae_degradation_pct`, `max_psi` and `status` columns. The code also defines `<scenario>/mae_degradation_pct`, `<scenario>/psi/<feature>`, the summaries `n_alert` and `retrain_recommended`, and optional HTML media `drift_report_html`. Use the local replay if the run is absent. [Logging code](../src/monitoring/wandb_log.py) · [Local replay](../reports/drift_report.html)
5. **9:00–10:00 · Registry.** In the app navigation, select **Registry**, open the model registry and search for the `delhi-bus-delay-model` collection. Open **Versions** → `v0` and read its aliases and linked source artifact. The committed ledger records `production` on this collection version. Treat the UI as the evidence for its current assignment. [W&B Registry](https://docs.wandb.ai/models/registry)

## Model comparison fields

- `test_global_median_mae`
- `test_typical_baseline_mae`
- `test_recent_buses_baseline_mae`
- `test_ridge_mae`
- `test_random_forest_mae`
- `test_lightgbm_l2_mae`
- `test_lightgbm_l1_mae`

The scenario names for the two line plots are `clean_test`, `peak_traffic_surge`, `road_works`, `gps_outage` and `gradual_drift`. Keep the clean control visible and label the perturbations **SIMULATED**.
