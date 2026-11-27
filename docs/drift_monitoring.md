# Monitoring Delhi bus travel times

Run the two-minute demo from the repository root with the processed DVC splits and the trained LightGBM model available.

```sh
uv run python scripts/run_drift_demo.py
```

Open [the HTML report](../reports/drift_report.html) to inspect a scenario window by window. The [JSON report](../reports/drift_report.json) includes the test statistics for every model input. The demo runs locally. The reference consists of 606,967 training segments from days 1–61. Prediction uses the column order recorded by the trained booster.

## What changes in each scenario

| Scenario | Observation |
| --- | --- |
| clean_test | Real, unmodified days 77–92. A passing control requires every window to be OK. |
| peak_traffic_surge · SIMULATED | More trips occur at peak times and recent bus times rise. This changes the input distribution (data drift). |
| road_works · SIMULATED | Selected segments take longer at the same input values. This changes the target conditional on the inputs (concept drift). |
| gps_outage · SIMULATED | Previous-bus observations disappear. Coverage and the distribution of real-time inputs change. |
| gradual_drift · SIMULATED | Recent bus times and observed travel times increase progressively. Inspect when the first alert appears. |

Performance drift is a change in prediction error. Computing MAE needs observed travel times, which can arrive later than predictions. A PSI shift alone cannot establish a change in accuracy or its cause.

## Reading the signals

The [drift configuration](../params.yaml) sets inclusive PSI warning and alert thresholds of 0.10 and 0.25. Numeric features also flag ALERT when the two-sample KS p-value is below 0.01 and its statistic is at least 0.10. Categorical features use PSI for status and report a chi-square p-value. The ordered weekday codes also have a diagnostic KS statistic. Numeric KS uses a seeded sample of 50,000 training rows.

All 25 features appear in every window. Two have `alerting_excluded` set. A two-day window contains only part of the weekly cycle for `relative_weekday`. For `te_vehicle`, training uses out-of-fold target encodings while held-out data uses a fixed mapping fitted on the full training set. These expected differences make their distributions poor triggers against the training reference. Their statistics remain available for inspection. Window max PSI, the list of flagged inputs and the top features use the eligible inputs.

A window reaches ALERT if eligible max PSI, target PSI or prediction PSI reaches 0.25, or if MAE degradation reaches 22.94%. It reaches WARNING if eligible max PSI reaches 0.10 or MAE degradation reaches 14.72%. Other windows are OK. A KS flag identifies a feature for investigation. A scenario takes the most severe status of its windows. Two consecutive ALERT windows recommend retraining. Serving a candidate still requires the quality gate and typed human approval.

## Calibrating performance limits

Reproduce the calculation with `uv run python -m src.monitoring.calibrate`. Score the validation period (days 62–76) and express each complete, non-overlapping two-day window’s MAE as a percentage change from the full validation MAE of 29.3348 s. Day 76 contributes to the baseline only. The seven window MAEs were 26.427, 27.876, 31.607, 30.974, 25.199, 29.327 and 30.398 s. Their mean percentage change was −1.7214%, with a sample standard deviation of 8.2195 percentage points. Warning uses mean plus two standard deviations and alert uses mean plus three, rounded to two decimals. Apply those fixed percentage limits to the demo’s existing test-period baseline of 30.60 s.

Seven windows give a rough operating limit. They offer little information about the tails or seasonal patterns. Even the unperturbed monthly mean travel times change across the dataset (117.0, 118.2 and 121.7 s). The test-period baseline describes this retrospective demonstration. A live system needs a baseline frozen before monitoring starts, aligned predictions and delayed outcomes, and enough history to estimate its false-alarm rate.

## Observed replay

All eight control windows were OK. The first alerts for peak traffic, road works and GPS outage occurred in w4 (days 83–84). Gradual drift first alerted in w6 (days 87–88). Road works produced an isolated alert, below the consecutive-window retraining rule.
