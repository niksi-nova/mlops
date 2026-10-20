# Delhi Bus Travel-Time Model - Final Evaluation (TEST)

This is the one and only look at `data/processed/test.parquet` (day_index
77-92). No feature or model choice was changed based on these numbers -
`best_model` (`lightgbm_l1`) and `best_baseline` (`recent_buses_baseline`)
were both already selected on validation in `reports/train_metrics.json`;
this document only reports how those pre-selected choices perform on test.
Produced by `src/models/evaluate.py`; full numbers in
`reports/test_metrics.json`; figures in `reports/figures/eval/`.

## All 7 models: validation vs test MAE

| model | val MAE (s) | test MAE (s) |
|---|---:|---:|
| global_median | 68.56 | 71.13 |
| typical_baseline | 38.00 | 41.29 |
| recent_buses_baseline | 37.73 | 38.22 |
| ridge | 37.51 | 49.67 |
| random_forest | 30.30 | 31.78 |
| lightgbm_l2 | 29.84 | 31.30 |
| **lightgbm_l1 (best)** | **29.33** | **30.60** |

## Success criterion on test

Target: the best model beats the best baseline (`recent_buses_baseline`) by
at least `success_target_pct` = 10% MAE improvement.

- **Result: MET.** `lightgbm_l1` beats `recent_buses_baseline` by
  **19.93%** on test MAE (30.60s vs 38.22s) - well above the 10% bar, though
  somewhat lower than the 22.25% margin seen on validation.
- Improvement over the `global_median` baseline on test: **56.98%**
  (30.60s vs 71.13s).

## Validation vs test MAE (generalisation gap)

`lightgbm_l1`: val MAE 29.33s -> test MAE 30.60s, a **4.32%** increase
(`generalisation_gap_pct` in `reports/test_metrics.json`). This is a small,
unremarkable gap - well inside the "test can be a bit higher" expectation
and nowhere near the ~20%-worse threshold that would indicate a real
val/test mismatch.

## What the model does well

- Median absolute error on test is **12.27s** - half of all predictions are
  within about 12 seconds of the real travel time, even though the MAE
  (30.60s) is pulled up by a long right tail of slow/anomalous segments.
- **76.01%** of test predictions are within 30 seconds of actual, and
  **89.35%** are within 60 seconds.
- It is consistently the best of all 7 models across every single-number
  metric reported (MAE, RMSE, R2, median AE, within-30s%, within-60s%) -
  see `reports/figures/eval/01_test_mae_by_model.png`.

## Where it fails

**By travel-time bucket** (`reports/figures/eval/05_mae_by_travel_time_bucket.png`):
error grows sharply with segment length - MAE is 10.3s for 0-60s segments,
16.0s for 60-120s, 34.8s for 120-300s, 113.6s for 300-600s, and **397.9s**
for the 2,474 segments (1.6% of test rows) that take 600s or more. The
model is excellent on typical short segments and much weaker on the small
number of very long ones.

**By nodes_spanned**: segments that skip a node (`nodes_spanned > 1`, 6,843
test rows, 4.5%) have MAE **72.63s**, vs **28.64s** for normal single-node
segments - these skipped-reading segments are disproportionately hard,
consistent with the EDA finding that they have much higher raw travel time.

**By has_prev_bus**: rows with no qualifying previous bus (5,788 rows,
3.8% of test) have MAE **51.20s**, nearly double the 29.80s MAE for rows
that do have a previous-bus signal - confirming the recent-conditions
features are doing real work, and the model is weaker exactly where that
signal is unavailable.

**By route x direction**
(`reports/figures/eval/04_mae_by_hour.png` / route table in
`reports/test_metrics.json`): route 425 is harder than 534 in both
directions - 425 up MAE 43.66s (n=14,970), 425 down 39.58s (n=16,394), vs
534 down 31.13s (n=60,444) and 534 up only 24.55s (n=61,897). Route 425 has
far fewer training rows than 534 (see `docs/data_inventory.md`), which
likely explains the gap.

**By hour**: the single worst hour is hour 4 (MAE 73.72s), but that bucket
has only 1 test row and is not meaningful on its own (see EDA finding #6 on
hour-4 sparsity). Among hours with real volume, evening hours 18-19 are the
worst (MAE 43.08s and 44.57s on 8,074 and 7,147 rows respectively) -
consistent with evening congestion being harder to predict than the rest
of the day.

**Worst 10 segments by MAE** (`segment_id`, MAE, row count):

| segment_id | MAE (s) | rows |
|---|---:|---:|
| 425_up_1 | 293.78 | 363 |
| 534_down_54 | 195.03 | 1,112 |
| 534_down_19 | 140.08 | 1,171 |
| 425_down_40 | 110.01 | 340 |
| 425_down_36 | 107.61 | 424 |
| 425_down_37 | 102.00 | 376 |
| 425_down_33 | 100.71 | 442 |
| 534_down_1 | 99.12 | 1,205 |
| 425_up_7 | 83.94 | 433 |
| 425_down_10 | 83.22 | 420 |

`425_up_1` was also the slowest segment by median travel time in the EDA
(finding #7), so this is a known, persistently hard bottleneck segment, not
a new surprise.

## Delay view (test rows with delay_s > 60s)

**10.90%** of test rows (`share_of_rows_delayed_gt_60s`) are more than 60
seconds slower than their months-1-2 typical baseline. On exactly these
delayed rows:

| model | MAE on delayed rows (s) |
|---|---:|
| typical_baseline | 220.42 |
| recent_buses_baseline | 161.16 |
| **lightgbm_l1 (best model)** | **136.05** |

The best model is meaningfully better than both baselines specifically on
the hardest (delayed) rows - `recent_buses_baseline` already helps a lot
over the static `typical_baseline` here (161.16s vs 220.42s), and the full
model adds a further improvement on top of that (136.05s).

## Daily MAE trend (day_index 62-92, val + test)

See `reports/figures/eval/06_daily_mae_val_test.png`. Mean daily MAE is
**29.55s** across the 15 val days and **30.70s** across the 16 test days -
close to each other, and day-to-day noise (daily MAE ranges roughly
24-39s) dominates any systematic trend. Splitting the 16 test days in half,
the **first 8 test days average 31.60s** and the **second 8 average
29.81s** - i.e. error does **not** creep upward across month 3; if
anything it drifts slightly down toward the end. The single highest daily
MAE in the whole 62-92 window is the last validation day (day_index 76,
~38.6s), a one-day spike rather than part of a trend. This is a reassuring
Phase-3 preview: nothing in this 31-day window suggests the model is
stale by the end of test.

## Why recent-bus features matter

Two independent pieces of evidence in this evaluation confirm the
recent-conditions features earn their place:

1. `recent_buses_baseline` (which just predicts the median of the last 3
   qualifying buses on the same segment) is already the stronger of the
   two baselines on test (38.22s vs `typical_baseline`'s 41.29s MAE), and
   on the delayed-rows slice it's even more clearly ahead (161.16s vs
   220.42s) - recent conditions track real-time congestion that a static
   historical median cannot.
2. The `has_prev_bus` breakdown above shows the full model is nearly twice
   as accurate when a previous-bus signal is available (29.80s vs 51.20s
   MAE) - the model has clearly learned to lean on that feature.

## Sweep result and interpretation

The W&B Bayesian sweep (`7rcgds9b`, 20 runs) found `num_leaves=255,
learning_rate=0.02217, min_child_samples=50, colsample_bytree=0.829,
subsample=0.879, reg_lambda=0.0265` as the best configuration, with
validation MAE **29.321s** vs **29.434s** for the untuned defaults - a
**~0.38%** relative improvement (consistent with the "about 0.4% gain"
expected). All 20 sweep trials beat or matched the default, and the top
few were tightly clustered, with higher `num_leaves` and a lower
`learning_rate` than the default showing up repeatedly among the leaders.
**Interpretation**: this is a small, consistent refinement from
hyperparameter tuning, not a breakthrough - the real gains in this project
came from feature engineering (recent-bus signal, historical typicals),
not from tuning LightGBM harder. With the tuned params adopted in
`params.yaml`, the retrained model scored val MAE **29.33s** (matching the
sweep's winning run closely) and test MAE **30.60s**.

## Limitations

- **Only 2 routes** (425, 534) - the model has not seen any other Delhi
  bus route and should not be assumed to generalize to one.
- **No real calendar dates or weather.** `relative_weekday` is a synthetic
  7-day cycle, not tied to an actual weekday; there is no way to join
  weather, and no holiday/seasonal signal exists in this data at all (see
  `docs/data_inventory.md` and `docs/eda_findings.md`).
- **No GPS coordinates.** All positional features are in terms of `Node`
  (a stop-sequence index), not physical distance - the model cannot reason
  about road geometry, distance in km, or anything location-based beyond
  the node/segment identity itself.
- **Ridge's poor test R2 (-18.64, vs -2.26 on val)** shows the log1p-space
  linear model backtransforms very badly on the long right tail of
  `travel_time_s` for a handful of outlier rows - its MAE is still
  reasonable, but it should not be trusted for anything relying on
  variance/R2, and this gap also illustrates why generalisation gaps can
  look very different on different metrics for the same model.

## Next steps for Phase 2

1. Investigate the handful of worst segments (`425_up_1`, `534_down_54`,
   `534_down_19`, ...) directly - these may reflect genuine physical
   bottlenecks (traffic signals, junctions) worth a dedicated feature
   rather than being left to the general model.
2. Route 425's weaker performance (43.66s / 39.58s MAE vs 534's 24.55s /
   31.13s) tracks its much smaller row count - either collect more route
   425 data or consider a route-aware regularization/weighting scheme.
3. The `nodes_spanned > 1` and `has_prev_bus=0` slices are the two clearest
   weak spots (MAE 72.63s and 51.20s respectively) - targeted features or
   a separate model for these minority-but-high-error slices could help.
4. Since calendar dates are unavailable, any real deployment should log
   true dates going forward so a genuine day-of-week/holiday signal and
   weather join become possible in a future dataset version.
5. Given the daily-MAE trend is flat within this 31-day window, re-run this
   same evaluation after more calendar time has passed to confirm the
   model doesn't degrade over a longer horizon than this dataset covers.
