# Model Evaluation — TTC Bus Delay Prediction

Final evaluation on `data/processed/test.parquet` (year 2024, 58,320 rows) —
the only step in this project that touches the test split. No model
hyperparameters or features were changed based on these results.

## Results: all 6 models, validation vs test

| Model | val MAE | test MAE | test RMSE | test R² | test median AE |
|---|---|---|---|---|---|
| naive_baseline | 8.114 | 8.903 | 20.804 | 0.332 | 3.555 |
| global_median | 9.773 | 10.413 | 26.452 | -0.080 | 5.000 |
| ridge | 8.153 | 8.947 | 22.797 | 0.198 | 3.835 |
| random_forest | 7.924 | 8.644 | 22.582 | 0.213 | 3.673 |
| lightgbm | 7.871 | 8.607 | 22.699 | 0.205 | 3.619 |
| **lightgbm_l1 (best)** | **7.408** | **8.302** | **21.891** | **0.260** | **3.003** |

`lightgbm_l1` has the lowest MAE on both validation and test, and is the
model saved as `models/best_model.joblib`.

## Success criterion: NOT met

**Target:** ≥15% MAE improvement over the route×incident naive baseline, measured on test.

**Actual: 6.75% improvement** (naive_baseline test MAE 8.903 → lightgbm_l1 test MAE 8.302).
This is honestly short of the 15% target — `target_met: false` in
`reports/test_metrics.json`.

Against the simpler `global_median` baseline, the improvement is larger:
**20.27%** (10.413 → 8.302). The naive baseline is a much stronger
competitor than it looks, because `(route, incident)` grouping already
captures a lot of the real signal (certain routes/incident types are
reliably short or long delays) — the model has to beat a baseline that is
itself doing meaningful work, not a dummy constant.

## What the model does well

- Median absolute error is **3.00 minutes** — for a typical incident, the
  prediction is very close to the actual delay.
- **62.98%** of test predictions are within 5 minutes of the actual delay,
  and **79.79%** are within 10 minutes.
- It is reliably accurate on the two most common, well-represented incident
  types: `MECHANICAL` (MAE 3.70 min, 19,769 rows) and
  `OPERATIONS - OPERATOR` (MAE 4.20 min, 10,013 rows) — together over half
  of all test incidents.
- Error is lowest in the 5–20 minute delay range (MAE 2.48 for the 5-10
  bucket, 3.01 for the 10-20 bucket, by far the two largest buckets at
  11,062 and 24,488 rows) — the model is best exactly where most of the
  data lives.
- Known (non-`OTHER`) locations get a much lower MAE (6.29 min) than rows
  bucketed into `OTHER` (10.49 min, 27,926 rows — nearly half of test).

## Where it fails

- **Long delays are badly underpredicted.** The 60+ minute bucket has MAE
  **89.35 minutes** (1,982 rows) — these are rare, extreme events the model
  can't see coming from routine features. The 0-5 minute bucket is also
  comparatively poor (MAE 14.91 min, 6,668 rows): the model tends to predict
  a "typical" delay even when the actual delay is near zero.
- **`DIVERSION` incidents are the worst-predicted common incident type**:
  MAE 52.14 minutes (3,995 rows) — consistent with the EDA finding that
  diversions have the highest mean delay of any frequent incident type
  (~67 min) and are inherently high-variance.
- **`GENERAL DELAY`** is also weak (MAE 13.89 min, 4,063 rows) — a
  catch-all incident label that likely covers very different real causes.
- Rows with `location == "OTHER"` (the ~47.88% of rare/ungrouped locations
  from featurization) have 67% higher MAE than known locations (10.49 vs
  6.29 min) — location detail that got collapsed into OTHER really was
  carrying signal.
- Late night (1am-4am) and especially 2am have elevated MAE (11.96, 17.55,
  13.13, 12.74 minutes respectively) despite lower incident volume — this
  matches the EDA finding that late-night incidents are rarer but tend to
  run much longer, and the model doesn't fully capture that.
- Weekend MAE (8.83 min) is somewhat higher than weekday (8.13 min).

## Why L1 helps MAE, and why RMSE/R² stay modest

`min_delay` is heavily right-skewed (confirmed in the EDA: median 10 min,
p99 137 min) — a small share of incidents run extremely long. An L2
(squared-error) objective is dominated by those few huge-delay rows during
training, pulling the whole model's predictions upward to reduce big
squared errors, at the cost of accuracy on the much more common short/
medium delays. An L1 (absolute-error) objective weights every row's error
equally regardless of magnitude, so it optimizes directly for the median-
like behavior that MAE rewards — which is exactly why `lightgbm_l1` has the
best MAE (8.302) *and* the best median AE (3.003) of all six models.

RMSE and R² stay modest for the same underlying reason, but in reverse:
those metrics are quadratic in the error, so a model that's deliberately
not chasing the long tail (to keep MAE low) still gets heavily penalized by
RMSE/R² whenever that tail *does* show up — exactly what the 60+ minute
bucket's 89.35-minute MAE shows happening. The 60+ bucket has only 1,982
rows (3.4% of test) but contributes disproportionately to RMSE. This is an
intrinsic trade-off of this target's distribution, not a modeling mistake:
optimizing MAE and optimizing RMSE pull in different directions when the
target has a long tail like this one.

## Validation vs test gap

val MAE 7.408 → test MAE 8.302, a gap of **0.893 minutes** (~12% relative
increase). This is a real but moderate generalisation gap, not alarming on
its own, and it lines up with the EDA drift finding: mean delay rose from
15.12 min (2018-2019) to 17.13 min (2023-2024) and incident mix shifted
slightly. The gap is consistent with mild, gradual drift in the target
distribution and incident mix year over year, rather than a sign of
overfitting to 2023 specifically — validation (2023) and test (2024) are
adjacent years and the gap is modest.

## Limitations

- The 15% improvement target was not met; `lightgbm_l1` beats the naive
  baseline by 6.75% on test, which is real but modest.
- The naive baseline's strength (route × incident group means) means a
  large share of the "learnable" signal is already captured by that simple
  lookup — remaining gains likely require richer features (e.g. real-time
  conditions, more granular location, something that distinguishes
  `DIVERSION`/`GENERAL DELAY` causes) rather than more complex models on
  the current feature set.
- `location == "OTHER"` covers ~47.88% of rows and performs much worse than
  known locations — the rare-location grouping done in Phase 1 trades
  overfitting risk for a real loss of location-specific signal.
- Rare, very long delays (the 60+ bucket) are fundamentally hard to predict
  from routine features and drag down RMSE/R² regardless of which model is
  chosen.
- Only one validation year (2023) and one test year (2024) were used; the
  val→test gap is a single data point, not a trend.

## Next steps for Phase 2

- Serve `models/best_model.joblib` (`lightgbm_l1`) behind the Phase 2 API,
  reusing `data/processed/feature_spec.json` for the exact feature
  columns, target-encoding tables, and kept route/location sets.
- Investigate a dedicated path for `DIVERSION`/`GENERAL DELAY` incidents
  (e.g. a separate model or additional features specific to these types),
  since they are both common enough to matter and poorly predicted.
- Explore reducing the `OTHER`-location penalty: a coarser but non-trivial
  location feature (e.g. nearest major intersection/station, or geographic
  clustering) instead of collapsing ~47.88% of rows into one bucket.
- Consider a two-part model (classify "will this be a long delay?" then
  regress conditionally) to address the long-tail weakness without
  sacrificing median-case accuracy.
- Re-evaluate the 15% target's realism: given the naive baseline's
  strength, revisit whether 15% over *that* specific baseline is the right
  bar, or whether improvement over `global_median` (currently 20.27%) is
  the more meaningful comparison for Phase 2 planning.
