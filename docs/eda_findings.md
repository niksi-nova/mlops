# EDA Findings — TTC Bus Delay Data

Source: `notebooks/01_eda.ipynb`, run against `data/interim/merged.parquet`
(382,131 rows, 2018-01-01 to 2024-12-31). Figures referenced below are in
`reports/figures/`.

## Key findings

1. The dataset has 382,131 rows across 480 routes, 58,640 distinct location
   strings, 34 incident types, and 3,998 vehicles.
2. Missingness is low and concentrated in a few columns: `vehicle` is missing
   on 6.76% of rows, `incident` on 0.24%, `min_gap` on 0.14%, and `location`
   on 0.08%.
3. `min_delay` is heavily right-skewed: median 10 min, p75 18 min, p90 28 min,
   p95 30 min, p99 137 min, mean 16.22 min — the mean is pulled well above
   the median by a long tail of rare, very large delays.
4. 4.67% of rows have a zero-minute delay (kept in the data per
   `params.yaml: drop_zero_delay: false`).
5. Incidents peak at 15:00 (3pm) by volume, but the *highest* mean delay by
   hour is at 02:00 (26.99 min) — late-night incidents are rarer but tend to
   be much longer once they happen. Weekend incidents also run longer on
   average than weekday ones (17.82 min vs. 15.78 min).
6. 2020 incident volume drops 42.0% versus 2019 (35,633 vs. 61,471 rows),
   consistent with reduced TTC service during the COVID-19 pandemic; mean
   delay that year (15.80 min) is close to the overall average, so the drop
   is in volume, not severity.
7. `MECHANICAL` is the most frequent incident type (121,438 rows, 31.78% of
   all incidents), but it is not the most delay-heavy: `DIVERSION` has the
   highest mean delay among frequent types, at 67.00 minutes.
8. Route `32` has the most incidents of any route (11,746 rows) among the
   top 15 routes by volume.
9. Locations are extremely fragmented: of 58,640 unique location strings,
   54,726 (93.3% of distinct locations) occur fewer than 10 times each,
   together accounting for 23.27% of all rows — most individual location
   values are not reusable as a stable categorical feature on their own.
10. `UNKNOWN` is the third most common "direction" value (68,807 rows,
    18.01% of all rows) — direction cleaning leaves a large non-trivial
    bucket of unmapped text, mostly behind N (84,783), S (79,423), and
    ahead of E (77,222) and W (71,049); `B` ("both ways") is rare (847 rows).
11. Weather effects on mean delay are small in this univariate view: heavy
    precipitation (>2.5mm) averages 17.31 min vs. 16.22 min with no
    precipitation; any snowfall averages 16.59 min vs. 16.20 min with none —
    real but modest shifts, not dominant drivers on their own.
12. `min_gap` is strongly correlated with `min_delay` (r = 0.88, n =
    381,601) — it is recorded as the gap to the next scheduled vehicle,
    which TTC only finalises after the disrupted trip (and its delay) is
    already known, so using it as a model input would leak the target.
13. Comparing 2018-2019 to 2023-2024: mean delay rose from 15.12 min to
    17.13 min (median 10 to 11 min), and `MECHANICAL`'s share of incidents
    rose slightly from 32.69% to 34.29% — a real but moderate shift in both
    the target distribution and the incident-type mix.

## Decisions for feature engineering

| Finding | Modelling decision |
|---|---|
| `min_delay` is heavily right-skewed (median 10, p99 137, mean 16.22) | Train on `log1p(min_delay)` and/or evaluate with MAE or a quantile loss rather than plain MSE/RMSE, which would be dominated by the long tail. |
| Strong hour-of-day and weekend/weekday pattern (peak volume at 15:00, peak mean delay at 02:00, weekend 17.82 vs. weekday 15.78 min) | Use cyclic hour encoding (`hour_sin`/`hour_cos`, already built in `src/features/time_features.py`) plus `is_weekend` / `is_rush_hour` as model inputs instead of raw hour. |
| 2020 is a volume outlier (-42.0% incidents vs. 2019) from COVID service cuts | Keep `year` as a feature but treat 2020 as a known regime shift rather than noise; consider a year/era indicator or down-weighting 2020 if it hurts generalisation to recent years. |
| Incident type has very different count vs. delay rankings (`MECHANICAL` most frequent at 31.78%, `DIVERSION` most delay-heavy at 67.0 min) | Keep incident type as a categorical feature (one-hot or target-encoded) — count alone would hide that some rarer types drive the longest delays. |
| Locations are extremely fragmented (93.3% of 58,640 distinct locations occur <10 times, 23.27% of rows) | Group rare locations into an "OTHER" bucket (or target-encode with smoothing) rather than one-hot encoding raw location strings, to avoid a huge sparse, overfitting-prone feature space. |
| `UNKNOWN` is 18.01% of the `direction` column | Keep `UNKNOWN` as its own explicit category rather than imputing it — it's too large a share to drop or guess, and may itself carry signal (e.g. correlate with certain incident types or locations). |
| Weather shows only modest univariate effects (heavy precip 17.31 vs 16.22 min; snow 16.59 vs 16.20 min) | Keep weather features but don't expect them to be dominant on their own; consider interactions (e.g. snowfall x rush hour) rather than relying on main effects alone. |
| `min_gap` correlates with `min_delay` at r = 0.88 and is recorded after the incident | Drop `min_gap` entirely from the modelling feature set — it is post-outcome information and would leak the target. |
| Mean delay drifted from 15.12 to 17.13 min and incident mix shifted (`MECHANICAL` 32.69% -> 34.29%) between 2018-2019 and 2023-2024 | Use a time-based (not random) train/validation/test split, as already planned in `docs/dataset_card.md` section 7, and monitor for further drift rather than assuming the target distribution is stationary. |

## Data limitations

- `location` is free text with no canonical gazetteer, so the same physical
  stop can appear under many different spellings — the 58,640-value count in
  finding 9 almost certainly overstates the number of truly distinct
  locations.
- `vehicle` is missing on 6.76% of rows and was not explored further here;
  if used, its missingness should be modelled explicitly rather than imputed
  silently.
- Weather is hourly and matched by rounding the incident timestamp down to
  the hour (see `src/data/merge.py`), so it reflects conditions at the start
  of that hour, not necessarily the exact minute of the incident.
- `UNKNOWN` direction (18.01% of rows) is a cleaning artifact of
  `normalise_direction` rather than a true "no direction" category, and its
  composition (which original free-text values fall into it) was not broken
  down further in this pass.
- The drift comparison (finding 13) only contrasts two 2-year windows; it
  does not establish a trend shape (gradual vs. step change) and 2020-2022
  were excluded from that specific comparison to avoid mixing in the
  COVID-era volume anomaly.
