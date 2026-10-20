# Delhi Bus GPS - EDA Findings

Computed from `data/interim/segments.parquet` (901,487 segments, 21,116
trips, 202 vehicles, 92 days, routes 425/534 x up/down) and
`data/interim/stop_events.parquet`, via `notebooks/01_eda.ipynb`. All
figures are in `reports/figures/` (`01_...` to `16_...`).

## Key findings

1. The dataset has 901,487 stop-to-stop segments from 21,116 trips across 202 vehicles and 92 days, with zero missing values in any column - segments are heavily skewed toward route 534 (722,166 segments, 80%) versus route 425 (179,321 segments, 20%).
2. `travel_time_s` is right-skewed with median 90s, mean 118.9s, std 125.2s, p95 320s and p99 658s, hard-bounded at [1s, 1800s] by the cleaning pipeline's `segments.min/max_segment_seconds`.
3. `seconds_per_stop` (travel_time_s normalized by nodes_spanned) has a lower median of 80s and mean of 111.3s than raw travel_time_s, since it correctly discounts segments that skip multiple nodes.
4. Only 3.8% of segments have `nodes_spanned > 1` (a skipped node reading), and these have far higher raw travel time (mean 315.4s vs 111.1s for single-node segments) but a similar normalized `seconds_per_stop` (mean 116.2s vs 111.1s), confirming the normalization works as intended.
5. Travel time follows a clear rush-hour pattern: mean travel time peaks at hour 18 (152.9s) and is fastest at hour 5 (66.9s), while trip departures peak at hour 8 (1,961 trips starting) - classic morning-dispatch, evening-congestion behaviour.
6. `relative_weekday` 0 covers 14 distinct days (`day_index` 1, 8, 15, ..., 85, not just the first day of collection), and restricted to daytime hours 7-20 (hour 4 has only 41 segments from 15 trips across the whole 92-day window and is too noisy to use), it has the fastest mean travel time of all seven weekday groups - 108.5s vs 121.5-127.5s for the others, 13.0s faster than the next-fastest weekday - which, being consistent across all 14 of its days, looks like a genuine weekly effect (possibly a Sunday-like, lower-traffic day) rather than a one-off artifact; note it does *not* have the fewest trips (2,719-3,161 daytime trips across groups, `relative_weekday` 0 actually highest), so the effect is about speed, not volume.
7. The slowest segment by median travel time is route 425 `up` from `from_node` 1 (median 720s, IQR 530s) - a clear outlier bottleneck compared to typical segment medians in the tens-of-seconds-to-low-hundreds range.
8. Segment-level variability (IQR) is uneven across stops: the top bottleneck segments (e.g. route 425 up node 1, route 534 down node 26) have IQRs of 181-530s, far above the typical segment.
9. The time-space diagram for route 534's busiest day (`day_index` 1) shows dense, regular interleaved up/down trip lines, consistent with a real scheduled bus service rather than noise.
10. A bus's travel time on a segment correlates with the immediately preceding bus on the same route/direction/segment that day (r=0.727 over 870,377 matched pairs, same-day and strictly-earlier only) and correlates even more strongly with the median of the previous 3 buses (r=0.746).
11. Volume is concentrated in a subset of vehicles: the 10 most frequent vehicles each contribute 25,847-31,261 segments, well above the per-vehicle average of ~4,463 segments (901,487 / 202 vehicles).
12. There is a mild month-over-month upward drift in mean travel time: 116.99s (month 1) -> 118.18s (month 2) -> 121.70s (month 3), while the median stays close (80-90s across all three months).
13. Using months 1-2 medians per (route, direction, from_node, hour) as a "typical" baseline, month-3 `delay_s` has mean 19.0s, median 0s, and 10.70% of month-3 segments are more than 60s slower than typical (294,518 matched segments, 99.9993% of month-3 segments had a baseline to compare against).
14. The pipeline's bounds filtering (`segments.min_segment_seconds=1`, `max_segment_seconds=1800`) and the clean-stage trip filters mean the segments table reaching EDA already has 0 nulls and 0 raw duplicates - the remaining data-quality work is about choosing good features, not more cleaning.

## Decisions for feature engineering

| Finding | Modelling decision |
|---|---|
| #2, #3: travel_time_s is right-skewed, seconds_per_stop better normalizes multi-node segments | Predict/evaluate on `seconds_per_stop` (or `log1p(travel_time_s)`) rather than raw `travel_time_s`, especially for multi-node segments |
| #4: 3.8% of segments span >1 node with very different raw travel time | Include `nodes_spanned` as a feature, or model single-node and multi-node segments with awareness of this split (e.g. an indicator feature) |
| #5: strong hour-of-day effect, rush-hour peak at 18:00 | Include `hour` (and a rush-hour flag) as a core feature |
| #6: `relative_weekday` 0 is consistently ~13s faster during daytime hours across all 14 of its days - a genuine weekly effect | Keep `relative_weekday` as a real feature (not just a weak/candidate one); the effect looks like a Sunday-like lower-traffic day rather than a data artifact |
| #7, #8: a handful of bottleneck/high-variance segments dominate tail risk | Include `(route, direction, from_node)` as a categorical/target-encoded feature so the model can learn per-segment baselines |
| #10: strong correlation with the previous bus (r=0.727) and previous-3-median (r=0.746) | Build `prev_bus_travel_time_s` and `prev3_median_travel_time_s` as serving-time features, using only strictly-earlier same-day buses (as validated in the notebook) |
| #11: a small set of vehicles carries disproportionate volume | Consider a vehicle-level feature or random effect if vehicle-specific speed differences are confirmed to matter beyond route/direction/node |
| #12, #13: mild month-over-month drift, 10.7% of month-3 segments >60s slower than a months-1-2 baseline | Track drift explicitly (e.g. a rolling/recent-baseline feature) rather than training once on months 1-2 and assuming stationarity; consider a `delay_s`-style target or monitoring metric for production |
| #14: data already clean (0 nulls/duplicates) | No further cleaning needed before featurization; focus effort on feature design, not data repair |

## Data limitations

- **No real calendar dates.** Only a synthetic `day_index` (1..92) and `relative_weekday` (`(day_index-1) % 7`) exist - there is no way to know which real-world weekday, holiday, or season any segment belongs to (see `docs/data_inventory.md`).
- **No weather data.** Since there are no real dates, no weather source can be joined to this data; `src/data/fetch_weather.py` and `src/data/merge.py` were removed from the project for this reason.
- **No GPS coordinates.** The only positional information is `Node`, an integer stop-sequence index per route/direction - there is no latitude/longitude and no way to compute real distances, so all travel-time modelling is in terms of node-to-node segments, not physical distance.
- **Only 2 routes.** Routes 425 and 534 only - any model trained on this data should not be assumed to generalize to other Delhi bus routes without further data.
- **Relative weekday only, not validated against a real week.** `relative_weekday` is a 7-day cycle over `day_index`, not a real weekday label - finding #6 shows a consistent speed effect for `relative_weekday` 0 across all 14 of its days, which is suggestive of a genuine weekly pattern (e.g. Sunday), but without real calendar dates this cannot be confirmed to actually be a Sunday or any other specific day.
