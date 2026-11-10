# Dataset Card: Delhi Bus GPS

## Source

- **Provider:** [bus-data.github.io](https://github.com/bus-data/bus-data.github.io) -
  a research dataset of Delhi real-time bus GPS data, already stop-matched
  (positions are given as an integer `Node` stop-sequence index, not raw
  lat/lon).
- **Download URL:** https://github.com/bus-data/bus-data.github.io/raw/master/asset/dataset.zip
  (set as `data.source_url` in `params.yaml`, fetched by `src/data/ingest.py`).
- **Licence:** none stated in the source repository. Used here for academic
  research purposes only, with attribution to the original provider; no
  licence terms are claimed or implied by this project.

## Coverage

- **Routes:** 2 - route `425` and route `534`, each with `up` and `down`
  directions.
- **Time span:** 92 consecutive relative days, grouped into 3 relative
  "months" (`Month_1`: 31 days, `Month_2`: 30 days, `Month_3`: 31 days).
  **There are no real calendar dates anywhere in the archive** - checked
  file names, folder names, cell contents and Excel workbook metadata; see
  `docs/data_inventory.md` for the full investigation. Day ordering within
  and across months is reliable (no missing day-labels), but the data
  cannot be tied to a specific real-world week, month or season.

## Raw data

- **924,375 rows** across **184 files** (**43.5 MB** on disk), one file per
  (route, month, day) at `data/raw/delhi_gps/Month_<m>/<route>/Day<d>.xlsx`.
- **Columns** (identical in every file):
  - `Node` (int): stop-sequence index along the route/direction (not a
    GPS coordinate) - low to high in travel order for both directions.
  - `Number` (str): vehicle registration plate.
  - `Route` (str): actually encodes **direction** (`up`/`down`) - the real
    route number (`425`/`534`) comes from the folder, not this column.
  - `Time` (str, `HH:MM:SS:mmm`): time of day only, no date.
- 202 unique vehicles (46 on route 425, 156 on route 534); node coverage
  41/37 distinct nodes for route 425 down/up and 56/52 for route 534
  down/up.

## Cleaning results (`src/data/clean.py`, `reports/cleaning_report.json`)

- 924,375 raw rows -> 0 duplicate rows removed -> 22,127 candidate trips
  after splitting on node resets / long gaps (with midnight-crossing trips
  corrected: **38 trips** got a +86400s offset instead of being wrongly
  split or dropped).
- Trip filtering removed 1,011 distinct trips (0 for node going backward,
  11 for time going backward after the midnight fix, 1,000 for having
  fewer than `clean.min_nodes_per_trip` recorded stops), removing 1,734
  rows.
- **Final: 922,641 rows across 21,116 trips** (425 down 2,966 / 425 up
  2,902 / 534 down 7,500 / 534 up 7,748 trips).

## Segment table (`src/data/build_segments.py`, `reports/segments_report.json`)

- 901,525 stop-to-stop segments built from consecutive stop arrivals; 1
  zero-span segment dropped (two arrivals at the same node) and 37 dropped
  for being outside `[min_segment_seconds, max_segment_seconds]` =
  `[1, 1800]` seconds.
- **Final: 901,487 segments.** 3.84% have `nodes_spanned > 1` (a skipped
  node reading). `travel_time_s` percentiles: p50 90s, p75 140s, p90 220s,
  p95 320s, p99 658s.
- Validated with Pandera (`src/data/validate.py`) before use - this check
  caught a real zero-length-segment bug during development (see
  `docs/project_overview.md` risk R1).

## Train / validation / test split (`reports/featurize_report.json`)

A day-index split, so no trip is ever cut across splits and no information
from a later day leaks into training:

| split | day_index range | rows |
|---|---|---:|
| train | 1-61 (Month_1 + Month_2) | 606,967 |
| val | 62-76 | 140,815 |
| test | 77-92 | 153,705 |

## DVC-versioned raw data

- **`delhi-data-v1`**: months 1-2 only (`data.months_included: [1, 2]`).
- **`delhi-data-v2`**: months 1-3, the full archive (`data.months_included: [1, 2, 3]`)
  - this is the version used for the Phase 1 pipeline and all numbers in
    this document.

Both are DVC-tracked (`data/raw.dvc`) and pushed to the DagsHub remote; see
the git tags `delhi-data-v1` / `delhi-data-v2`.

## Known limitations

- No real calendar dates (relative `day_index`/`relative_weekday` only).
- No GPS coordinates - positions are an integer stop-sequence index only.
- Only 2 routes; not representative of the full Delhi bus network.
- No licence stated by the source.
- No weather, schedule/GTFS, or holiday data included or joinable (no real
  dates to join on).

## Why this dataset (over the alternatives considered)

- **vs. BMTC (Bangalore) real-time GPS data:** BMTC's publicly available
  snapshot covers only a **single day**, which rules out any time-based
  train/val/test split or measuring real drift over time - both central to
  this project's MLOps goals (Phase 1 generalisation gap, Phase 3 drift
  monitoring). This Delhi dataset's **3 relative months (92 days)** make a
  genuine time-based split and a real val-to-test drift measurement
  possible (see `docs/model_evaluation.md`).
- **vs. Telangana/Andhra Pradesh public transport sources:** those publish
  **scheduled timetables only**, not actual vehicle movements - there is no
  real travel-time signal to predict, only the published schedule itself.
  This Delhi dataset instead records **actual stop-to-stop GPS-derived
  travel times**, which is what the problem (predicting real travel
  time/delay, not just repeating a timetable) requires.
