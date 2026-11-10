# Demo website

A multi-page static site (vanilla HTML/CSS/JS, no build step, no framework)
served by the same FastAPI app as the prediction API — under `/`, `/trip`,
`/data`, `/models`, `/pipeline`, `/about`. Everything under
`src/api/static/` ships in the Docker image (no new Python runtime
dependencies).

## Regenerating the site data

Every number on the site comes from repo artifacts, never typed by hand.
Regenerate after retraining, re-evaluating, or re-running the quality gate:

```bash
uv run python -m scripts.build_site_data
```

This needs training-time dependencies (pandas, lightgbm, joblib) that are
**not** in the slim Docker image — the script itself is never shipped, only
the small JSON files it writes to `src/api/static/data/` (committed to
git, ~0.8MB total). It reads:

- `reports/*.json` (train/test metrics, quality gate, model registry, cleaning/segments/validation reports)
- `params.yaml`, `dvc.yaml`
- `data/processed/feature_spec.json`, `data/processed/{train,val,test}.parquet`
- `models/<best_model>.joblib` (for feature importance and the scatter/residual samples)

It prints a summary of what it wrote (filename + size) and exits non-zero
on failure. It's deterministic (fixed random seeds) and never embeds
vehicle identifiers or raw dataset rows beyond the 6 presets, a handful of
example trips, and a 2000-point (actual, predicted) scatter sample.

## Fonts

Bricolage Grotesque, Courier Prime, and IBM Plex Mono — all SIL Open Font
License 1.1 — self-hosted as `.woff2` under `src/api/static/fonts/`
(regular + bold weights only, Latin subset). Extracted once from the
`@fontsource/*` npm packages into a scratch directory outside the repo:

```bash
mkdir /tmp/font-extract && cd /tmp/font-extract
npm pack @fontsource/bricolage-grotesque @fontsource/courier-prime @fontsource/ibm-plex-mono
# extract each .tgz, copy files/*-latin-{400,700}-normal.woff2 into
# src/api/static/fonts/, renamed <family>-<weight>.woff2
```

No CDN, no runtime font fetch — if npm isn't available, fall back to a
system font stack (`Georgia`/serif for headings, `Courier New`/monospace
for ticket text and numbers) and note it; this repo did have npm
available, so the real fonts are checked in.

## Testing

- `tests/test_site.py` — pytest, no network: every page route returns 200,
  every static asset a page links to exists on disk, every data JSON file
  parses with the keys pages expect, presets/trip rows have all 25
  features and pass `PredictionRequest` validation, the numbers shown on
  `/models` match `reports/test_metrics.json`, and the JavaScript
  time-feature formulas (`static/js/time_features.js`) are checked against
  a golden 24-hour table (built by `build_site_data.py`, mirroring
  `src/features/time_features.py` exactly) by running the JS under Node
  directly from the test (GitHub-hosted runners ship Node by default).
- Browser QA: a one-off Playwright (Python) script visited every page at
  1440×900 and 390×844, clicked through presets, ran a prediction, and
  exercised the trip page, failing on any console error or failed
  request. Not checked into the repo as an automated test (no new runtime
  dependency), but the exact script and its findings are documented in
  the PR description.

## Known simplifications

- The design covers the full spec (ticket stub, ruler, route-line nav,
  model passport, pipeline stations) but some charts favor clarity over
  decoration given the time budget - e.g. the heatmap only has simple
  binned-median tooltips via the data-table toggle, not hover tooltips.
- Only two routes exist in this dataset (425, 534) - the Counter page says
  so directly rather than following a literal "top 20 routes" framing that
  doesn't apply here.
