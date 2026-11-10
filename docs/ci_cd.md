# CI/CD

[![CI](https://github.com/niksi-nova/mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/niksi-nova/mlops/actions/workflows/ci.yml)
[![CD](https://github.com/niksi-nova/mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/niksi-nova/mlops/actions/workflows/cd.yml)

## CI — `.github/workflows/ci.yml`

Triggers: push to `main` or any `phase2-*` branch, and pull requests into `main`.

Three jobs, `quality-gate` and `docker-smoke` both depend on `tests`:

1. **tests** — `uv sync --frozen`, configures the DVC remote from
   `DAGSHUB_USER`/`DAGSHUB_TOKEN` (see below), pulls only the DVC-tracked
   paths the suite needs, then runs `uv run pytest -q`.
   - If the DVC pull fails (bad/missing secrets, DagsHub outage, etc.), the
     job doesn't fail outright — it falls back to `pytest -q -m "not needs_data"`,
     skipping the two tests that need real pulled data, and posts a
     `::warning::`. Trade-off: full coverage for CI resilience; a persistent
     pull failure needs a human to look at the DagsHub side, not endless
     red CI runs.
   - `WANDB_API_KEY`/`WANDB_BASE_URL` are passed through as env vars
     whenever those secrets exist, which lets `tests/test_api.py`'s
     `@pytest.mark.integration` test load the real `production` model;
     otherwise it auto-skips (same behavior as running locally without
     credentials).

2. **quality-gate** — pulls the same minimal DVC targets, runs
   `uv run python -m src.models.quality_gate` (job fails on its non-zero
   exit code automatically), uploads `reports/quality_gate.json` as a
   build artifact, then runs `uv run dvc status`.
   - **Caveat on the `dvc status` check**: on this repo, plain `dvc status`
     is *never* fully empty, even right after a clean pull/push. Several
     report JSONs (`quality_gate.json`, `model_registry.json`, etc.) embed
     a fresh timestamp on every run, and `git checkout` resets source file
     mtimes, so DVC always reports "modified" deps/outs noise that has
     nothing to do with whether the pipeline is actually stale. Failing
     the job on *any* `dvc status` output would make this check permanently
     red. Instead, the job only fails when the output mentions
     `not in cache` or `missing` — i.e. DVC genuinely can't find data it
     expects, which is the thing worth blocking on.

3. **docker-smoke** — pulls `feature_spec.json` and `models/lightgbm_l1.joblib`,
   builds the image, runs it in local-file mode
   (`MODEL_PATH=/app/model/lightgbm_l1.joblib`, no W&B needed), polls
   `docker inspect` until the healthcheck reports `healthy` (30×2s timeout),
   calls `GET /health` and `POST /predict` with the example row, and fails
   if the response is missing `predicted_travel_time_s` or it falls outside
   `[0, 3600]`. The container is always stopped and removed (`if: always()`),
   and its logs are always printed for debugging.

No lint step was added: `ruff` isn't a project dependency, and the task's
own rule was not to add new dependencies for this.

### Which tests need pulled data?

Only two, both in `tests/test_registry.py`, marked `@pytest.mark.needs_data`:
`test_load_model_staging_succeeds` and `test_prediction_equivalence_and_wrapper_contract`.
They call the real `load_model("staging")` against the real
`reports/model_registry.json`, which needs `models/lightgbm_l1.joblib` and
`data/processed/feature_spec.json`/`test.parquet` on disk. Every other test
uses `tmp_path`/`monkeypatch` fixtures with synthetic data and never touches
DVC-tracked files.

### Why a *targeted* `dvc pull`, not a plain one

A plain `dvc pull` would fetch the entire `models/` directory tracked by
the `train` stage — including `models/random_forest.joblib`, which alone is
**~610MB** and isn't used by anything in `tests/`, `quality_gate.py`, or the
API (the production model is `lightgbm_l1`). The full pull is ~766MB; the
targeted one (`$DVC_PULL_TARGETS` in the workflow) is ~80MB. DVC supports
pulling individual files out of a directory-tracked output, so this is a
safe, supported operation, not a workaround.

## CD — `.github/workflows/cd.yml`

Triggers: push to `main`, or manual `workflow_dispatch`.

Builds the (slimmed, `requirements-serve.txt`-based) image, logs into
`ghcr.io` with the built-in `GITHUB_TOKEN` (no extra secret needed for
this), and pushes two tags to `ghcr.io/niksi-nova/mlops`:

- the short commit SHA (`type=sha,format=short`)
- `latest`

Each image carries OCI labels (`org.opencontainers.image.source`,
`.revision`, `.version`). The push's resulting digest is written to the
job summary (`$GITHUB_STEP_SUMMARY`), visible on the workflow run page
without digging through logs. Build layers are cached via GitHub Actions
cache (`cache-from`/`cache-to: type=gha`).

This workflow does not deploy anywhere — it only builds and publishes the
image. A hosting/deploy step is intentionally left for later.

## Secrets the workflows expect

| Secret | Used by | Required? |
|---|---|---|
| `DAGSHUB_USER` | CI (`tests`, `quality-gate`, `docker-smoke`) | Needed for `dvc pull` to succeed; CI degrades gracefully in `tests` if missing (see fallback above), but `quality-gate`/`docker-smoke` will fail without it. |
| `DAGSHUB_TOKEN` | same as above | Same as above. |
| `WANDB_API_KEY` | CI `tests` job | Optional — only needed for the `integration`-marked test to run instead of skip. |
| `WANDB_BASE_URL` | CI `tests` job | Optional, same as above, only matters for a non-default W&B host. |
| `GITHUB_TOKEN` | CD | Automatic — GitHub provides this for every workflow run, nothing to configure. |

`DAGSHUB_USER`/`DAGSHUB_TOKEN` map directly to this repo's `.dvc/config`
remote (`origin`, an S3-compatible endpoint at
`https://dagshub.com/niksi-nova/mlops.s3`): CI runs
`dvc remote modify --local origin access_key_id "$DAGSHUB_USER"` and
`... secret_access_key "$DAGSHUB_TOKEN"`, matching exactly how this
machine's own `.dvc/config.local` authenticates.

## Running a workflow manually

CD supports `workflow_dispatch`: on GitHub, go to **Actions → CD → Run workflow**,
pick the branch (normally `main`), and run. CI has no manual trigger — it
only runs on push/PR as configured above.
