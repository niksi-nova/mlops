# AI-assisted Public Transport Information System: Transit Delay Prediction

## Problem statement

Bus delays on the Toronto Transit Commission (TTC) network are frequent and hard for riders to plan around. This project builds an end-to-end MLOps pipeline that predicts the **delay duration in minutes** for a reported TTC bus incident, using features such as the route, time of day and day of week, location, incident type and weather conditions. Accurate delay estimates can power rider-facing information systems and help operators prioritise responses.

## MLOps tooling

| Tool | Role in this project | Why it was chosen |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python/dependency manager, lockfile, task runner | Single fast tool replacing pip+venv+poetry; reproducible installs via `uv.lock`; `uv run` keeps every script pinned to the project environment. |
| [DVC](https://dvc.org/) | Data & pipeline versioning | The raw TTC XLSX/CSV files are large and change monthly; DVC tracks them outside git (remote: DagsHub) and will later version the ingest -> feature -> train pipeline as a DAG (`dvc.yaml`) so stages re-run only when their inputs change. |
| [MLflow](https://mlflow.org/) | Experiment tracking & model registry | Lets us log params/metrics/artifacts for every LightGBM/XGBoost run and compare them, and registers the best model for later serving. |
| [pandera](https://pandera.readthedocs.io/) | Data schema validation | The raw files are hand-edited open data with inconsistent column names across years (see `docs/data_inventory.md`) - pandera schemas will catch drift/bad rows before they reach the feature pipeline. |
| [pandas / numpy](https://pandas.pydata.org/) | Data loading & transformation | Standard tabular data toolkit for cleaning the yearly delay files and building features. |
| [LightGBM](https://lightgbm.readthedocs.io/) / [XGBoost](https://xgboost.readthedocs.io/) | Gradient-boosted tree models | Strong baseline performance on tabular data with mixed categorical/numeric features (route, incident type, time) and fast training/inference. |
| [scikit-learn](https://scikit-learn.org/) | Preprocessing, splitting, metrics, baselines | Standard, well-tested building blocks (encoders, pipelines, cross-validation) that compose with the boosting libraries above. |
| [holidays](https://pypi.org/project/holidays/) | Canadian/Ontario holiday calendar | Delay patterns shift on holidays and long weekends; used as a time-based feature. |
| [requests](https://requests.readthedocs.io/) | HTTP client | Used by `src/data/ingest.py` to call the City of Toronto CKAN API and download resources by name. |
| [openpyxl](https://openpyxl.readthedocs.io/) | Excel engine for pandas | The City of Toronto publishes 2014-2024 delay data as one `.xlsx` workbook per year, with one sheet per month. |
| [matplotlib / seaborn](https://matplotlib.org/) | Plotting | Exploratory analysis and figures saved under `reports/figures/`. |
| [Jupyter](https://jupyter.org/) | Interactive EDA | Notebooks under `notebooks/` for exploration before code is promoted into `src/`. |
| [PyYAML](https://pyyaml.org/) | Config loading | `params.yaml` centralises pipeline parameters (CKAN endpoint, package id, years to pull) so stages don't hardcode them. |

## Data ingestion

- `params.yaml` holds the CKAN base URL, the Open Data package id (`ttc-bus-delay-data`) and the list of historical years to pull.
- `src/data/ingest.py` calls `package_show` on that package and resolves each file it needs **by resource name** (never a hardcoded URL/id, since CKAN resource ids can change on re-publish):
  - yearly workbooks -> `data/raw/historical/ttc-bus-delay-data-<year>.xlsx`
  - the code lookup table -> `data/raw/reference/Code Descriptions.csv`
  - with `--include-recent`, the rolling 2025-onward extract -> `data/raw/recent/TTC Bus Delay Data since 2025.csv`
  - already-downloaded files are skipped; downloads are streamed to a `.part` file and renamed on success.
- `scripts/inspect_raw.py` reads every downloaded file and writes a sheet/row/column/sample summary to `docs/data_inventory.md`, including a check for column-name drift across years (there is drift: 2018-2019 use `Min Delay`/`Min Gap`, 2020 shortens these to `Delay`/`Gap`, and 2021 onward rename `Report Date` to `Date`).

```bash
uv run python -m src.data.ingest              # historical years + reference only
uv run python -m src.data.ingest --include-recent  # also pulls the 2025+ rolling extract
uv run python scripts/inspect_raw.py
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```
