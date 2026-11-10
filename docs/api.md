# Delhi Bus Delay Prediction API

FastAPI inference service for the Delhi bus segment travel-time model
(`lightgbm_l1`), served from the W&B model registry
(`delhi-bus-delay-model`, workspace `anikaub20-rv-college`, project
`delhi-bus-delay`).

## Running it

```bash
uv run uvicorn src.api.main:app --reload
```

The app loads the model once at startup via `src.models.registry.load_model()`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ALIAS` | `production` | Registry alias to load when `MODEL_PATH` is not set. |
| `MODEL_PATH` | unset | Path to a local `.joblib` file. If set, loads from disk directly and never contacts W&B — useful for Docker images that bake the model in. |
| `WANDB_API_KEY` | unset | Read by the `wandb` client when resolving a registry alias remotely. |
| `WANDB_BASE_URL` | unset | Same as above, for a non-default W&B host. |

## Endpoints

- `GET /health` — `{"status": "ok", "model_loaded": true}`, or HTTP 503 with `{"status": "degraded", "model_loaded": false}` if the model isn't loaded.
- `GET /model-info` — registry alias, version, model name, git commit, and the 25 feature columns in the exact order the model expects. 503 if not loaded.
- `POST /predict` — one segment's 25 features in, one prediction out.
- `POST /predict/batch` — up to 500 requests in one call.

Interactive docs: `http://127.0.0.1:8000/docs`.

## curl examples

### `/predict`

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"route": "425", "direction": "down", "segment_id": "425_down_26", "relative_weekday": 6, "from_node": 26, "nodes_spanned": 1, "route_position": 0.6341463414634146, "stops_into_trip": 1, "elapsed_trip_s": 0.0, "hour": 8, "minute_of_day": 484, "hour_sin": 0.8660254037844387, "hour_cos": -0.49999999999999983, "is_peak_am": 1, "is_peak_pm": 0, "prev_bus_tt_s": 30.0, "prev3_median_tt_s": 30.0, "headway_s": 52.0, "n_buses_recent": 3, "has_prev_bus": 1, "own_prev_segment_tt_s": 40.0, "own_prev_segment_ratio": 1.0, "typical_segment_hour_s": 40.0, "typical_segment_s": 40.0, "te_vehicle": 144.26669174404856}'
```

Response:

```json
{"predicted_travel_time_s": 30.765926119190304, "model_alias": "production", "model_version": "v1", "request_id": "..."}
```

### `/predict/batch`

```bash
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"route": "425", "direction": "down", "segment_id": "425_down_26", "relative_weekday": 6, "from_node": 26, "nodes_spanned": 1, "route_position": 0.6341463414634146, "stops_into_trip": 1, "elapsed_trip_s": 0.0, "hour": 8, "minute_of_day": 484, "hour_sin": 0.8660254037844387, "hour_cos": -0.49999999999999983, "is_peak_am": 1, "is_peak_pm": 0, "prev_bus_tt_s": 30.0, "prev3_median_tt_s": 30.0, "headway_s": 52.0, "n_buses_recent": 3, "has_prev_bus": 1, "own_prev_segment_tt_s": 40.0, "own_prev_segment_ratio": 1.0, "typical_segment_hour_s": 40.0, "typical_segment_s": 40.0, "te_vehicle": 144.26669174404856}]}'
```

## Honest note on real-time features

The API takes the **25 already-computed features** as input — it does not
compute any of them itself. In particular, the real-time features
(`prev_bus_tt_s`, `prev3_median_tt_s`, `headway_s`, `n_buses_recent`,
`has_prev_bus`, `own_prev_segment_tt_s`, `own_prev_segment_ratio`) must be
supplied by the caller. Computing them from a live bus feed — tracking
which buses are currently running, their recent segment times, and how long
ago each one passed a given stop — is out of scope for this project. A
real deployment would need a separate service maintaining that live state
and assembling the feature row before calling this API.
