# Build the W&B dashboard in ten minutes

Use the workspace for `anikaub20-rv-college/delhi-bus-delay`. The panel fields below were read from the W&B API on 28 November 2026. The replay submitted one monitoring run with notifications off. [Open the verified drift replay](https://forge.coreweave.com/wandb/anikaub20-rv-college/delhi-bus-delay/runs/4t0vffa3)

1. **0:00–2:00 · Runs table.** Open the project → **Workspace** → expand the runs table. Use the column controls to show run name, job type, `test_mae_best_model` and `val_mae_best_model`. Sort `test_mae_best_model` ascending and filter to rows with that value. The evaluator writes that test metric to `final-evaluation-test`. Evaluation runs `ip0m9wom` and `s6p8q79d` both have these fields.
2. **2:00–4:00 · Model comparison.** In Workspace, select **Add panels** → **Bar plot** and choose `final-evaluation-test` (run `ip0m9wom`). Use the following summary fields as separate metrics. If the editor accepts only one, use a custom chart that reshapes the summary fields to model and MAE, or present the existing Models page comparison. The seven values come from one evaluation run. [Evaluation run](https://forge.coreweave.com/wandb/anikaub20-rv-college/delhi-bus-delay/runs/ip0m9wom)
3. **4:00–6:00 · Sweep.** Open **Sweeps** → `7rcgds9b` → **Parameter importance**. Select `val_mae` as the objective if the panel prompts for a metric. Show **Parallel coordinates** alongside it to explain the relationship among settings. The sweep minimizes validation MAE. [Verified sweep](https://forge.coreweave.com/wandb/anikaub20-rv-college/delhi-bus-delay/sweeps/7rcgds9b)
4. **6:00–9:00 · Drift replay.** Select `drift-demo-simulated` (run `4t0vffa3`, job type `monitoring`). Add two **Line plot** panels. Set X to `window_idx` and select the Y fields from the table below. Keep the clean control visible and label the perturbations **SIMULATED**. The run logged no custom charts. Open `status_table` to see all 40 scenario windows, or open `drift_report_html` for the rendered replay. [Monitoring run](https://forge.coreweave.com/wandb/anikaub20-rv-college/delhi-bus-delay/runs/4t0vffa3)
5. **9:00–10:00 · Registry.** In the app navigation, select **Registry**, open the model registry and search for the `delhi-bus-delay-model` collection. Open **Versions** → `v0` and read its aliases and linked source artifact. The API returned `candidate`, `staging`, `latest` and `production` on collection `v0`, linked to `delhi-lightgbm_l1:v2`. [Verified ledger](../reports/version_ledger.json)

## Model comparison fields

- `test_global_median_mae`
- `test_typical_baseline_mae`
- `test_recent_buses_baseline_mae`
- `test_ridge_mae`
- `test_random_forest_mae`
- `test_lightgbm_l2_mae`
- `test_lightgbm_l1_mae`

## Drift panel fields

| MAE panel | Maximum PSI panel |
| --- | --- |
| `clean_test/mae` | `clean_test/max_psi` |
| `gps_outage/mae` | `gps_outage/max_psi` |
| `gradual_drift/mae` | `gradual_drift/max_psi` |
| `peak_traffic_surge/mae` | `peak_traffic_surge/max_psi` |
| `road_works/mae` | `road_works/max_psi` |

The summary fields are `n_alert` and `retrain_recommended`. The `status_table` columns are `scenario`, `window`, `n`, `mae`, `mae_degradation_pct`, `max_psi`, `status`. W&B also stored the table as the run artifact `run-4t0vffa3-status_table:v0`.

## Exact monitoring history keys

<details>
<summary>Show all 141 keys from the run history</summary>

```text
clean_test/mae
clean_test/mae_degradation_pct
clean_test/max_psi
clean_test/psi/direction
clean_test/psi/elapsed_trip_s
clean_test/psi/from_node
clean_test/psi/has_prev_bus
clean_test/psi/headway_s
clean_test/psi/hour
clean_test/psi/hour_cos
clean_test/psi/hour_sin
clean_test/psi/is_peak_am
clean_test/psi/is_peak_pm
clean_test/psi/minute_of_day
clean_test/psi/n_buses_recent
clean_test/psi/nodes_spanned
clean_test/psi/own_prev_segment_ratio
clean_test/psi/own_prev_segment_tt_s
clean_test/psi/prev3_median_tt_s
clean_test/psi/prev_bus_tt_s
clean_test/psi/relative_weekday
clean_test/psi/route
clean_test/psi/route_position
clean_test/psi/segment_id
clean_test/psi/stops_into_trip
clean_test/psi/te_vehicle
clean_test/psi/typical_segment_hour_s
clean_test/psi/typical_segment_s
gps_outage/mae
gps_outage/mae_degradation_pct
gps_outage/max_psi
gps_outage/psi/direction
gps_outage/psi/elapsed_trip_s
gps_outage/psi/from_node
gps_outage/psi/has_prev_bus
gps_outage/psi/headway_s
gps_outage/psi/hour
gps_outage/psi/hour_cos
gps_outage/psi/hour_sin
gps_outage/psi/is_peak_am
gps_outage/psi/is_peak_pm
gps_outage/psi/minute_of_day
gps_outage/psi/n_buses_recent
gps_outage/psi/nodes_spanned
gps_outage/psi/own_prev_segment_ratio
gps_outage/psi/own_prev_segment_tt_s
gps_outage/psi/prev3_median_tt_s
gps_outage/psi/prev_bus_tt_s
gps_outage/psi/relative_weekday
gps_outage/psi/route
gps_outage/psi/route_position
gps_outage/psi/segment_id
gps_outage/psi/stops_into_trip
gps_outage/psi/te_vehicle
gps_outage/psi/typical_segment_hour_s
gps_outage/psi/typical_segment_s
gradual_drift/mae
gradual_drift/mae_degradation_pct
gradual_drift/max_psi
gradual_drift/psi/direction
gradual_drift/psi/elapsed_trip_s
gradual_drift/psi/from_node
gradual_drift/psi/has_prev_bus
gradual_drift/psi/headway_s
gradual_drift/psi/hour
gradual_drift/psi/hour_cos
gradual_drift/psi/hour_sin
gradual_drift/psi/is_peak_am
gradual_drift/psi/is_peak_pm
gradual_drift/psi/minute_of_day
gradual_drift/psi/n_buses_recent
gradual_drift/psi/nodes_spanned
gradual_drift/psi/own_prev_segment_ratio
gradual_drift/psi/own_prev_segment_tt_s
gradual_drift/psi/prev3_median_tt_s
gradual_drift/psi/prev_bus_tt_s
gradual_drift/psi/relative_weekday
gradual_drift/psi/route
gradual_drift/psi/route_position
gradual_drift/psi/segment_id
gradual_drift/psi/stops_into_trip
gradual_drift/psi/te_vehicle
gradual_drift/psi/typical_segment_hour_s
gradual_drift/psi/typical_segment_s
peak_traffic_surge/mae
peak_traffic_surge/mae_degradation_pct
peak_traffic_surge/max_psi
peak_traffic_surge/psi/direction
peak_traffic_surge/psi/elapsed_trip_s
peak_traffic_surge/psi/from_node
peak_traffic_surge/psi/has_prev_bus
peak_traffic_surge/psi/headway_s
peak_traffic_surge/psi/hour
peak_traffic_surge/psi/hour_cos
peak_traffic_surge/psi/hour_sin
peak_traffic_surge/psi/is_peak_am
peak_traffic_surge/psi/is_peak_pm
peak_traffic_surge/psi/minute_of_day
peak_traffic_surge/psi/n_buses_recent
peak_traffic_surge/psi/nodes_spanned
peak_traffic_surge/psi/own_prev_segment_ratio
peak_traffic_surge/psi/own_prev_segment_tt_s
peak_traffic_surge/psi/prev3_median_tt_s
peak_traffic_surge/psi/prev_bus_tt_s
peak_traffic_surge/psi/relative_weekday
peak_traffic_surge/psi/route
peak_traffic_surge/psi/route_position
peak_traffic_surge/psi/segment_id
peak_traffic_surge/psi/stops_into_trip
peak_traffic_surge/psi/te_vehicle
peak_traffic_surge/psi/typical_segment_hour_s
peak_traffic_surge/psi/typical_segment_s
road_works/mae
road_works/mae_degradation_pct
road_works/max_psi
road_works/psi/direction
road_works/psi/elapsed_trip_s
road_works/psi/from_node
road_works/psi/has_prev_bus
road_works/psi/headway_s
road_works/psi/hour
road_works/psi/hour_cos
road_works/psi/hour_sin
road_works/psi/is_peak_am
road_works/psi/is_peak_pm
road_works/psi/minute_of_day
road_works/psi/n_buses_recent
road_works/psi/nodes_spanned
road_works/psi/own_prev_segment_ratio
road_works/psi/own_prev_segment_tt_s
road_works/psi/prev3_median_tt_s
road_works/psi/prev_bus_tt_s
road_works/psi/relative_weekday
road_works/psi/route
road_works/psi/route_position
road_works/psi/segment_id
road_works/psi/stops_into_trip
road_works/psi/te_vehicle
road_works/psi/typical_segment_hour_s
road_works/psi/typical_segment_s
window_idx
```

</details>
