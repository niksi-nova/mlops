/* Data page: all charts precomputed in data_overview.json. */

async function init() {
  const meta = await fetchJSON(`${DATA_BASE}/meta.json`);
  document.getElementById("site-footer").textContent = `snapshot built ${meta.snapshot_built} from commit ${meta.git_commit}`;

  const d = await fetchJSON(`${DATA_BASE}/data_overview.json`);

  const c = d.cleaning;
  document.getElementById("cleaning-summary").textContent =
    `Started from ${c.raw_rows.toLocaleString()} raw stop-event rows. Removed ${c.trips_removed.total_distinct_trips_removed} bad trips ` +
    `(${c.trips_removed.time_backward} with time going backward, ${c.trips_removed.too_few_nodes} too short), ` +
    `${c.rows_removed_with_bad_trips} rows with them, and fixed ${c.midnight_crossing_trips_fixed} midnight-crossing trips. ` +
    `Final: ${c.final_rows.toLocaleString()} rows across ${c.final_trips.toLocaleString()} trips. ` +
    `Validation status: ${d.validation.status} (${d.validation.row_count.toLocaleString()} rows, ${d.validation.checks_run.length} checks run).`;

  histogram(document.getElementById("hist-chart"), d.histogram.log1p_counts, d.histogram.log1p_bin_edges, {
    xLabel: "log(1+travel_time_s)", tableTarget: document.getElementById("hist-table"),
  });

  const hourlyPoints = Object.entries(d.by_hour).map(([h, v]) => ({ x: Number(h), y: v.median, lo: v.q25, hi: v.q75 })).sort((a, b) => a.x - b.x);
  lineBandChart(document.getElementById("hourly-chart"), hourlyPoints, {
    xLabel: "hour", yLabel: "median travel time (s)", color: "#1F6B3A", tableTarget: document.getElementById("hourly-table"),
  });

  const weekdays = Object.keys(d.heatmap_hour_weekday).sort((a, b) => Number(a) - Number(b));
  const hours = d.heatmap_hours.map(String);
  const matrix = weekdays.map((wd) => hours.map((h) => d.heatmap_hour_weekday[wd][h] ?? null));
  heatmap(document.getElementById("heatmap-chart"), matrix, hours, weekdays, { tableTarget: document.getElementById("heatmap-table") });

  const headwayPoints = d.headway_vs_traveltime.x_headway_s.map((x, i) => ({ x, y: d.headway_vs_traveltime.y_median_tt_s[i] }));
  lineBandChart(document.getElementById("headway-chart"), headwayPoints, {
    xLabel: "headway (s)", yLabel: "median travel time (s)", color: "#1F6FEB", tableTarget: document.getElementById("headway-table"),
  });

  const prevOwnPoints = d.prev_vs_own.x_prev_tt_s.map((x, i) => ({ x, y: d.prev_vs_own.y_median_tt_s[i] }));
  lineBandChart(document.getElementById("prevown-chart"), prevOwnPoints, {
    xLabel: "previous bus travel time (s)", yLabel: "median travel time (s)", color: "#D7263D", tableTarget: document.getElementById("prevown-table"),
  });

  barChart(
    document.getElementById("split-chart"),
    [{ label: "rows", color: "#1B1A17", values: [
      { x: "train", y: d.splits.train }, { x: "val", y: d.splits.val }, { x: "test", y: d.splits.test },
    ] }],
    { yLabel: "rows" }
  );
}

init();
