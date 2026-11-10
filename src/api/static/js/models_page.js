/* Models page: comparison, importance, scatter, residuals, slices. */

const GROUP_COLORS = { realtime: "#1F6FEB", historical: "#2E9E5B", time: "#F2B705", position: "#D7263D", other: "#1B1A17" };

async function init() {
  const meta = await fetchJSON(`${DATA_BASE}/meta.json`);
  document.getElementById("site-footer").textContent = `snapshot built ${meta.snapshot_built} from commit ${meta.git_commit}`;

  const m = await fetchJSON(`${DATA_BASE}/models.json`);

  document.getElementById("models-summary").textContent =
    `Best model: ${m.best_model}, beating the "${m.best_baseline}" baseline by ${m.improvement_vs_recent_pct}% ` +
    `and the typical-time baseline by ${m.improvement_vs_typical_pct}% (target was ${m.success_target_pct}%). ` +
    `Test MAE ${m.test_mae_best_model}s, generalization gap ${m.generalisation_gap_pct}% vs validation.`;

  barChart(
    document.getElementById("mae-chart"),
    [
      { label: "val MAE", color: "#1B1A17", values: m.comparison.map((c) => ({ x: c.model, y: c.val_mae })) },
      { label: "test MAE", color: "#E8590C", values: m.comparison.map((c) => ({ x: c.model, y: c.test_mae })) },
    ],
    { yLabel: "MAE (s)", width: 760, tableTarget: document.getElementById("mae-table") }
  );

  if (m.feature_importance.length) {
    const container = document.getElementById("importance-chart");
    const width = 560, rowH = 20, height = m.feature_importance.length * rowH + 20;
    const maxGain = Math.max(...m.feature_importance.map((f) => f.gain));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", "100%");
    const labelW = 160;
    m.feature_importance.forEach((f, i) => {
      const barW = ((width - labelW - 10) * f.gain) / maxGain;
      const y = i * rowH;
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", labelW - 6); label.setAttribute("y", y + 13); label.setAttribute("text-anchor", "end");
      label.setAttribute("font-size", "10"); label.setAttribute("font-family", "var(--font-ticket)");
      label.textContent = f.feature;
      svg.appendChild(label);
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", labelW); rect.setAttribute("y", y + 2);
      rect.setAttribute("width", Math.max(barW, 1)); rect.setAttribute("height", rowH - 5);
      rect.setAttribute("fill", GROUP_COLORS[f.group] || GROUP_COLORS.other);
      svg.appendChild(rect);
    });
    container.innerHTML = ""; container.appendChild(svg);
    fillDataTable(document.getElementById("importance-table"), ["feature", "gain", "group"], m.feature_importance.map((f) => [f.feature, f.gain, f.group]));
  } else {
    document.getElementById("importance-chart").textContent = "Feature importance unavailable (model artifact not present when the site data was built).";
  }

  if (m.scatter_sample.actual.length) {
    const points = m.scatter_sample.actual.map((a, i) => ({ x: a, y: m.scatter_sample.predicted[i] }));
    scatterCanvas(document.getElementById("scatter-canvas"), points, { tableTarget: document.getElementById("scatter-table") });
  }
  if (m.residual_histogram.counts.length) {
    histogram(document.getElementById("residual-chart"), m.residual_histogram.counts, m.residual_histogram.bin_edges, { xLabel: "residual (s)", color: "#E8590C" });
  }

  const hourSeries = m.mae_by_hour.filter((h) => h.n >= 100).sort((a, b) => a.hour - b.hour);
  barChart(
    document.getElementById("mae-hour-chart"),
    [{ label: "MAE", color: "#1F6B3A", values: hourSeries.map((h) => ({ x: h.hour, y: h.mae })) }],
    { yLabel: "MAE (s)", width: 760, tableTarget: document.getElementById("mae-hour-table") }
  );
}

init();
