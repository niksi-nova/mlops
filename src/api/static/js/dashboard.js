"use strict";

const byId = id => document.getElementById(id);
let evidence;
const repo = "https://github.com/niksi-nova/mlops";

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function evidenceLink(path, label = path) {
  const link = element("a", label);
  link.href = repo + (path.endsWith("/") ? "/tree/main/" : "/blob/main/") + path;
  return link;
}

function records(container, headers, rows) {
  const table = element("table");
  const head = table.createTHead().insertRow();
  for (const label of headers) {
    const th = element("th", label);
    th.scope = "col";
    head.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    row.forEach((value, i) => {
      const td = tr.insertCell();
      td.dataset.label = headers[i];
      td.append(value instanceof Node ? value : String(value ?? "Unrecorded"));
    });
  }
  container.replaceChildren(table);
}

function stages() {
  for (const stage of evidence.stages) {
    const li = element("li");
    li.append(evidenceLink(stage.evidence, stage.name), element("small", stage.tool),
      element("span", stage.status, "stage-result"));
    byId("stages").append(li);
  }
}

async function dashboard() {
  try {
    evidence = await fetchJSON(DATA_BASE + "/dashboard.json");
    stages();
    performancePanel();
    driftPanel();
    versionPanel();
    governancePanel();
  } catch {
    byId("snapshot-error").textContent = "The dashboard evidence could not be loaded. Refresh to try again.";
  }
}

dashboard();

function svgNode(tag, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function performancePanel() {
  const test = evidence.test_metrics;
  const models = Object.entries(test.models).map(([name, model]) => ({
    name, val: evidence.train_metrics.models[name].val_mae, test: model.mae
  }));
  const scale = Math.max(...models.flatMap(model => [model.val, model.test]));
  const list = byId("performance-bars");
  for (const model of models) {
    const row = element("div", undefined, "bar-row");
    const label = element("span", model.name.replaceAll("_", " "));
    const svg = svgNode("svg", {viewBox: "0 0 100 14", preserveAspectRatio: "none", "aria-hidden": "true"});
    for (const [value, y, color] of [[model.val, 1, "var(--ink)"], [model.test, 8, "var(--dtc-green)"]]) {
      svg.append(svgNode("rect", {x: 0, y, width: 100 * value / scale, height: 5, fill: color}));
    }
    row.append(label, svg, element("span", model.val.toFixed(2) + " / " + model.test.toFixed(2), "num"));
    list.append(row);
  }
  const checks = Object.entries(evidence.quality_gate.checks);
  byId("gate-title").textContent = "Quality gate · " + checks.filter(([, c]) => c.passed).length + " of " + checks.length + " passed";
  records(byId("checks"), ["Check", "Measured", "Threshold", "Result"], checks.map(([name, c]) => [
    name.replaceAll("_", " "), Number(c.value.toFixed(2)), c.operator + " " + c.threshold,
    chip(c.passed ? "PASS" : "FAIL")
  ]));
}

function chip(label) {
  return element("span", label, "chip " + label.toLowerCase());
}

function trendChart(container, label, key, scenario, control) {
  const width = 360, height = 136, left = 44, right = 15, top = 14, bottom = 25;
  const chart = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label + " by window, values in the table below"});
  const max = Math.max(...scenario.windows.concat(control.windows).map(w => w[key]), 0.01);
  const y = value => height - bottom - value / max * (height - top - bottom);
  const x = i => left + i / Math.max(1, scenario.windows.length - 1) * (width - left - right);
  function tick(value, xx, yy, anchor) {
    const text = svgNode("text", {x: xx, y: yy, "text-anchor": anchor, "font-size": 12, fill: "var(--ink)"});
    text.textContent = value;
    chart.append(text);
  }
  for (const value of [0, max / 2, max]) {
    chart.append(svgNode("line", {x1: left, x2: width-right, y1: y(value), y2: y(value), stroke: "var(--ink)", opacity: .25}));
    tick(value.toFixed(key === "mae" ? 1 : 2), left - 5, y(value) + 4, "end");
  }
  scenario.windows.forEach((w, i) => tick(w.window, x(i), height - 5, "middle"));
  for (const [series, color, dashed] of [[scenario, "var(--dtc-green)", false], [control, "var(--ink)", true]]) {
    const points = series.windows.map((w, i) => x(i) + "," + y(w[key])).join(" ");
    chart.append(svgNode("polyline", {points, fill: "none", stroke: color, "stroke-width": 2.3, "stroke-dasharray": dashed ? "4 4" : "none"}));
  }
  container.replaceChildren(element("figcaption", label), chart);
}

function driftPanel() {
  const report = evidence.drift_report;
  const select = byId("scenario");
  const control = report.scenarios.find(s => !s.induced);
  for (const s of report.scenarios) select.add(new Option(s.name.replaceAll("_", " "), s.name));
  select.value = (report.scenarios.find(s => s.induced) || control).name;
  byId("disclaimer").textContent = report.disclaimer;
  function render() {
    const s = report.scenarios.find(item => item.name === select.value);
    const marker = chip(s.induced ? "SIMULATED" : "CONTROL");
    if (s.induced) marker.classList.add("sim");
    byId("drift-status").replaceChildren(marker, " ", chip(s.status), "  " + s.description);
    byId("control-status").textContent = "Clean control · " + control.status + "   |   First alert · " + (s.first_alert_window || "None");
    trendChart(byId("mae-chart"), "Mean absolute error (seconds)", "mae", s, control);
    trendChart(byId("psi-chart"), "Maximum eligible feature PSI", "max_psi", s, control);
    records(byId("windows"), ["Window", "Days", "MAE (s)", "Control (s)", "Max PSI", "Control PSI", "Status"],
      s.windows.map((w, i) => [w.window, w.days, w.mae.toFixed(2), control.windows[i].mae.toFixed(2),
        w.max_psi.toFixed(3), control.windows[i].max_psi.toFixed(3), chip(w.status)]));
    records(byId("features"), ["Feature", "PSI", "Peak window"], s.top_drifted_features.map(f =>
      [f.feature, f.psi.toFixed(3), f.peak_window]));
  }
  select.addEventListener("change", render);
  render();
}

function versionPanel() {
  const ledger = evidence.version_ledger;
  byId("version-source").textContent = ledger.source === "wandb"
    ? "W&B snapshot · " + ledger.generated_at
    : "Local ledger snapshot · " + ledger.generated_at + ". Offline aliases and digests are recorded examples and have no live W&B verification.";
  records(byId("layers"), ["Layer", "Recorded identifiers"], Object.entries(ledger.layers).map(([key, values]) =>
    [key, Object.entries(values).map(([field, value]) => field.replaceAll("_", " ") + " · " + (Array.isArray(value) ? value.join(", ") : value)).join(" / ")]));
  records(byId("versions"), ["Artifact", "Version", "Aliases", "Digest"], ledger.model_versions.map(v =>
    [v.artifact, v.version, v.aliases.join(", ") || "None", v.digest]));
  byId("collection").textContent = ledger.registry.collection;
  records(byId("aliases"), ["Collection version", "Aliases", "Source artifact"], ledger.registry.versions.map(v =>
    [v.version, v.aliases.join(", ") || "None", v.source]));
}

function controlEvidence(entry) {
  if (entry.id === "G09") {
    const link = element("a", entry.evidence);
    link.href = repo + "/settings/branches";
    return link;
  }
  const span = element("span");
  for (const [i, path] of entry.evidence.split(", ").entries()) {
    if (i) span.append(", ");
    const file = path.startsWith(".gitignore ") ? ".gitignore" : path;
    span.append(evidenceLink(file, path));
  }
  return span;
}

function governancePanel() {
  const gov = evidence.governance_status;
  byId("governance-counts").textContent = Object.entries(gov.summary)
    .map(([status, count]) => count + " " + status.toUpperCase()).join(" · ");
  records(byId("controls"), ["Control", "Risk", "Status", "Evidence"], gov.controls.map(c =>
    [c.id + " · " + c.name, c.risk, chip(c.status), controlEvidence(c)]));
}
