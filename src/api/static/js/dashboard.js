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
