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
  } catch {
    byId("snapshot-error").textContent = "The dashboard evidence could not be loaded. Refresh to try again.";
  }
}

dashboard();
