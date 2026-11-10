/* Trip page: whole-trip batch prediction + bulk CSV/JSON runner. */

const MAX_BULK_ROWS = 500;
const FEATURE_KEYS = [
  "route", "direction", "segment_id", "relative_weekday", "from_node", "nodes_spanned",
  "route_position", "stops_into_trip", "elapsed_trip_s", "hour", "minute_of_day", "hour_sin",
  "hour_cos", "is_peak_am", "is_peak_pm", "prev_bus_tt_s", "prev3_median_tt_s", "headway_s",
  "n_buses_recent", "has_prev_bus", "own_prev_segment_tt_s", "own_prev_segment_ratio",
  "typical_segment_hour_s", "typical_segment_s", "te_vehicle",
];

let trips = [];

async function loadFooter() {
  const meta = await fetchJSON(`${DATA_BASE}/meta.json`);
  document.getElementById("site-footer").textContent = `snapshot built ${meta.snapshot_built} from commit ${meta.git_commit}`;
}

function populateTripSelect() {
  const sel = document.getElementById("trip-select");
  sel.innerHTML = "";
  trips.forEach((t, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${t.route} ${t.direction} — ${t.n_segments} segments (trip ${t.trip_id})`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => runTrip(trips[Number(sel.value)]));
}

async function runTrip(trip) {
  const rows = trip.segments.map(({ actual_travel_time_s, ...feats }) => feats);
  const res = await fetch("/predict/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requests: rows }),
  });
  if (!res.ok) {
    document.getElementById("trip-total-error").textContent = `batch prediction failed (HTTP ${res.status})`;
    return;
  }
  const body = await res.json();
  const predicted = body.predictions.map((p) => p.predicted_travel_time_s);
  const actual = trip.segments.map((s) => s.actual_travel_time_s);

  drawTripChart(predicted, actual);

  let cumPred = 0, cumActual = 0;
  const rowsOut = predicted.map((p, i) => {
    cumPred += p; cumActual += actual[i];
    return [i + 1, p.toFixed(1), actual[i].toFixed(1), cumPred.toFixed(1), cumActual.toFixed(1)];
  });
  const tableDiv = document.getElementById("trip-table");
  tableDiv.innerHTML = "";
  const tbl = document.createElement("table");
  tbl.innerHTML = "<thead><tr><th>#</th><th>predicted (s)</th><th>actual (s)</th><th>cum. predicted</th><th>cum. actual</th></tr></thead>";
  const tbody = document.createElement("tbody");
  rowsOut.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = r.map((c) => `<td>${c}</td>`).join("");
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
  tableDiv.appendChild(tbl);

  const totalErr = cumPred - cumActual;
  document.getElementById("trip-total-error").textContent =
    `Total predicted trip time: ${cumPred.toFixed(1)}s vs actual ${cumActual.toFixed(1)}s (${totalErr >= 0 ? "+" : ""}${totalErr.toFixed(1)}s).`;
}

function drawTripChart(predicted, actual) {
  const container = document.getElementById("trip-chart");
  const width = Math.max(500, predicted.length * 40);
  const height = 240;
  const margin = { top: 10, right: 10, bottom: 24, left: 40 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const maxV = Math.max(...predicted, ...actual) * 1.15;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "predicted vs actual travel time per segment");
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.setAttribute("transform", `translate(${margin.left},${margin.top})`);

  const barW = w / predicted.length;
  predicted.forEach((p, i) => {
    const x = i * barW;
    const predH = (p / maxV) * h;
    const actH = (actual[i] / maxV) * h;
    const predRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    predRect.setAttribute("x", x); predRect.setAttribute("y", h - predH);
    predRect.setAttribute("width", barW * 0.35); predRect.setAttribute("height", predH);
    predRect.setAttribute("fill", "#E8590C");
    g.appendChild(predRect);
    const actRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    actRect.setAttribute("x", x + barW * 0.4); actRect.setAttribute("y", h - actH);
    actRect.setAttribute("width", barW * 0.35); actRect.setAttribute("height", actH);
    actRect.setAttribute("fill", "#2E9E5B");
    g.appendChild(actRect);
  });
  g.appendChild(Object.assign(document.createElementNS("http://www.w3.org/2000/svg", "line"), {}));
  svg.appendChild(g);
  container.innerHTML = "";
  container.appendChild(svg);
}

function parseCsvRows(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      const raw = cells[i];
      row[h] = isNaN(Number(raw)) ? raw : Number(raw);
    });
    return row;
  });
}

function rowsToCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")].concat(rows.map((r) => headers.map((h) => r[h]).join(",")));
  return lines.join("\n");
}

async function runBulk() {
  const errBox = document.getElementById("bulk-error");
  errBox.style.display = "none";
  let rows = [];
  const csvFile = document.getElementById("csv-upload").files[0];
  try {
    if (csvFile) {
      const text = await csvFile.text();
      rows = parseCsvRows(text);
    } else {
      const raw = document.getElementById("bulk-json").value.trim();
      if (!raw) { errBox.textContent = "Paste JSON rows or choose a CSV file first."; errBox.style.display = "block"; return; }
      rows = JSON.parse(raw);
    }
  } catch (e) {
    errBox.textContent = "Could not parse input: " + e.message;
    errBox.style.display = "block";
    return;
  }
  if (rows.length > MAX_BULK_ROWS) {
    errBox.textContent = `Max ${MAX_BULK_ROWS} rows (got ${rows.length}).`;
    errBox.style.display = "block";
    return;
  }

  const res = await fetch("/predict/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requests: rows }),
  });
  const body = await res.json();
  if (!res.ok) {
    errBox.textContent = `Request failed (HTTP ${res.status}): ${JSON.stringify(body.detail || body)}`;
    errBox.style.display = "block";
    return;
  }

  const resultsRows = rows.map((row, i) => ({ ...row, predicted_travel_time_s: body.predictions[i].predicted_travel_time_s }));
  const div = document.getElementById("bulk-results");
  div.textContent = `${resultsRows.length} predictions returned.`;
  const csvBtn = document.getElementById("download-csv-btn");
  csvBtn.style.display = "inline-block";
  csvBtn.onclick = () => {
    const blob = new Blob([rowsToCsv(resultsRows)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "predictions.csv"; a.click();
    URL.revokeObjectURL(url);
  };
}

async function init() {
  await loadFooter();
  trips = await fetchJSON(`${DATA_BASE}/trips.json`);
  populateTripSelect();
  if (trips.length) await runTrip(trips[0]);
  document.getElementById("run-bulk-btn").addEventListener("click", runBulk);
}

init();
