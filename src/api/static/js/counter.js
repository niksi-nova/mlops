/* Counter page logic: segment picker, time/real-time inputs, predict. */

let routesMeta = null;
let currentLookup = null;
let presets = [];
let selectedRoute = null, selectedDirection = null, selectedFromNode = null;
let testMaeS = null;

async function loadMeta() {
  const meta = await fetchJSON(`${DATA_BASE}/meta.json`);
  document.getElementById("site-footer").textContent =
    `snapshot built ${meta.snapshot_built} from commit ${meta.git_commit}`;
  try {
    const models = await fetchJSON(`${DATA_BASE}/models.json`);
    testMaeS = models.test_mae_best_model;
  } catch { /* non-fatal */ }
}

function populateRouteSelects() {
  const routeSel = document.getElementById("route-select");
  routeSel.innerHTML = "";
  Object.keys(routesMeta).forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r; opt.textContent = `Route ${r}`;
    routeSel.appendChild(opt);
  });
  routeSel.addEventListener("change", onRouteChange);
  document.getElementById("direction-select").addEventListener("change", onDirectionChange);
}

async function onRouteChange() {
  selectedRoute = document.getElementById("route-select").value;
  const dirSel = document.getElementById("direction-select");
  dirSel.innerHTML = "";
  Object.keys(routesMeta[selectedRoute]).forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d; opt.textContent = d;
    dirSel.appendChild(opt);
  });
  await onDirectionChange();
}

async function onDirectionChange() {
  selectedDirection = document.getElementById("direction-select").value;
  currentLookup = await fetchJSON(`${DATA_BASE}/lookup_${selectedRoute}_${selectedDirection}.json`);
  selectedFromNode = Object.keys(currentLookup)[0];
  drawRouteLine();
}

function drawRouteLine() {
  const info = routesMeta[selectedRoute][selectedDirection];
  const svg = document.getElementById("route-line-svg");
  const n = info.n_stops;
  const width = Math.max(400, n * 22);
  svg.setAttribute("viewBox", `0 0 ${width} 90`);
  svg.innerHTML = "";
  const y = 45;
  const spacing = (width - 40) / (n - 1);

  for (let i = 0; i < n - 1; i++) {
    const seg = info.segments[String(i)];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", 20 + i * spacing);
    line.setAttribute("y1", y);
    line.setAttribute("x2", 20 + (i + 1) * spacing);
    line.setAttribute("y2", y);
    line.setAttribute("class", "seg-line" + (String(i) === selectedFromNode ? " selected" : ""));
    line.dataset.fromNode = i;
    line.style.cursor = "pointer";
    line.addEventListener("click", () => { selectedFromNode = String(i); drawRouteLine(); });
    svg.appendChild(line);
  }
  for (let i = 0; i < n; i++) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", 20 + i * spacing);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", 6);
    circle.setAttribute("class", "stop-dot");
    svg.appendChild(circle);
  }
  const seg = info.segments[selectedFromNode];
  document.getElementById("segment-readout").textContent =
    `segment ${seg ? seg.segment_id : "?"} (${info.segments[selectedFromNode]?.n_rows ?? 0} test rows)`;
  applyLookupDefaults();
}

function currentHour() {
  return Math.floor(Number(document.getElementById("minute-slider").value) / 60);
}

function applyLookupDefaults() {
  if (!currentLookup || !selectedFromNode) return;
  const seg = currentLookup[selectedFromNode];
  if (!seg) return;
  const hourKey = String(currentHour());
  const hourly = seg.by_hour[hourKey] || seg.overall;
  document.getElementById("prev-bus-tt").value = hourly.prev_bus_tt_s;
  document.getElementById("prev3-median-tt").value = hourly.prev3_median_tt_s;
  document.getElementById("headway").value = hourly.headway_s;
  document.getElementById("n-buses-recent").value = hourly.n_buses_recent;
  setHasPrevBus(hourly.has_prev_bus >= 0.5 ? 1 : 0);
}

function setHasPrevBus(val) {
  document.querySelectorAll("#has-prev-bus-toggle button").forEach((b) => {
    b.setAttribute("aria-pressed", String(Number(b.dataset.val) === val));
  });
}

function getHasPrevBus() {
  const pressed = document.querySelector('#has-prev-bus-toggle button[aria-pressed="true"]');
  return pressed ? Number(pressed.dataset.val) : 1;
}

function updateMinuteReadout() {
  const minute = Number(document.getElementById("minute-slider").value);
  document.getElementById("minute-readout").textContent = formatMinuteOfDay(minute);
}

function buildRequestFromForm() {
  const seg = currentLookup[selectedFromNode];
  const minuteOfDay = Number(document.getElementById("minute-slider").value);
  const timeFeatures = deriveTimeFeatures(minuteOfDay);
  const hourKey = String(timeFeatures.hour);
  const hourly = seg.by_hour[hourKey] || seg.overall;

  return {
    route: selectedRoute,
    direction: selectedDirection,
    segment_id: seg ? routesMeta[selectedRoute][selectedDirection].segments[selectedFromNode].segment_id : "",
    relative_weekday: Number(document.getElementById("weekday-select").value),
    from_node: Number(selectedFromNode),
    nodes_spanned: 1,
    route_position: seg.static.route_position,
    stops_into_trip: Math.round(seg.static.stops_into_trip),
    elapsed_trip_s: seg.static.elapsed_trip_s,
    hour: timeFeatures.hour,
    minute_of_day: timeFeatures.minute_of_day,
    hour_sin: timeFeatures.hour_sin,
    hour_cos: timeFeatures.hour_cos,
    is_peak_am: timeFeatures.is_peak_am,
    is_peak_pm: timeFeatures.is_peak_pm,
    prev_bus_tt_s: Number(document.getElementById("prev-bus-tt").value),
    prev3_median_tt_s: Number(document.getElementById("prev3-median-tt").value),
    headway_s: Number(document.getElementById("headway").value),
    n_buses_recent: Math.round(Number(document.getElementById("n-buses-recent").value)),
    has_prev_bus: getHasPrevBus(),
    own_prev_segment_tt_s: hourly.own_prev_segment_tt_s,
    own_prev_segment_ratio: hourly.own_prev_segment_ratio,
    typical_segment_hour_s: hourly.typical_segment_hour_s,
    typical_segment_s: seg.static.typical_segment_s,
    te_vehicle: seg.static.te_vehicle,
  };
}

function renderPresets() {
  const wrap = document.getElementById("preset-buttons");
  wrap.innerHTML = "";
  presets.forEach((p) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "secondary";
    btn.textContent = p.label;
    btn.addEventListener("click", () => applyPreset(p));
    wrap.appendChild(btn);
  });
}

async function applyPreset(preset) {
  const f = preset.features;
  document.getElementById("route-select").value = f.route;
  await onRouteChange();
  document.getElementById("direction-select").value = f.direction;
  await onDirectionChange();
  const info = routesMeta[f.route][f.direction];
  const fromNode = Object.keys(info.segments).find((k) => info.segments[k].segment_id === f.segment_id);
  selectedFromNode = fromNode ?? Object.keys(info.segments)[0];
  document.getElementById("minute-slider").value = f.minute_of_day;
  updateMinuteReadout();
  document.getElementById("weekday-select").value = f.relative_weekday;
  document.getElementById("prev-bus-tt").value = f.prev_bus_tt_s;
  document.getElementById("prev3-median-tt").value = f.prev3_median_tt_s;
  document.getElementById("headway").value = f.headway_s;
  document.getElementById("n-buses-recent").value = f.n_buses_recent;
  setHasPrevBus(f.has_prev_bus);
  drawRouteLine();
  await runPredict(preset.actual_travel_time_s);
}

function renderTicket(predictedS, requestId, actualS) {
  const ticket = document.getElementById("ticket");
  ticket.classList.remove("printing");
  void ticket.offsetWidth; // restart animation
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) ticket.classList.add("printing");

  document.getElementById("ticket-serial").textContent = (requestId || "").slice(0, 8).toUpperCase();
  document.getElementById("ticket-seconds").textContent = predictedS.toFixed(1);
  const stamp = document.getElementById("ticket-stamp");
  if (actualS !== undefined && actualS !== null) {
    stamp.style.display = "inline-block";
    const diff = predictedS - actualS;
    stamp.textContent = Math.abs(diff) <= 30 ? "ON TIME" : diff > 0 ? "RUNNING LATE" : "AHEAD OF SCHEDULE";
    stamp.style.borderColor = Math.abs(diff) <= 30 ? "var(--dtc-green)" : "var(--stamp-red)";
    stamp.style.color = Math.abs(diff) <= 30 ? "var(--dtc-green)" : "var(--stamp-red)";
  } else {
    stamp.style.display = "none";
  }
  document.getElementById("ticket-meta").textContent =
    `route ${selectedRoute} · ${selectedDirection} · ${formatMinuteOfDay(Number(document.getElementById("minute-slider").value))}` +
    (actualS !== undefined && actualS !== null ? ` · observed ${actualS.toFixed(1)}s` : "");
}

function renderRuler(predictedS, typicalS, actualS) {
  const track = document.getElementById("ruler-track");
  track.innerHTML = "";
  const maxV = Math.max(predictedS, typicalS, actualS || 0) * 1.25 + 10;

  const ticks = [
    { value: typicalS, cls: "typical", label: `typical ${typicalS.toFixed(0)}s` },
    { value: predictedS, cls: "predicted", label: `predicted ${predictedS.toFixed(0)}s` },
  ];
  if (actualS !== undefined && actualS !== null) {
    ticks.push({ value: actualS, cls: "actual", label: `observed ${actualS.toFixed(0)}s` });
  }
  // Stagger label heights by proximity so close-together ticks (a common
  // case: predicted and observed are often within a few seconds of each
  // other) never render their labels on top of one another.
  const sorted = ticks.slice().sort((a, b) => a.value - b.value);
  const closeThresholdPct = 10;
  let tier = 0;
  sorted.forEach((t, i) => {
    if (i > 0) {
      const prevPct = (sorted[i - 1].value / maxV) * 100;
      const pct = (t.value / maxV) * 100;
      tier = Math.abs(pct - prevPct) < closeThresholdPct ? tier + 1 : 0;
    }
    t.labelOffset = 18 + tier * 16;
  });

  ticks.forEach(({ value, cls, label, labelOffset }) => {
    const pct = (value / maxV) * 100;
    const tick = document.createElement("div");
    tick.className = `ruler-tick ${cls}`;
    tick.style.left = `${pct}%`;
    const lab = document.createElement("span");
    lab.className = "tick-label";
    lab.style.bottom = `${labelOffset}px`;
    lab.textContent = label;
    tick.appendChild(lab);
    track.appendChild(tick);
  });

  const delay = predictedS - typicalS;
  const note = document.getElementById("ruler-note");
  note.textContent =
    `${delay >= 0 ? "+" : ""}${delay.toFixed(1)}s vs typical for this segment and hour.` +
    (testMaeS ? ` Usually within about ±${testMaeS.toFixed(0)}s (test MAE).` : "");
}

async function runPredict(actualS) {
  const errBox = document.getElementById("predict-error");
  errBox.style.display = "none";
  const request = buildRequestFromForm();
  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    const body = await res.json();
    if (res.status === 422) {
      errBox.textContent = "Some inputs are out of range: " + JSON.stringify(body.detail);
      errBox.style.display = "block";
      return;
    }
    if (res.status === 503) {
      errBox.textContent = "The model isn't loaded right now. Try again in a moment.";
      errBox.style.display = "block";
      return;
    }
    if (!res.ok) {
      errBox.textContent = `Prediction failed (HTTP ${res.status}).`;
      errBox.style.display = "block";
      return;
    }
    document.getElementById("result-area").style.display = "grid";
    renderTicket(body.predicted_travel_time_s, body.request_id, actualS);
    renderRuler(body.predicted_travel_time_s, request.typical_segment_hour_s, actualS);
    document.getElementById("req-res-json").textContent = JSON.stringify({ request, response: body }, null, 2);
  } catch (e) {
    errBox.textContent = "Network error calling /predict.";
    errBox.style.display = "block";
  }
}

async function init() {
  await loadMeta();
  routesMeta = await fetchJSON(`${DATA_BASE}/routes.json`);
  presets = await fetchJSON(`${DATA_BASE}/presets.json`);
  populateRouteSelects();
  renderPresets();
  await onRouteChange();
  updateMinuteReadout();

  document.getElementById("minute-slider").addEventListener("input", () => { updateMinuteReadout(); applyLookupDefaults(); });
  document.querySelectorAll("#has-prev-bus-toggle button").forEach((b) => b.addEventListener("click", () => setHasPrevBus(Number(b.dataset.val))));
  document.getElementById("predict-btn").addEventListener("click", () => runPredict());
}

init();
