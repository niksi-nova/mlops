/* Small hand-built chart helpers - no chart library. Each function renders
 * into a container element and, if a `tableTarget` is given, also fills an
 * accessible <table> inside a <details class="data-table-toggle"> for the
 * "view data table" requirement. */

const CHART_COLORS = {
  ink: "#1B1A17",
  green: "#1F6B3A",
  red: "#B3261E",
  orange: "#E8590C",
  lineRed: "#D7263D",
  lineYellow: "#F2B705",
  lineBlue: "#1F6FEB",
  lineGreen: "#2E9E5B",
};

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v !== undefined && v !== null) el.setAttribute(k, v);
  }
  return el;
}

function fillDataTable(tableTarget, headers, rows) {
  if (!tableTarget) return;
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  tableTarget.innerHTML = "";
  tableTarget.appendChild(table);
}

/** Paired/grouped vertical bar chart. series: [{label, color, values: [{x, y}]}] */
function barChart(container, series, { width = 640, height = 280, yLabel = "", tableTarget = null } = {}) {
  const margin = { top: 16, right: 16, bottom: 64, left: 48 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const categories = series[0].values.map((v) => v.x);
  const maxY = Math.max(...series.flatMap((s) => s.values.map((v) => v.y)), 1) * 1.1;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img", "aria-label": yLabel });
  const g = svgEl("g", { transform: `translate(${margin.left},${margin.top})` });

  // axes
  g.appendChild(svgEl("line", { x1: 0, y1: h, x2: w, y2: h, stroke: CHART_COLORS.ink, "stroke-width": 1.5 }));
  g.appendChild(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: h, stroke: CHART_COLORS.ink, "stroke-width": 1.5 }));

  const groupWidth = w / categories.length;
  const barWidth = (groupWidth * 0.7) / series.length;

  categories.forEach((cat, ci) => {
    series.forEach((s, si) => {
      const val = s.values[ci].y;
      const barH = (val / maxY) * h;
      const x = ci * groupWidth + groupWidth * 0.15 + si * barWidth;
      g.appendChild(
        svgEl("rect", { x, y: h - barH, width: barWidth * 0.9, height: barH, fill: s.color })
      );
    });
    // Long category names (e.g. model names) would overlap their
    // neighbours if drawn horizontally at this density - rotate them.
    const longLabels = String(cat).length > 6;
    const labelX = ci * groupWidth + groupWidth / 2;
    const labelY = h + (longLabels ? 10 : 16);
    const label = svgEl("text", {
      x: labelX,
      y: labelY,
      "text-anchor": longLabels ? "end" : "middle",
      "font-size": "10",
      "font-family": "var(--font-ticket)",
      transform: longLabels ? `rotate(-35 ${labelX} ${labelY})` : undefined,
    });
    label.textContent = cat;
    g.appendChild(label);
  });

  // y ticks
  for (let i = 0; i <= 4; i++) {
    const val = (maxY / 4) * i;
    const y = h - (val / maxY) * h;
    g.appendChild(svgEl("line", { x1: -4, y1: y, x2: w, y2: y, stroke: CHART_COLORS.ink, "stroke-width": 0.3, opacity: 0.3 }));
    const t = svgEl("text", { x: -8, y: y + 3, "text-anchor": "end", "font-size": "9", "font-family": "var(--font-num)" });
    t.textContent = val.toFixed(0);
    g.appendChild(t);
  }

  svg.appendChild(g);
  container.innerHTML = "";
  container.appendChild(svg);

  if (tableTarget) {
    const headers = ["category", ...series.map((s) => s.label)];
    const rows = categories.map((cat, ci) => [cat, ...series.map((s) => s.values[ci].y.toFixed(2))]);
    fillDataTable(tableTarget, headers, rows);
  }
}

/** Line/area chart with optional IQR band. points: [{x, y, lo, hi}] */
function lineBandChart(container, points, { width = 640, height = 260, color = CHART_COLORS.ink, tableTarget = null, xLabel = "", yLabel = "" } = {}) {
  const margin = { top: 16, right: 16, bottom: 32, left: 48 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const xs = points.map((p) => p.x);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const allY = points.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]);
  const maxY = Math.max(...allY) * 1.1;
  const minY = 0;

  const sx = (x) => ((x - minX) / (maxX - minX || 1)) * w;
  const sy = (y) => h - ((y - minY) / (maxY - minY || 1)) * h;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img", "aria-label": `${yLabel} by ${xLabel}` });
  const g = svgEl("g", { transform: `translate(${margin.left},${margin.top})` });

  g.appendChild(svgEl("line", { x1: 0, y1: h, x2: w, y2: h, stroke: CHART_COLORS.ink, "stroke-width": 1.5 }));
  g.appendChild(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: h, stroke: CHART_COLORS.ink, "stroke-width": 1.5 }));

  if (points.some((p) => p.lo !== undefined)) {
    const bandPts = points.map((p) => `${sx(p.x)},${sy(p.hi)}`).concat(points.slice().reverse().map((p) => `${sx(p.x)},${sy(p.lo)}`));
    g.appendChild(svgEl("polygon", { points: bandPts.join(" "), fill: color, opacity: 0.15 }));
  }

  const linePts = points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ");
  g.appendChild(svgEl("polyline", { points: linePts, fill: "none", stroke: color, "stroke-width": 2 }));
  points.forEach((p) => g.appendChild(svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 2.5, fill: color })));

  [minX, Math.round((minX + maxX) / 2), maxX].forEach((xv) => {
    const t = svgEl("text", { x: sx(xv), y: h + 14, "text-anchor": "middle", "font-size": "9", "font-family": "var(--font-num)" });
    t.textContent = xv;
    g.appendChild(t);
  });

  svg.appendChild(g);
  container.innerHTML = "";
  container.appendChild(svg);

  if (tableTarget) {
    fillDataTable(tableTarget, [xLabel, "median", "q25", "q75"], points.map((p) => [p.x, p.y.toFixed(1), (p.lo ?? "").toString(), (p.hi ?? "").toString()]));
  }
}

/** Heatmap. matrix: rows (y categories) of arrays (x categories), values or null. */
function heatmap(container, matrix, xLabels, yLabels, { width = 640, cell = 24, tableTarget = null, colorScale = null } = {}) {
  const height = yLabels.length * cell + 50;
  const margin = { top: 10, right: 10, bottom: 30, left: 40 };
  const flat = matrix.flat().filter((v) => v !== null && v !== undefined);
  const maxV = Math.max(...flat, 1);
  const minV = Math.min(...flat, 0);
  const color = colorScale || ((v) => {
    const t = (v - minV) / (maxV - minV || 1);
    const r = Math.round(243 - t * (243 - 232));
    const g = Math.round(235 - t * (235 - 89));
    const b = Math.round(216 - t * (216 - 12));
    return `rgb(${r},${g},${b})`;
  });

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img", "aria-label": "heatmap" });
  const g = svgEl("g", { transform: `translate(${margin.left},${margin.top})` });
  const cellW = (width - margin.left - margin.right) / xLabels.length;

  matrix.forEach((row, ri) => {
    row.forEach((v, ci) => {
      const rect = svgEl("rect", {
        x: ci * cellW, y: ri * cell, width: cellW - 1, height: cell - 1,
        fill: v === null || v === undefined ? "#eee" : color(v),
        stroke: "#fff",
      });
      g.appendChild(rect);
    });
    const yt = svgEl("text", { x: -6, y: ri * cell + cell / 2 + 3, "text-anchor": "end", "font-size": "9", "font-family": "var(--font-num)" });
    yt.textContent = yLabels[ri];
    g.appendChild(yt);
  });
  xLabels.forEach((lab, ci) => {
    if (ci % 2 !== 0) return;
    const xt = svgEl("text", { x: ci * cellW + cellW / 2, y: yLabels.length * cell + 14, "text-anchor": "middle", "font-size": "9", "font-family": "var(--font-num)" });
    xt.textContent = lab;
    g.appendChild(xt);
  });

  svg.appendChild(g);
  container.innerHTML = "";
  container.appendChild(svg);

  if (tableTarget) {
    fillDataTable(tableTarget, ["", ...xLabels], matrix.map((row, ri) => [yLabels[ri], ...row.map((v) => (v === null ? "—" : v))]));
  }
}

/** Scatter plot on canvas (fast for ~2000 points) with a y=x reference line. */
function scatterCanvas(canvas, points, { xLabel = "actual (s)", yLabel = "predicted (s)", tableTarget = null } = {}) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 640;
  const cssH = 360;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.height = cssH + "px";
  ctx.scale(dpr, dpr);

  const margin = { top: 10, right: 10, bottom: 30, left: 44 };
  const w = cssW - margin.left - margin.right;
  const h = cssH - margin.top - margin.bottom;
  const maxV = Math.max(...points.map((p) => Math.max(p.x, p.y)), 1) * 1.05;

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.translate(margin.left, margin.top);

  ctx.strokeStyle = CHART_COLORS.ink;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, h); ctx.lineTo(w, h);
  ctx.moveTo(0, 0); ctx.lineTo(0, h);
  ctx.stroke();

  const sx = (v) => (v / maxV) * w;
  const sy = (v) => h - (v / maxV) * h;

  ctx.strokeStyle = CHART_COLORS.orange;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(0), sy(0));
  ctx.lineTo(sx(maxV), sy(maxV));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(31,111,235,0.35)";
  points.forEach((p) => {
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 2.2, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = CHART_COLORS.ink;
  ctx.font = "9px monospace";
  ctx.fillText("0", -8, h + 10);
  ctx.fillText(String(Math.round(maxV)), w - 10, h + 10);

  if (tableTarget) {
    const sample = points.slice(0, 200);
    fillDataTable(tableTarget, [xLabel, yLabel], sample.map((p) => [p.x.toFixed(1), p.y.toFixed(1)]));
  }
}

/** Simple histogram from precomputed counts/bin_edges. */
function histogram(container, counts, binEdges, { width = 640, height = 220, color = CHART_COLORS.green, tableTarget = null, xLabel = "" } = {}) {
  const margin = { top: 10, right: 10, bottom: 28, left: 40 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const maxCount = Math.max(...counts, 1);

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img", "aria-label": `histogram of ${xLabel}` });
  const g = svgEl("g", { transform: `translate(${margin.left},${margin.top})` });
  g.appendChild(svgEl("line", { x1: 0, y1: h, x2: w, y2: h, stroke: CHART_COLORS.ink, "stroke-width": 1.5 }));

  const barW = w / counts.length;
  counts.forEach((c, i) => {
    const barH = (c / maxCount) * h;
    g.appendChild(svgEl("rect", { x: i * barW, y: h - barH, width: barW - 1, height: barH, fill: color }));
  });

  [0, Math.floor(counts.length / 2), counts.length - 1].forEach((i) => {
    const t = svgEl("text", { x: i * barW, y: h + 14, "text-anchor": "middle", "font-size": "9", "font-family": "var(--font-num)" });
    t.textContent = binEdges[i] !== undefined ? binEdges[i].toFixed(1) : "";
    g.appendChild(t);
  });

  svg.appendChild(g);
  container.innerHTML = "";
  container.appendChild(svg);

  if (tableTarget) {
    fillDataTable(tableTarget, ["bin start", "count"], counts.map((c, i) => [binEdges[i].toFixed(2), c]));
  }
}
