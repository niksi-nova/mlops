/* Pipeline page: three route-line tracks of stations + model passport + gate table. */

const STATIONS = {
  phase1: [
    { id: "ingest", label: "Ingest", tool: "Python, requests", file: "src/data/ingest.py", number: null, desc: "Downloads and extracts the raw Delhi bus GPS/stop-event archive." },
    { id: "clean", label: "Clean & segment", tool: "pandas", file: "src/data/clean.py, src/data/build_segments.py", number: "raw_rows", desc: "Dedupes stop events, drops bad trips, builds stop-to-stop segments." },
    { id: "validate", label: "Validate", tool: "Pandera", file: "src/data/validate.py", number: "validation_status", desc: "Schema/range checks on the cleaned segment table before featurizing." },
    { id: "featurize", label: "Featurize", tool: "pandas, sklearn GroupKFold", file: "src/features/build_features.py", number: "n_features", desc: "Builds the 25-feature train/val/test tables with out-of-fold historical stats." },
    { id: "train", label: "Train & tune", tool: "scikit-learn, LightGBM, W&B sweeps", file: "src/models/train.py", number: "n_models", desc: "Trains 7 candidate models, tracked in Weights & Biases." },
    { id: "evaluate", label: "Evaluate", tool: "pandas, matplotlib", file: "src/models/evaluate.py", number: "test_mae", desc: "Final held-out test evaluation of the model selected on validation." },
  ],
  phase2: [
    { id: "tests", label: "Tests & gate", tool: "pytest, custom quality gate", file: "tests/, src/models/quality_gate.py", number: "test_count", desc: "Unit tests plus a quality gate checking MAE, generalization gap, latency, and more." },
    { id: "registry", label: "Model registry", tool: "Weights & Biases Registry", file: "src/models/registry.py", number: "registry_version", desc: "candidate → staging (auto on gate pass) → production (human-approved only)." },
    { id: "api", label: "Prediction API", tool: "FastAPI, Pydantic", file: "src/api/main.py", number: null, desc: "Serves /predict, /predict/batch, /health, /model-info - this website too." },
    { id: "docker", label: "Docker", tool: "Docker, uv", file: "Dockerfile", number: null, desc: "Slim runtime image (constraints.txt pinned from uv.lock) under 1GB." },
    { id: "cicd", label: "CI/CD", tool: "GitHub Actions", file: ".github/workflows/ci.yml, cd.yml", number: null, desc: "Tests, quality gate, Docker smoke test on every push; build+push+deploy on main." },
    { id: "deploy", label: "Deploy", tool: "GHCR, Render", file: "docs/deployment.md", number: null, desc: "Published image deployed to a free Render web service via a deploy hook." },
  ],
  phase3: [
    { id: "logging", label: "Logging", tool: "planned", file: null, number: null, desc: "Structured request/prediction logs for every call (planned)." },
    { id: "dashboard", label: "Dashboard", tool: "planned", file: null, number: null, desc: "A live ops dashboard over those logs (planned)." },
    { id: "drift", label: "Drift detection", tool: "planned", file: null, number: null, desc: "Monitoring prediction/feature drift over time (planned)." },
    { id: "retrain", label: "Retrain trigger", tool: "planned", file: null, number: null, desc: "Automatic retraining when drift crosses a threshold (planned)." },
    { id: "compare", label: "Compare models", tool: "planned", file: null, number: null, desc: "Champion/challenger comparison before promotion (planned)." },
    { id: "redeploy", label: "Redeploy", tool: "planned", file: null, number: null, desc: "Automated redeploy of a newly-promoted model (planned)." },
  ],
};

function drawTrack(svgId, stations, planned) {
  const svg = document.getElementById(svgId);
  const n = stations.length;
  const width = Math.max(500, n * 150);
  svg.setAttribute("viewBox", `0 0 ${width} 110`);
  const y = 40;
  const spacing = (width - 60) / (n - 1 || 1);

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", 30); line.setAttribute("y1", y); line.setAttribute("x2", width - 30); line.setAttribute("y2", y);
  line.setAttribute("stroke", "#1B1A17"); line.setAttribute("stroke-width", 3);
  svg.appendChild(line);

  stations.forEach((st, i) => {
    const x = 30 + i * spacing;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "pipeline-station" + (planned ? " planned" : ""));
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "button");
    g.setAttribute("aria-label", st.label);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x); circle.setAttribute("cy", y); circle.setAttribute("r", 9);
    circle.setAttribute("class", "station-dot");
    g.appendChild(circle);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x); text.setAttribute("y", y + 26); text.setAttribute("text-anchor", "middle");
    text.textContent = st.label;
    g.appendChild(text);
    if (planned) {
      const tag = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tag.setAttribute("x", x); tag.setAttribute("y", y - 16); tag.setAttribute("text-anchor", "middle");
      tag.setAttribute("font-size", "9"); tag.textContent = "planned";
      g.appendChild(tag);
    }
    const activate = () => showStation(st);
    g.addEventListener("click", activate);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
    svg.appendChild(g);
  });
}

let pipelineData = null;

function resolveNumber(key) {
  if (!key || !pipelineData) return "n/a";
  if (key === "n_models") return "7";
  if (key === "test_count") return String(pipelineData.test_count);
  if (key === "registry_version") return pipelineData.model_passport.version || "n/a";
  return "see Data/Models pages";
}

function showStation(st) {
  document.getElementById("station-detail").style.display = "block";
  document.getElementById("station-title").textContent = st.label;
  document.getElementById("station-desc").textContent = st.desc;
  document.getElementById("station-tool").textContent = st.tool;
  document.getElementById("station-file").textContent = st.file || "—";
  document.getElementById("station-number").textContent = resolveNumber(st.number);
}

function renderPassport(p) {
  const dl = document.getElementById("passport");
  const entries = [
    ["alias", p.model_passport.alias],
    ["version", p.model_passport.version],
    ["model", p.model_passport.model_id],
    ["git commit", p.model_passport.git_commit],
    ["approved by", p.model_passport.approved_by || "not yet promoted"],
    ["approved at", p.model_passport.approved_at || "—"],
    ["registry collection", p.model_passport.collection],
    ["workspace", p.model_passport.workspace],
    ["DVC stages", p.stages.join(" → ")],
  ];
  dl.innerHTML = "";
  entries.forEach(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    dl.appendChild(dt); dl.appendChild(dd);
  });
}

function renderGateTable(p) {
  const tbody = document.getElementById("gate-table");
  tbody.innerHTML = "";
  Object.entries(p.quality_gate.checks).forEach(([name, c]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td style="text-align:left">${name}</td><td class="text-right">${c.operator} ${c.threshold}</td><td class="text-right">${c.value.toFixed(2)}</td><td><span class="pill ${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</span></td>`;
    tbody.appendChild(tr);
  });
}

async function init() {
  const meta = await fetchJSON(`${DATA_BASE}/meta.json`);
  document.getElementById("site-footer").textContent = `snapshot built ${meta.snapshot_built} from commit ${meta.git_commit}`;

  pipelineData = await fetchJSON(`${DATA_BASE}/pipeline.json`);
  drawTrack("phase1-track", STATIONS.phase1, false);
  drawTrack("phase2-track", STATIONS.phase2, false);
  drawTrack("phase3-track", STATIONS.phase3, true);
  renderPassport(pipelineData);
  renderGateTable(pipelineData);
}

init();
