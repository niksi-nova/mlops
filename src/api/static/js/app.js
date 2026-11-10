/* Shared across every page: the status chip (health/model-info polling),
 * a small fetch-JSON helper, and marking the current nav station. */

const DATA_BASE = "/static/data";

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

function markCurrentNavStation() {
  const here = window.location.pathname.replace(/\/$/, "") || "/";
  document.querySelectorAll(".route-nav a").forEach((a) => {
    const target = a.getAttribute("href").replace(/\/$/, "") || "/";
    if (target === here) a.setAttribute("aria-current", "page");
  });
}

async function pollStatusChip() {
  const chip = document.getElementById("status-chip");
  if (!chip) return;

  const dot = chip.querySelector(".dot");
  const bus = chip.querySelector(".bus");
  const label = chip.querySelector(".chip-label");

  function setState(cls, text) {
    chip.classList.remove("ok", "bad", "waking");
    chip.classList.add(cls);
    label.textContent = text;
  }

  setState("waking", "checking service…");

  const deadline = Date.now() + 120000;
  let attempt = 0;
  while (Date.now() < deadline) {
    attempt += 1;
    try {
      const health = await fetchJSON("/health");
      if (health.model_loaded) {
        try {
          const info = await fetchJSON("/model-info");
          setState(
            "ok",
            `${info.alias} ${info.version} · ${info.source} · ${(info.git_commit || "").slice(0, 7)}`
          );
        } catch {
          setState("ok", "model loaded");
        }
        return;
      }
      setState("waking", "waking the service, about a minute…" + " ".repeat(0));
    } catch {
      setState("waking", `waking the service, about a minute… (attempt ${attempt})`);
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
  setState("bad", "service unavailable");
}

document.addEventListener("DOMContentLoaded", () => {
  markCurrentNavStation();
  pollStatusChip();
});
