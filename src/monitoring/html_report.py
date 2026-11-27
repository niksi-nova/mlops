"""Render the drift report dict as a standalone HTML page."""
from __future__ import annotations

import html

COLORS = {"OK": "#1a7f37", "WARNING": "#9a6700", "ALERT": "#cf222e"}


def _badge(status):
    return '<span class="b" style="background:%s">%s</span>' % (
        COLORS.get(status, "#555"), html.escape(str(status)))


def render_html(rep):
    e = html.escape
    out = ["<!doctype html><html><head><meta charset='utf-8'>",
           "<title>Drift monitoring report</title><style>",
           "body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1f2328}",
           "table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}",
           "th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:right;font-size:.9rem}",
           "th{background:#f6f8fa}td.l,th.l{text-align:left}",
           ".b{color:#fff;padding:2px 8px;border-radius:10px;font-size:.8rem}",
           ".warn{background:#fff8c5;border:1px solid #d4a72c;padding:.6rem 1rem;border-radius:6px}",
           "</style></head><body><h1>Drift monitoring report</h1>"]
    out.append("<div class='warn'><b>SIMULATED.</b> %s</div>" % e(rep.get("disclaimer", "")))
    ref = rep["reference"]
    out.append("<p>Generated %s. Reference: %s, %s, %s rows. Model %s (test MAE %s s).</p>" % (
        e(rep["generated_at"]), e(str(ref["dataset_version"])), e(str(ref["split"])),
        e(str(ref["n_rows"])), e(str(ref.get("model_id", ""))), e(str(ref.get("model_test_mae_s", "")))))
    out.append("<h2>Thresholds</h2><p>%s</p>" % e(", ".join(
        "%s=%s" % (k, v) for k, v in rep["thresholds"].items())))
    sm = rep.get("summary", {})
    out.append("<h2>Summary</h2><p>%s</p>" % e(", ".join("%s: %s" % (k, v) for k, v in sm.items())))
    for s in rep["scenarios"]:
        out.append("<h2>%s %s</h2>" % (e(s["name"] + " · " + s.get("label", s["name"])), _badge(s["status"])))
        out.append("<p>%s<br>Induced (simulated): %s. First alert: %s. Retrain recommended: %s.</p>" % (
            e(s.get("description", "")), s["induced"], s.get("first_alert_window"),
            s.get("retrain_recommended")))
        out.append("<table><tr><th class='l'>Window</th><th>Days</th><th>n</th><th>MAE (s)</th>"
                   "<th>MAE vs 30.60 (%)</th><th>Max PSI</th><th>Has prev bus</th>"
                   "<th class='l'>Drifted features</th><th>Status</th></tr>")
        for w in s["windows"]:
            out.append("<tr><td class='l'>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                       "<td>%s</td><td class='l'>%s</td><td>%s</td></tr>" % (
                           e(w["window"]), e(str(w.get("days", ""))), w["n"], w["mae"],
                           w["mae_degradation_pct"], w["max_psi"],
                           w.get("has_prev_bus_share", ""), e(", ".join(w["drifted_features"])),
                           _badge(w["status"])))
        out.append("</table>")
        top = s.get("top_drifted_features") or []
        if top:
            out.append("<p><b>Top drifted features:</b> %s</p>" % e(", ".join(
                "%s (PSI %s)" % (t["feature"], t.get("psi")) for t in top)))
    out.append("</body></html>")
    return "".join(out)
