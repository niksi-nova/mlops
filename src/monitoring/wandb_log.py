"""Log the drift report to W&B (project delhi-bus-delay, job_type monitoring)."""
from __future__ import annotations

import os


def log_to_wandb(rep, html_path=None):
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set, skipping W&B logging")
        return None
    import wandb

    run = wandb.init(project="delhi-bus-delay", entity="anikaub20-rv-college",
                     job_type="monitoring", name="drift-demo-simulated",
                     config={"thresholds": rep["thresholds"], "simulated": True})
    run.define_metric("window_idx")
    run.define_metric("*", step_metric="window_idx")
    table = wandb.Table(columns=["scenario", "window", "n", "mae", "mae_degradation_pct",
                                 "max_psi", "status"])
    for s in rep["scenarios"]:
        for i, w in enumerate(s["windows"]):
            row = {"window_idx": i + 1,
                   "%s/mae" % s["name"]: w["mae"],
                   "%s/max_psi" % s["name"]: w["max_psi"],
                   "%s/mae_degradation_pct" % s["name"]: w["mae_degradation_pct"]}
            for f in w["features"]:
                row["%s/psi/%s" % (s["name"], f["feature"])] = f["psi"]
            run.log(row)
            table.add_data(s["name"], w["window"], w["n"], w["mae"],
                           w["mae_degradation_pct"], w["max_psi"], w["status"])
    run.log({"status_table": table})
    run.summary["n_alert"] = rep["summary"]["n_alert"]
    run.summary["retrain_recommended"] = rep["summary"]["retrain_recommended"]
    if html_path and os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as fh:
            run.log({"drift_report_html": wandb.Html(fh.read())})
    try:
        for s in rep["scenarios"]:
            if s["status"] == "ALERT":
                wandb.alert(title="SIMULATED drift alert: %s" % s["name"],
                            text="First alert at %s. Retrain recommended (candidate only; needs "
                                 "quality gate and human approval)." % s["first_alert_window"],
                            level=wandb.AlertLevel.WARN)
    except Exception as exc:
        print("wandb.alert failed:", exc)
    url = run.url
    run.finish()
    return url
