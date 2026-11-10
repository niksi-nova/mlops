"""Export the W&B registry's production-alias model to a local directory.

Lets the Docker image run in a "baked in" mode: build once with W&B access
to pull the model, then run the container with MODEL_PATH pointing at the
exported file and no W&B credentials needed at all.

Usage:
    uv run python -m scripts.export_production_model [--alias ALIAS] [--out-dir DIR]

Writes:
    <out-dir>/model.joblib    the raw model (as registry.load_model's
                              direct-path branch expects: joblib.load() ->
                              a raw estimator, not the ProductionModel wrapper)
    <out-dir>/metadata.json   {"alias", "version", "model_name", "git_commit"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.models.registry import REGISTRY_REPORT_PATH, load_model

DEFAULT_OUT_DIR = Path("model_export")


def _registry_git_commit() -> str | None:
    if not REGISTRY_REPORT_PATH.exists():
        return None
    try:
        record = json.loads(REGISTRY_REPORT_PATH.read_text())
        return record.get("metadata", {}).get("git_commit")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a registry model for MODEL_PATH-based loading")
    parser.add_argument("--alias", default="production", help="Registry alias to export (default: production)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory (default: ./model_export)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapped = load_model(args.alias)

    model_path = out_dir / "model.joblib"
    joblib.dump(wrapped.model, model_path)

    metadata = {
        "alias": wrapped.alias,
        "version": wrapped.version,
        "model_name": wrapped.model_name,
        "git_commit": _registry_git_commit(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Exported '{args.alias}' ({wrapped.model_name} {wrapped.version}) -> {model_path}")
    print(f"Wrote {out_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
