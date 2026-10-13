"""MLflow tracking setup shared by src/models/train.py and
src/models/tune_lightgbm.py.

Loads credentials from .env (never logged or printed) and resolves the
tracking URI: MLFLOW_TRACKING_URI from the environment if set, otherwise a
local ./mlruns folder with a warning. Also builds the standard run tags
(git commit, branch, dirty flag, featurize-stage data hash, dataset
version) so every run records exactly what code and data produced it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DVC_LOCK_PATH = PROJECT_ROOT / "dvc.lock"
DATASET_VERSION = "data-v2"


def configure_tracking() -> str:
    """Load .env, then point MLflow at MLFLOW_TRACKING_URI if set, else a
    local ./mlruns folder (with a warning). Returns the tracking URI used.

    Credentials (MLFLOW_TRACKING_USERNAME/PASSWORD) are read by MLflow
    itself from the environment; this function never reads, prints or logs
    them."""
    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = f"file:{PROJECT_ROOT / 'mlruns'}"
        print(
            "WARNING: MLFLOW_TRACKING_URI is not set; falling back to a "
            f"local tracking store at {tracking_uri}"
        )

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _run(cmd: list[str]) -> str:
    """Run a git command from the project root and return stripped stdout,
    or "" if it fails (e.g. not a git repo)."""
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_commit_hash() -> str:
    """Current HEAD commit hash, or "" if unavailable."""
    return _run(["git", "rev-parse", "HEAD"])


def git_branch() -> str:
    """Current branch name, or "" if unavailable."""
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


GIT_DIRTY_IGNORED_PREFIXES = ("dvc.lock", "reports/", "models/", "data/", "mlruns/")


def git_is_dirty() -> bool:
    """True if the working tree has uncommitted changes, ignoring changes
    under dvc.lock, reports/, models/, data/ and mlruns/ — those are
    pipeline outputs that routinely change during `dvc repro` and would
    otherwise mark every run as dirty."""
    output = _run(["git", "status", "--porcelain"])
    for line in output.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:  # renames: "old -> new"
            path = path.split(" -> ")[-1].strip().strip('"')
        if not path.startswith(GIT_DIRTY_IGNORED_PREFIXES):
            return True
    return False


def featurize_outputs_md5(dvc_lock_path: Path = DVC_LOCK_PATH) -> str:
    """A single hash summarising the featurize stage's outputs
    (data/processed/*) as recorded in dvc.lock, by hashing the sorted
    per-file md5s together. Returns "" if dvc.lock or the featurize stage
    is not found."""
    if not dvc_lock_path.exists():
        return ""
    lock = yaml.safe_load(dvc_lock_path.read_text())
    featurize = lock.get("stages", {}).get("featurize")
    if not featurize:
        return ""
    file_hashes = sorted(out["md5"] for out in featurize.get("outs", []) if "md5" in out)
    if not file_hashes:
        return ""
    return hashlib.md5("".join(file_hashes).encode()).hexdigest()


def run_tags() -> dict[str, str]:
    """Standard tags to attach to every MLflow run: git commit, branch,
    dirty flag, the featurize-stage data hash, and the dataset version."""
    return {
        "git_commit": git_commit_hash(),
        "git_branch": git_branch(),
        "git_dirty": str(git_is_dirty()),
        "data_processed_md5": featurize_outputs_md5(),
        "dataset_version": DATASET_VERSION,
    }
