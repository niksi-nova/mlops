"""Weights & Biases tracking setup shared by src/models/train.py and
src/models/sweep_train.py.

Resolves whether a W&B login is available (falling back to WANDB_MODE=
offline with a warning if not - never printing or logging the API key
itself), and builds the standard lineage info (git commit, branch, dirty
flag, featurize-stage data hash, dataset version) that every run records in
its config so it's clear exactly what code and data produced it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import wandb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DVC_LOCK_PATH = PROJECT_ROOT / "dvc.lock"
DATASET_VERSION = "delhi-data-v2"


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def ensure_wandb_login() -> bool:
    """Check whether a W&B login is available (without prompting or ever
    printing/logging the API key itself). If not - and the caller hasn't
    already set WANDB_MODE explicitly (e.g. to "disabled" for tests) - fall
    back to WANDB_MODE=offline and print a warning. Returns True if logged
    in."""
    logged_in = bool(wandb.login(timeout=0))
    if not logged_in and "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
        print(
            "WARNING: no W&B login available (run `uv run wandb login`); "
            "falling back to WANDB_MODE=offline - runs will be logged locally only."
        )
    return logged_in


def _run(cmd: list[str]) -> str:
    """Run a git command from the project root and return its stdout with
    only the trailing newline removed, or "" if it fails (e.g. not a git
    repo). Uses rstrip("\n"), NOT strip(): `git status --porcelain` lines
    legitimately start with a space (e.g. " M path"), and a plain .strip()
    would eat that leading space off the first line, corrupting the
    fixed-width status-code/path parsing in _porcelain_path()."""
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.rstrip("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_commit_hash() -> str:
    """Current HEAD commit hash, or "" if unavailable."""
    return _run(["git", "rev-parse", "HEAD"])


def git_branch() -> str:
    """Current branch name, or "" if unavailable."""
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


GIT_DIRTY_IGNORED_PREFIXES = ("dvc.lock", "reports/", "models/", "data/", "wandb/", ".dvc/")


def _porcelain_path(line: str) -> str:
    """The path from one `git status --porcelain` line. Status codes
    always occupy the first 2 characters, so the path starts at index 3
    (no stripping of the line itself — that's git's fixed-width format,
    not incidental whitespace). A rename/copy line reads "old -> new";
    we only care about the new path."""
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path


def dirty_files() -> list[str]:
    """Paths from `git status --porcelain` that are NOT under an ignored
    prefix (dvc.lock, reports/, models/, data/, wandb/, .dvc/). Empty
    means the tree is clean (for tagging purposes)."""
    output = _run(["git", "status", "--porcelain"])
    paths = [_porcelain_path(line) for line in output.splitlines()]
    return [path for path in paths if not path.startswith(GIT_DIRTY_IGNORED_PREFIXES)]


def git_is_dirty() -> bool:
    """True if the working tree has uncommitted changes outside of
    dvc.lock, reports/, models/, data/, wandb/ and .dvc/ — those are
    pipeline-internal paths that routinely change during `dvc repro` and
    would otherwise mark every run as dirty."""
    return bool(dirty_files())


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


def lineage_config() -> dict[str, Any]:
    """Git commit/branch/dirty-state, the featurize-stage data hash, and the
    dataset version — everything needed to know exactly what code and data
    produced a run. Intended to be merged into wandb.config."""
    dirty = dirty_files()
    return {
        "git_commit": git_commit_hash(),
        "git_branch": git_branch(),
        "git_dirty": bool(dirty),
        "git_dirty_files": dirty,
        "data_processed_md5": featurize_outputs_md5(),
        "dataset_version": DATASET_VERSION,
    }


def start_run(name: str, job_type: str, config: dict[str, Any], group: str | None = None, tags: list[str] | None = None):
    """Start a W&B run named `name` in params.yaml's train.wandb_project,
    with `config` merged with the standard lineage info (git/data/dataset
    version) in wandb.config. Falls back to offline mode if not logged in."""
    ensure_wandb_login()
    params = load_params()
    full_config = {**config, **lineage_config()}
    return wandb.init(
        project=params["train"]["wandb_project"],
        name=name,
        job_type=job_type,
        group=group,
        tags=tags,
        config=full_config,
    )
