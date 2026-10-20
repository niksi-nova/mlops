"""Unit tests for src/models/tracking.py's git-dirty check and W&B run setup.

The git call and wandb.init/wandb.login calls are mocked (monkeypatched);
no test depends on this repo's actual working-tree state or talks to the
real W&B service. WANDB_MODE=disabled is also set so nothing escapes the
mocks accidentally.
"""

import os

from src.models.tracking import dirty_files, git_is_dirty, start_run

os.environ["WANDB_MODE"] = "disabled"


def mock_git_status(monkeypatch, porcelain_output: str) -> None:
    """Make tracking._run(["git", "status", "--porcelain"]) return
    `porcelain_output` instead of calling real git."""
    monkeypatch.setattr("src.models.tracking._run", lambda cmd: porcelain_output)


def test_dvc_lock_change_is_ignored(monkeypatch):
    mock_git_status(monkeypatch, " M dvc.lock")
    assert dirty_files() == []
    assert git_is_dirty() is False


def test_untracked_models_dir_is_ignored(monkeypatch):
    mock_git_status(monkeypatch, "?? models/")
    assert dirty_files() == []
    assert git_is_dirty() is False


def test_source_file_change_counts_as_dirty(monkeypatch):
    mock_git_status(monkeypatch, " M src/models/train.py")
    assert dirty_files() == ["src/models/train.py"]
    assert git_is_dirty() is True


def test_empty_status_means_clean(monkeypatch):
    mock_git_status(monkeypatch, "")
    assert dirty_files() == []
    assert git_is_dirty() is False


def test_mixed_ignored_and_real_changes_only_reports_the_real_one(monkeypatch):
    mock_git_status(
        monkeypatch,
        " M dvc.lock\n M reports/train_metrics.json\n?? models/\n M params.yaml",
    )
    assert dirty_files() == ["params.yaml"]
    assert git_is_dirty() is True


def test_dot_dvc_path_is_ignored(monkeypatch):
    mock_git_status(monkeypatch, "?? .dvc/tmp/rwlock")
    assert dirty_files() == []
    assert git_is_dirty() is False


def test_renamed_file_uses_the_new_path(monkeypatch):
    mock_git_status(monkeypatch, "R  src/models/old_train.py -> src/models/train.py")
    assert dirty_files() == ["src/models/train.py"]
    assert git_is_dirty() is True


def test_start_run_merges_config_with_lineage_and_falls_back_to_offline(monkeypatch):
    mock_git_status(monkeypatch, "")
    monkeypatch.setattr("src.models.tracking.git_commit_hash", lambda: "abc123")
    monkeypatch.setattr("src.models.tracking.git_branch", lambda: "main")
    monkeypatch.setattr("src.models.tracking.featurize_outputs_md5", lambda: "deadbeef")
    monkeypatch.setattr("src.models.tracking.wandb.login", lambda timeout=0: False)
    monkeypatch.delenv("WANDB_MODE", raising=False)  # test the real fallback, not the "disabled" override

    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return "fake-run"

    monkeypatch.setattr("src.models.tracking.wandb.init", fake_init)

    result = start_run("ridge", job_type="train", config={"alpha": 1.0}, group="baselines-v1")

    assert result == "fake-run"
    assert os.environ["WANDB_MODE"] == "offline"  # fell back since login() returned False
    assert captured["name"] == "ridge"
    assert captured["job_type"] == "train"
    assert captured["group"] == "baselines-v1"
    assert captured["config"]["alpha"] == 1.0
    assert captured["config"]["git_commit"] == "abc123"
    assert captured["config"]["git_branch"] == "main"
    assert captured["config"]["data_processed_md5"] == "deadbeef"
    assert captured["config"]["dataset_version"] == "delhi-data-v2"
    assert captured["config"]["git_dirty"] is False
    os.environ["WANDB_MODE"] = "disabled"  # restore for subsequent tests


def test_ensure_wandb_login_respects_explicit_wandb_mode(monkeypatch):
    """When the caller has already set WANDB_MODE (e.g. "disabled" for
    tests), ensure_wandb_login must not override it even if login() fails."""
    from src.models.tracking import ensure_wandb_login

    monkeypatch.setattr("src.models.tracking.wandb.login", lambda timeout=0: False)
    os.environ["WANDB_MODE"] = "disabled"

    assert ensure_wandb_login() is False
    assert os.environ["WANDB_MODE"] == "disabled"  # untouched, not overwritten to "offline"


def test_start_run_does_not_fall_back_when_logged_in(monkeypatch):
    mock_git_status(monkeypatch, "")
    monkeypatch.setattr("src.models.tracking.wandb.login", lambda timeout=0: True)
    monkeypatch.setattr("src.models.tracking.wandb.init", lambda **kwargs: "fake-run")
    os.environ.pop("WANDB_MODE", None)

    start_run("global_median", job_type="train", config={})

    assert "WANDB_MODE" not in os.environ
    os.environ["WANDB_MODE"] = "disabled"  # restore for subsequent tests
