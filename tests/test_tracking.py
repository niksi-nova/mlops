"""Unit tests for src/models/tracking.py's git-dirty check.

The git call itself is mocked (monkeypatched); no test depends on this
repo's actual working-tree state.
"""

from src.models.tracking import dirty_files, git_is_dirty


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
