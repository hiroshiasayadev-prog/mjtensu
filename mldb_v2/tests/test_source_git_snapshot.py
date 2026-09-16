import subprocess
from pathlib import Path

import pytest

from mldb_v2.src.source.git_snapshot import (
    _current_repository_commit,
    _read_committed_source_bytes,
    _selected_source_mismatches,
    _working_tree_source_matches_commit,
)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "MLDB Test")
    _git(repo, "config", "user.email", "mldb@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "selected.txt").write_bytes(b"committed\n")
    (repo / "other.txt").write_bytes(b"other\n")
    _git(repo, "add", "selected.txt", "other.txt")
    _git(repo, "commit", "-m", "initial")
    commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    return repo, commit


def test_current_commit_and_exact_committed_byte_read(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    assert _current_repository_commit(repo) == commit
    assert _read_committed_source_bytes(repo, commit=commit, path="selected.txt") == b"committed\n"


def test_dirty_selected_file_is_detected_but_unrelated_dirty_file_is_ignored(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    (repo / "selected.txt").write_bytes(b"dirty\n")
    assert not _working_tree_source_matches_commit(repo, commit=commit, path="selected.txt")
    assert _selected_source_mismatches(repo, commit=commit, paths=["selected.txt"]) == ["selected.txt"]

    (repo / "selected.txt").write_bytes(b"committed\n")
    (repo / "other.txt").write_bytes(b"unrelated dirty\n")
    assert _working_tree_source_matches_commit(repo, commit=commit, path="selected.txt")
    assert _selected_source_mismatches(repo, commit=commit, paths=["selected.txt"]) == []


def test_missing_commit_or_path_fails(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    with pytest.raises(FileNotFoundError):
        _read_committed_source_bytes(repo, commit="0" * 40, path="selected.txt")
    with pytest.raises(FileNotFoundError):
        _read_committed_source_bytes(repo, commit=commit, path="missing.txt")


@pytest.mark.parametrize(
    "path",
    ["../secret", "dir/../../secret", "/absolute/path", r"C:\absolute\path", "C:/absolute/path", "dir/./file.py", r"dir\file.py"],
)
def test_repository_source_path_escape_and_platform_ambiguity_are_rejected(
    tmp_path: Path,
    path: str,
) -> None:
    repo, commit = _make_repo(tmp_path)
    with pytest.raises(ValueError):
        _read_committed_source_bytes(repo, commit=commit, path=path)


def test_git_helpers_do_not_stage_commit_or_mutate_repository(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    head_before = _git(repo, "rev-parse", "HEAD")
    index_before = _git(repo, "diff", "--cached", "--name-only")
    status_before = _git(repo, "status", "--porcelain")

    assert _read_committed_source_bytes(repo, commit=commit, path="selected.txt") == b"committed\n"
    assert _selected_source_mismatches(repo, commit=commit, paths=["selected.txt"]) == []

    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "diff", "--cached", "--name-only") == index_before
    assert _git(repo, "status", "--porcelain") == status_before
