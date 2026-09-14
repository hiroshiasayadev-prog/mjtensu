"""Internal committed Git source-byte boundary for MLDB v2 source pinning."""

from __future__ import annotations

import re as _re
import stat as _stat
import subprocess as _subprocess
from pathlib import Path
from typing import Iterable

from mldb_v2.src.storage._paths import _validate_safe_relative_path


_COMMIT_RE = _re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", _re.ASCII)


def _repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    return root


def _run_git(root: Path, *args: str) -> _subprocess.CompletedProcess[bytes]:
    try:
        return _subprocess.run(
            ["git", "-C", str(root), *args],
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable is not available") from exc


def _validate_commit_id(value: object) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("commit must be a full lowercase Git object id")
    return value


def _require_commit(root: Path, commit: str) -> None:
    validated = _validate_commit_id(commit)
    result = _run_git(root, "cat-file", "-e", f"{validated}^{{commit}}")
    if result.returncode != 0:
        raise FileNotFoundError(f"Git commit not found: {validated}")


def _current_repository_commit(repo_root: str | Path) -> str:
    root = _repo_root(repo_root)
    result = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    if result.returncode != 0:
        raise FileNotFoundError("repository has no current commit")
    try:
        commit = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("git returned a non-ASCII commit id") from exc
    return _validate_commit_id(commit)


def _read_committed_source_bytes(
    repo_root: str | Path,
    *,
    commit: str,
    path: str,
) -> bytes:
    root = _repo_root(repo_root)
    _require_commit(root, commit)
    relative = _validate_safe_relative_path(path, label="repository source path")
    object_spec = f"{commit}:{relative}"

    object_type = _run_git(root, "cat-file", "-t", object_spec)
    if object_type.returncode != 0 or object_type.stdout.strip() != b"blob":
        raise FileNotFoundError(f"committed source file not found: {relative}")
    content = _run_git(root, "cat-file", "blob", object_spec)
    if content.returncode != 0:
        raise FileNotFoundError(f"committed source file not found: {relative}")
    return content.stdout


def _working_tree_source_path(root: Path, relative: str) -> Path:
    lexical_target = root.joinpath(*relative.split("/"))
    try:
        resolved = lexical_target.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"working-tree source file not found: {relative}") from exc
    if not resolved.is_relative_to(root):
        raise ValueError("working-tree source path escapes repository root")
    if not _stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("working-tree source path must be a regular file")
    return resolved


def _working_tree_source_matches_commit(
    repo_root: str | Path,
    *,
    commit: str,
    path: str,
) -> bool:
    root = _repo_root(repo_root)
    relative = _validate_safe_relative_path(path, label="repository source path")
    committed = _read_committed_source_bytes(root, commit=commit, path=relative)
    try:
        working = _working_tree_source_path(root, relative).read_bytes()
    except FileNotFoundError:
        return False
    return working == committed


def _selected_source_mismatches(
    repo_root: str | Path,
    *,
    commit: str,
    paths: Iterable[str],
) -> list[str]:
    root = _repo_root(repo_root)
    _require_commit(root, commit)
    mismatches: list[str] = []
    for path in paths:
        relative = _validate_safe_relative_path(path, label="repository source path")
        if not _working_tree_source_matches_commit(root, commit=commit, path=relative):
            mismatches.append(relative)
    return mismatches
