"""Server-side candidate validation for runtime-registry SET."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile


class UvSyncValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UvSyncValidator:
    uv_binary: str = "uv"
    timeout_seconds: int = 1800
    cache_dir: str | None = None

    def validate(self, pyproject_toml: bytes, uv_lock: bytes) -> None:
        try:
            pyproject_toml.decode("utf-8")
            uv_lock.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UvSyncValidationError("pyproject.toml and uv.lock must be UTF-8") from exc

        with tempfile.TemporaryDirectory(prefix="mldb-runtime-registry-") as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_bytes(pyproject_toml)
            (root / "uv.lock").write_bytes(uv_lock)
            env = os.environ.copy()
            if self.cache_dir:
                env["UV_CACHE_DIR"] = self.cache_dir
            command = [
                self.uv_binary, "sync", "--locked", "--no-install-project",
                "--project", str(root),
            ]
            try:
                completed = subprocess.run(
                    command, cwd=root, env=env, capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise UvSyncValidationError(f"uv sync validation could not complete: {exc}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "uv sync failed").strip()
                if len(detail) > 4000:
                    detail = detail[-4000:]
                raise UvSyncValidationError(f"uv sync --locked failed: {detail}")
