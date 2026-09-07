"""Local filesystem adapter for the MLDB repository filesystem port."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class LocalFilesystem:
    """Minimal local-disk implementation of the repository filesystem boundary."""

    def file_exists(self, path: Path) -> bool:
        return path.is_file()

    def directory_exists(self, path: Path) -> bool:
        return path.is_dir()

    def read_text(self, path: Path, *, encoding: str) -> str:
        return path.read_text(encoding=encoding)

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        return tuple(sorted(path.iterdir(), key=lambda child: child.name))

    def ensure_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def replace_text(self, path: Path, text: str, *, encoding: str) -> None:
        self._replace_bytes(path, text.encode(encoding))

    def replace_bytes(self, path: Path, data: bytes) -> None:
        self._replace_bytes(path, data)

    @staticmethod
    def _replace_bytes(path: Path, data: bytes) -> None:
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(temporary_path, path)
        except BaseException:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
