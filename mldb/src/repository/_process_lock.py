"""Implementation-private cross-process repository lock."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def repository_process_lock(
    repository_root: Path,
    name: str,
    *,
    timeout_seconds: float = 30.0,
) -> Iterator[None]:
    """Serialize short local-repository mutations across Python processes."""
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise ValueError("lock name must be one safe path component")
    lock_dir = repository_root / ".local" / "mldb" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{name}.lock"
    deadline = time.monotonic() + float(timeout_seconds)
    with path.open("a+b", buffering=0) as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
        acquired = False
        try:
            while not acquired:
                try:
                    _lock_one_byte(stream)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out acquiring repository lock: {name}"
                        )
                    time.sleep(0.05)
            yield
        finally:
            if acquired:
                _unlock_one_byte(stream)


def _lock_one_byte(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_one_byte(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
