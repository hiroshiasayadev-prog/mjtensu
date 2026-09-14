"""Private cross-process file-lock primitive for repository-local coordination."""

from __future__ import annotations

import errno
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[path] = lock
        return lock


def _lock_path(lock_root: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return lock_root / f"{digest}.lock"


def _ensure_lock_byte(handle: object) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)


def _acquire_os_lock(handle: object) -> None:
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                time.sleep(0.01)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_os_lock(handle: object) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _process_file_lock(*, lock_root: Path, key: str) -> Iterator[None]:
    """Serialize one exact key across threads/processes without path-existence ownership."""
    lock_root.mkdir(parents=True, exist_ok=True)
    path = _lock_path(lock_root, key)
    thread_lock = _thread_lock(path)

    with thread_lock:
        with path.open("a+b") as handle:
            _ensure_lock_byte(handle)
            _acquire_os_lock(handle)
            try:
                yield
            finally:
                _release_os_lock(handle)
