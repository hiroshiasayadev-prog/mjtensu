"""Implementation-private Controller orchestration coordination."""

from collections.abc import Iterator as _Iterator
from contextlib import contextmanager as _contextmanager
from threading import RLock as _RLock


_ORCHESTRATION_LOCK = _RLock()


@_contextmanager
def orchestration_exclusion() -> _Iterator[None]:
    """Enter the Controller-wide process-local orchestration critical section.

    The single reentrant lock is intentionally private. Callers retain ownership of
    all canonical and Queue reads or mutations performed while the exclusion is held.
    """

    with _ORCHESTRATION_LOCK:
        yield
