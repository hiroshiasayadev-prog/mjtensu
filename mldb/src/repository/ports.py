"""Minimal physical filesystem I/O port for MLDB repository consumers.

This module fixes only path-oriented physical I/O needed by later resolution,
persistence, sealing, and entity-query implementations. It deliberately does not
own canonical path derivation, entity parsing, semantic serialization, transactions,
locking, caching, watching, temporary-file policy, SQLite APIs, or executable loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FilesystemPort(Protocol):
    """Consumer-oriented filesystem boundary for canonical MLDB repository access.

    Paths supplied to this port are already selected by repository-layout or feature
    logic. The port does not interpret MLDB entity kinds, IDs, metadata, lifecycle,
    schemas, or Run semantics.
    """

    def file_exists(self, path: Path) -> bool:
        """Return whether ``path`` exists as a regular file."""

        ...

    def directory_exists(self, path: Path) -> bool:
        """Return whether ``path`` exists as a directory."""

        ...

    def read_text(self, path: Path, *, encoding: str) -> str:
        """Read one text file using the caller-selected codec.

        Encoding choice belongs to the applicable file-format/serialization contract,
        not to the filesystem adapter.
        """

        ...

    def read_bytes(self, path: Path) -> bytes:
        """Read the exact bytes of one file."""

        ...

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        """Return immediate directory entries in deterministic name order.

        The result contains concrete child :class:`~pathlib.Path` values and does not
        recurse, glob, parse entity identities, or filter by MLDB kind. For the same
        directory contents, adapters must order entries by ``Path.name`` using normal
        Python string ordering so entity-query behavior does not depend on host
        filesystem enumeration order.
        """

        ...

    def ensure_directory(self, path: Path) -> None:
        """Ensure ``path`` exists as a directory, creating required parents as needed.

        Successful return only establishes directory existence. Ownership, lifecycle,
        permissions policy, and cleanup remain outside this port contract.
        """

        ...

    def replace_text(self, path: Path, text: str, *, encoding: str) -> None:
        """Commit complete text using the caller-selected codec.

        Encoding choice belongs to the applicable file-format/serialization contract.
        On successful return, consumers must not observe a partially written final
        destination. Temporary filenames, flushing, fsync behavior, rename flags, and
        other OS-specific mechanics are adapter implementation details. Parent
        directory creation is not implied; callers use :meth:`ensure_directory` when
        required.
        """

        ...

    def replace_bytes(self, path: Path, data: bytes) -> None:
        """Commit complete bytes as the destination file's accepted contents.

        Successful return has the same complete-replacement guarantee as
        :meth:`replace_text`. Serialization and interpretation of ``data`` belong to
        the calling feature rather than this filesystem boundary.
        """

        ...
