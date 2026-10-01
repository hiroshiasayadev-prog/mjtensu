"""Versioned global runtime-registry storage primitives."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sqlite3
from typing import Protocol


class ObjectStore(Protocol):
    def read_bytes(self, uri: str) -> bytes: ...
    def publish_bytes_immutable(self, uri: str, data: bytes) -> None: ...


class RegistryNotFound(KeyError):
    pass


class RegistryCorruption(RuntimeError):
    pass


@dataclass(frozen=True)
class SnapshotMetadata:
    version: int
    created_at: str
    snapshot_sha256: str
    pyproject_sha256: str
    uv_lock_sha256: str
    pyproject_uri: str
    uv_lock_uri: str


@dataclass(frozen=True)
class RuntimeSnapshot:
    metadata: SnapshotMetadata
    pyproject_toml: bytes
    uv_lock: bytes


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def snapshot_sha256(pyproject_toml: bytes, uv_lock: bytes) -> str:
    digest = sha256(b"mldb-runtime-registry-v1\0")
    for payload in (pyproject_toml, uv_lock):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class RegistryStore:
    def __init__(self, db_path: str | Path, object_store: ObjectStore, bucket: str, prefix: str = "mldb-runtime-registry") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._object_store = object_store
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registry_versions (
                    version INTEGER PRIMARY KEY CHECK(version > 0),
                    created_at TEXT NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    pyproject_sha256 TEXT NOT NULL,
                    uv_lock_sha256 TEXT NOT NULL,
                    pyproject_uri TEXT NOT NULL,
                    uv_lock_uri TEXT NOT NULL
                )
            """)

    def publish(self, pyproject_toml: bytes, uv_lock: bytes) -> tuple[SnapshotMetadata, bool]:
        snap_hash = snapshot_sha256(pyproject_toml, uv_lock)
        base = f"s3://{self._bucket}/{self._prefix}/snapshots/{snap_hash}"
        pyproject_uri = f"{base}/pyproject.toml"
        uv_lock_uri = f"{base}/uv.lock"
        self._object_store.publish_bytes_immutable(pyproject_uri, pyproject_toml)
        self._object_store.publish_bytes_immutable(uv_lock_uri, uv_lock)

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                "SELECT * FROM registry_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if latest is not None and latest["snapshot_sha256"] == snap_hash:
                conn.commit()
                return self._metadata(latest), False
            version = 1 if latest is None else int(latest["version"]) + 1
            created_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO registry_versions
                   (version, created_at, snapshot_sha256, pyproject_sha256, uv_lock_sha256,
                    pyproject_uri, uv_lock_uri)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (version, created_at, snap_hash, _sha256(pyproject_toml), _sha256(uv_lock),
                 pyproject_uri, uv_lock_uri),
            )
            row = conn.execute("SELECT * FROM registry_versions WHERE version = ?", (version,)).fetchone()
            conn.commit()
            assert row is not None
            return self._metadata(row), True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get(self, version: int | None = None) -> RuntimeSnapshot:
        with self._connect() as conn:
            if version is None:
                row = conn.execute("SELECT * FROM registry_versions ORDER BY version DESC LIMIT 1").fetchone()
            else:
                row = conn.execute("SELECT * FROM registry_versions WHERE version = ?", (version,)).fetchone()
        if row is None:
            raise RegistryNotFound("runtime registry is empty" if version is None else f"runtime registry version {version} does not exist")
        meta = self._metadata(row)
        pyproject = self._object_store.read_bytes(meta.pyproject_uri)
        lock = self._object_store.read_bytes(meta.uv_lock_uri)
        if _sha256(pyproject) != meta.pyproject_sha256 or _sha256(lock) != meta.uv_lock_sha256:
            raise RegistryCorruption(f"runtime registry version {meta.version} object hash mismatch")
        if snapshot_sha256(pyproject, lock) != meta.snapshot_sha256:
            raise RegistryCorruption(f"runtime registry version {meta.version} snapshot hash mismatch")
        return RuntimeSnapshot(meta, pyproject, lock)

    @staticmethod
    def _metadata(row: sqlite3.Row) -> SnapshotMetadata:
        return SnapshotMetadata(**{key: row[key] for key in SnapshotMetadata.__dataclass_fields__})
