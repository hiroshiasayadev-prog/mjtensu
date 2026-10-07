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


@dataclass(frozen=True)
class RuntimeImageMetadata:
    version: int
    profile: str
    state: str
    recipe_version: str
    repository: str
    tag: str
    base_image: str
    digest: str | None
    error: str | None
    requested_at: str
    updated_at: str

    @property
    def image_ref(self) -> str | None:
        if self.state != "READY" or self.digest is None:
            return None
        return f"{self.repository}@{self.digest}"


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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runtime_images (
                    version INTEGER NOT NULL,
                    profile TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('BUILDING', 'READY', 'FAILED', 'MISSING')),
                    recipe_version TEXT NOT NULL,
                    repository TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    base_image TEXT NOT NULL,
                    digest TEXT,
                    error TEXT,
                    requested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(version, profile),
                    FOREIGN KEY(version) REFERENCES registry_versions(version)
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

    def get_image(self, version: int, profile: str) -> RuntimeImageMetadata | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runtime_images WHERE version = ? AND profile = ?",
                (version, profile),
            ).fetchone()
        return None if row is None else self._image_metadata(row)

    def ensure_image(
        self,
        version: int,
        *,
        profile: str,
        recipe_version: str,
        repository: str,
        tag: str,
        base_image: str,
    ) -> RuntimeImageMetadata:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            version_row = conn.execute(
                "SELECT version FROM registry_versions WHERE version = ?", (version,)
            ).fetchone()
            if version_row is None:
                raise RegistryNotFound(f"runtime registry version {version} does not exist")
            row = conn.execute(
                "SELECT * FROM runtime_images WHERE version = ? AND profile = ?",
                (version, profile),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO runtime_images
                       (version, profile, state, recipe_version, repository, tag, base_image,
                        digest, error, requested_at, updated_at)
                       VALUES (?, ?, 'BUILDING', ?, ?, ?, ?, NULL, NULL, ?, ?)""",
                    (
                        version, profile, recipe_version, repository, tag, base_image, now, now,
                    ),
                )
            elif row["state"] == "MISSING":
                conn.execute(
                    """UPDATE runtime_images
                       SET state = 'BUILDING', digest = NULL, error = NULL,
                           requested_at = ?, updated_at = ?
                       WHERE version = ? AND profile = ?""",
                    (now, now, version, profile),
                )
            row = conn.execute(
                "SELECT * FROM runtime_images WHERE version = ? AND profile = ?",
                (version, profile),
            ).fetchone()
            conn.commit()
            assert row is not None
            return self._image_metadata(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def next_image_build(self) -> RuntimeImageMetadata | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM runtime_images
                   WHERE state = 'BUILDING'
                   ORDER BY requested_at ASC, version ASC
                   LIMIT 1"""
            ).fetchone()
        return None if row is None else self._image_metadata(row)

    def mark_image_ready(self, version: int, profile: str, digest: str) -> RuntimeImageMetadata:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE runtime_images
                   SET state = 'READY', digest = ?, error = NULL, updated_at = ?
                   WHERE version = ? AND profile = ?""",
                (digest, now, version, profile),
            )
        image = self.get_image(version, profile)
        if image is None:
            raise RegistryNotFound(
                f"runtime image {profile!r} for registry version {version} does not exist"
            )
        return image

    def mark_image_failed(self, version: int, profile: str, error: str) -> RuntimeImageMetadata:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE runtime_images
                   SET state = 'FAILED', digest = NULL, error = ?, updated_at = ?
                   WHERE version = ? AND profile = ?""",
                (error, now, version, profile),
            )
        image = self.get_image(version, profile)
        if image is None:
            raise RegistryNotFound(
                f"runtime image {profile!r} for registry version {version} does not exist"
            )
        return image

    def mark_image_missing(self, version: int, profile: str) -> RuntimeImageMetadata:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE runtime_images
                   SET state = 'MISSING', digest = NULL, error = NULL, updated_at = ?
                   WHERE version = ? AND profile = ?""",
                (now, version, profile),
            )
        image = self.get_image(version, profile)
        if image is None:
            raise RegistryNotFound(
                f"runtime image {profile!r} for registry version {version} does not exist"
            )
        return image

    def touch_image(self, version: int, profile: str) -> RuntimeImageMetadata:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE runtime_images
                   SET updated_at = ?
                   WHERE version = ? AND profile = ? AND state = 'READY'""",
                (now, version, profile),
            )
        image = self.get_image(version, profile)
        if image is None:
            raise RegistryNotFound(
                f"runtime image {profile!r} for registry version {version} does not exist"
            )
        return image

    def ready_images(self, profile: str) -> tuple[RuntimeImageMetadata, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM runtime_images
                   WHERE profile = ? AND state = 'READY'
                   ORDER BY updated_at DESC, version DESC""",
                (profile,),
            ).fetchall()
        return tuple(self._image_metadata(row) for row in rows)

    @staticmethod
    def _metadata(row: sqlite3.Row) -> SnapshotMetadata:
        return SnapshotMetadata(**{key: row[key] for key in SnapshotMetadata.__dataclass_fields__})

    @staticmethod
    def _image_metadata(row: sqlite3.Row) -> RuntimeImageMetadata:
        return RuntimeImageMetadata(
            **{key: row[key] for key in RuntimeImageMetadata.__dataclass_fields__}
        )
