"""Global runtime-registry client and precision5820 managed-runtime materialization."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


DEFAULT_RUNTIME_REGISTRY_URL = "https://mjtensu-dev.home.arpa/mldb-runtime-registry/"
_NAME_RE = re.compile(r"[-_.]+")
_EXACT_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s]+)$")


class RuntimeRegistryError(RuntimeError):
    """Fail-closed runtime-registry resolution/materialization failure."""


def _canonical_name(value: str) -> str:
    return _NAME_RE.sub("-", value).lower()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_sha256(pyproject_toml: bytes, uv_lock: bytes) -> str:
    digest = hashlib.sha256(b"mldb-runtime-registry-v1\0")
    for payload in (pyproject_toml, uv_lock):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeRegistrySnapshot:
    version: int
    snapshot_sha256: str
    pyproject_toml: bytes
    uv_lock: bytes


@dataclass(frozen=True)
class RuntimePackage:
    name: str
    version: str
    index_url: str


@dataclass(frozen=True)
class RuntimeConvergenceReport:
    version: int
    changed: bool
    installed: tuple[str, ...]
    removed: tuple[str, ...]


class RuntimeRegistryClient:
    def __init__(
        self,
        base_url: str = DEFAULT_RUNTIME_REGISTRY_URL,
        *,
        timeout_seconds: int = 30,
        opener: Callable[..., object] | None = None,
        ca_bundle: str | Path | None = None,
    ) -> None:
        if type(base_url) is not str or not base_url.strip():
            raise ValueError("runtime registry URL must be a non-empty string")
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout_seconds = timeout_seconds
        self._opener = opener
        self._ssl_context: ssl.SSLContext | None = None
        if ca_bundle is not None:
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=str(ca_bundle))
            self._ssl_context = context

    def get(self, version: int | None = None) -> RuntimeRegistrySnapshot:
        if version is not None and (type(version) is not int or version <= 0):
            raise ValueError("runtime registry version must be a positive integer")
        url = self._base_url
        if version is not None:
            url += "?" + urllib.parse.urlencode({"version": version})
        try:
            if self._opener is not None:
                response = self._opener(url, timeout=self._timeout_seconds)
            else:
                response = urllib.request.urlopen(
                    url, timeout=self._timeout_seconds, context=self._ssl_context
                )
            with contextlib.closing(response):
                payload = json.loads(response.read())
        except Exception as error:
            raise RuntimeRegistryError(
                f"runtime registry request failed for version {version!r}"
            ) from error
        if type(payload) is not dict:
            raise RuntimeRegistryError("runtime registry response must be an object")
        try:
            actual_version = payload["version"]
            snapshot_hash = payload["snapshot_sha256"]
            pyproject = payload["pyproject_toml"].encode("utf-8")
            uv_lock = payload["uv_lock"].encode("utf-8")
            pyproject_hash = payload["pyproject_sha256"]
            lock_hash = payload["uv_lock_sha256"]
        except (KeyError, AttributeError) as error:
            raise RuntimeRegistryError("runtime registry response is incomplete") from error
        if type(actual_version) is not int or actual_version <= 0:
            raise RuntimeRegistryError("runtime registry response version is invalid")
        if version is not None and actual_version != version:
            raise RuntimeRegistryError("runtime registry returned the wrong immutable version")
        if (
            type(snapshot_hash) is not str
            or type(pyproject_hash) is not str
            or type(lock_hash) is not str
            or _sha256(pyproject) != pyproject_hash
            or _sha256(uv_lock) != lock_hash
            or _snapshot_sha256(pyproject, uv_lock) != snapshot_hash
        ):
            raise RuntimeRegistryError("runtime registry snapshot hash validation failed")
        snapshot = RuntimeRegistrySnapshot(
            version=actual_version,
            snapshot_sha256=snapshot_hash,
            pyproject_toml=pyproject,
            uv_lock=uv_lock,
        )
        _runtime_packages(snapshot)
        return snapshot

    def latest_version(self) -> int:
        return self.get().version


def _runtime_packages(snapshot: RuntimeRegistrySnapshot) -> dict[str, RuntimePackage]:
    try:
        project = tomllib.loads(snapshot.pyproject_toml.decode("utf-8"))
        lock = tomllib.loads(snapshot.uv_lock.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeRegistryError("runtime registry snapshot TOML is invalid") from error

    packages: dict[str, RuntimePackage] = {}
    raw_packages = lock.get("package")
    if type(raw_packages) is not list:
        raise RuntimeRegistryError("runtime registry lock has no package list")
    for raw in raw_packages:
        if type(raw) is not dict:
            raise RuntimeRegistryError("runtime registry lock package is malformed")
        source = raw.get("source")
        if type(source) is dict and "virtual" in source:
            continue
        name = raw.get("name")
        version = raw.get("version")
        index_url = source.get("registry") if type(source) is dict else None
        if (
            type(name) is not str
            or type(version) is not str
            or type(index_url) is not str
            or not index_url
        ):
            raise RuntimeRegistryError(
                "runtime registry currently supports only versioned registry packages"
            )
        canonical = _canonical_name(name)
        if canonical in packages:
            raise RuntimeRegistryError("runtime registry lock contains duplicate package names")
        packages[canonical] = RuntimePackage(
            name=name,
            version=version,
            index_url=index_url,
        )

    project_table = project.get("project")
    dependencies = project_table.get("dependencies") if type(project_table) is dict else None
    if type(dependencies) is not list:
        raise RuntimeRegistryError("runtime registry pyproject dependencies are malformed")
    for requirement in dependencies:
        if type(requirement) is not str:
            raise RuntimeRegistryError("runtime registry dependency is not a string")
        match = _EXACT_REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            raise RuntimeRegistryError(
                "runtime registry direct dependencies must use exact == pins"
            )
        name, version = match.groups()
        package = packages.get(_canonical_name(name))
        if package is None or package.version != version:
            raise RuntimeRegistryError(
                "runtime registry direct pin does not match resolved lock metadata"
            )
    return packages


_PROBE = """import importlib.metadata as m, json, pathlib, re, sys
canon=lambda s: re.sub(r"[-_.]+", "-", s).lower()
prefix=pathlib.Path(sys.prefix).resolve()
out={}
for d in m.distributions():
    n=d.metadata.get("Name")
    if not n:
        continue
    try:
        p=str(d.locate_file(""))
        local=pathlib.Path(p).resolve().is_relative_to(prefix)
    except Exception:
        p=""
        local=False
    key=canon(n)
    if key not in out or local:
        out[key]={"version":d.version,"path":p}
print(json.dumps(out, sort_keys=True))
"""


class ManagedRuntimeMaterializer:
    """One reusable, system-site-packages venv owned by one reference worker."""

    def __init__(
        self,
        *,
        root: str | Path,
        registry: RuntimeRegistryClient,
        uv_binary: str = "/usr/local/bin/uv",
        base_python: str | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.root = Path(root)
        self.registry = registry
        self.uv_binary = uv_binary
        self.base_python = base_python or sys.executable
        self._run = command_runner
        self.venv = self.root / "venv"
        self.marker = self.root / "current.json"
        self.lock = self.root / "runtime.lock"
        self.snapshots = self.root / "snapshots"

    @property
    def managed_python(self) -> Path:
        return self.venv / "bin" / "python"

    def _snapshot_path(self, version: int) -> Path:
        return self.snapshots / f"v{version:08d}.json"

    def _write_atomic_json(self, path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    def _cached_snapshot(self, version: int) -> RuntimeRegistrySnapshot | None:
        path = self._snapshot_path(version)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            snapshot = RuntimeRegistrySnapshot(
                version=value["version"],
                snapshot_sha256=value["snapshot_sha256"],
                pyproject_toml=value["pyproject_toml"].encode("utf-8"),
                uv_lock=value["uv_lock"].encode("utf-8"),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, AttributeError) as error:
            raise RuntimeRegistryError("cached runtime registry snapshot is corrupt") from error
        if snapshot.version != version:
            raise RuntimeRegistryError("cached runtime registry snapshot version mismatch")
        if _snapshot_sha256(snapshot.pyproject_toml, snapshot.uv_lock) != snapshot.snapshot_sha256:
            raise RuntimeRegistryError("cached runtime registry snapshot hash mismatch")
        _runtime_packages(snapshot)
        return snapshot

    def _snapshot(self, version: int) -> RuntimeRegistrySnapshot:
        cached = self._cached_snapshot(version)
        if cached is not None:
            return cached
        snapshot = self.registry.get(version)
        self._write_atomic_json(
            self._snapshot_path(version),
            {
                "version": snapshot.version,
                "snapshot_sha256": snapshot.snapshot_sha256,
                "pyproject_toml": snapshot.pyproject_toml.decode("utf-8"),
                "uv_lock": snapshot.uv_lock.decode("utf-8"),
            },
        )
        return snapshot

    def _marker(self) -> dict[str, object] | None:
        if not self.marker.is_file():
            return None
        try:
            value = json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeRegistryError("managed runtime marker is corrupt") from error
        if (
            type(value) is not dict
            or type(value.get("version")) is not int
            or value["version"] <= 0
            or type(value.get("snapshot_sha256")) is not str
        ):
            raise RuntimeRegistryError("managed runtime marker is invalid")
        return value

    def _ensure_venv(self) -> None:
        if self.managed_python.is_file():
            config = self.venv / "pyvenv.cfg"
            if not config.is_file() or "include-system-site-packages = true" not in config.read_text(
                encoding="utf-8"
            ).lower():
                raise RuntimeRegistryError(
                    "managed runtime venv is not system-site-packages enabled"
                )
            return
        if self.venv.exists():
            raise RuntimeRegistryError("managed runtime venv is incomplete")
        self.root.mkdir(parents=True, exist_ok=True)
        completed = self._run(
            [
                self.base_python,
                "-m",
                "venv",
                "--system-site-packages",
                str(self.venv),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not self.managed_python.is_file():
            detail = (completed.stderr or completed.stdout or "venv creation failed").strip()
            raise RuntimeRegistryError(f"managed runtime venv creation failed: {detail[-2000:]}")

    def _installed(self) -> dict[str, dict[str, str]]:
        completed = self._run(
            [str(self.managed_python), "-c", _PROBE],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "distribution probe failed").strip()
            raise RuntimeRegistryError(
                f"managed runtime distribution probe failed: {detail[-2000:]}"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeRegistryError("managed runtime distribution probe was malformed") from error
        if type(value) is not dict:
            raise RuntimeRegistryError("managed runtime distribution probe was malformed")
        return value

    def _uv(self, arguments: Sequence[str]) -> None:
        completed = self._run(
            [self.uv_binary, *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "uv failed").strip()
            raise RuntimeRegistryError(f"managed runtime uv operation failed: {detail[-4000:]}")

    def _verify(
        self,
        target: Mapping[str, RuntimePackage],
        *,
        removed: Sequence[str] = (),
    ) -> None:
        installed = self._installed()
        mismatches = [
            f"{name}={installed.get(name, {}).get('version')!r} expected {package.version!r}"
            for name, package in sorted(target.items())
            if installed.get(name, {}).get("version") != package.version
        ]
        still_present = [name for name in sorted(set(removed)) if name in installed]
        if mismatches or still_present:
            detail = "; ".join(mismatches + [f"{name} should be absent" for name in still_present])
            raise RuntimeRegistryError(f"managed runtime verification failed: {detail}")

    def _converge(
        self,
        *,
        target_snapshot: RuntimeRegistrySnapshot,
        previous_snapshot: RuntimeRegistrySnapshot | None,
    ) -> RuntimeConvergenceReport:
        self._ensure_venv()
        target = _runtime_packages(target_snapshot)
        previous = {} if previous_snapshot is None else _runtime_packages(previous_snapshot)
        installed = self._installed()

        removed = sorted(set(previous) - set(target))
        local_root = self.venv.resolve()
        removable: list[str] = []
        for name in removed:
            raw_path = installed.get(name, {}).get("path")
            if not raw_path:
                continue
            with contextlib.suppress(OSError, ValueError):
                if Path(raw_path).resolve().is_relative_to(local_root):
                    removable.append(name)
        if removable:
            self._uv(
                [
                    "pip",
                    "uninstall",
                    "--python",
                    str(self.managed_python),
                    *removable,
                ]
            )

        needed = [
            package
            for name, package in sorted(target.items())
            if installed.get(name, {}).get("version") != package.version
        ]
        by_index: dict[str, list[RuntimePackage]] = {}
        for package in needed:
            by_index.setdefault(package.index_url, []).append(package)
        installed_requirements: list[str] = []
        for index_url in sorted(by_index):
            requirements = [
                f"{package.name}=={package.version}"
                for package in by_index[index_url]
            ]
            self._uv(
                [
                    "pip",
                    "install",
                    "--python",
                    str(self.managed_python),
                    "--no-deps",
                    "--no-python-downloads",
                    "--default-index",
                    index_url,
                    *requirements,
                ]
            )
            installed_requirements.extend(requirements)

        self._verify(target, removed=removed)
        self._write_atomic_json(
            self.marker,
            {
                "version": target_snapshot.version,
                "snapshot_sha256": target_snapshot.snapshot_sha256,
            },
        )
        return RuntimeConvergenceReport(
            version=target_snapshot.version,
            changed=True,
            installed=tuple(installed_requirements),
            removed=tuple(removable),
        )

    def prepare(
        self, target_version: int, *, hold_shared_lock: bool = False
    ) -> tuple[RuntimeConvergenceReport, object | None]:
        if os.name != "posix":
            raise RuntimeRegistryError("managed runtime materialization requires POSIX file locking")
        import fcntl

        if type(target_version) is not int or target_version <= 0:
            raise ValueError("target runtime registry version must be a positive integer")
        self.root.mkdir(parents=True, exist_ok=True)
        lock_handle = self.lock.open("a+b")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            marker = self._marker()
            target_snapshot = self._snapshot(target_version)
            target = _runtime_packages(target_snapshot)
            if (
                marker is not None
                and marker["version"] == target_version
                and marker["snapshot_sha256"] == target_snapshot.snapshot_sha256
            ):
                self._ensure_venv()
                self._verify(target)
                report = RuntimeConvergenceReport(
                    version=target_version,
                    changed=False,
                    installed=(),
                    removed=(),
                )
            else:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                marker = self._marker()
                if (
                    marker is not None
                    and marker["version"] == target_version
                    and marker["snapshot_sha256"] == target_snapshot.snapshot_sha256
                ):
                    self._ensure_venv()
                    self._verify(target)
                    report = RuntimeConvergenceReport(
                        version=target_version,
                        changed=False,
                        installed=(),
                        removed=(),
                    )
                else:
                    previous_snapshot = (
                        None
                        if marker is None
                        else self._snapshot(int(marker["version"]))
                    )
                    report = self._converge(
                        target_snapshot=target_snapshot,
                        previous_snapshot=previous_snapshot,
                    )
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)

            if hold_shared_lock:
                os.set_inheritable(lock_handle.fileno(), True)
                return report, lock_handle
            lock_handle.close()
            return report, None
        except Exception:
            lock_handle.close()
            raise


def runtime_registry_version_resolver(
    base_url: str = DEFAULT_RUNTIME_REGISTRY_URL,
    *,
    ca_bundle: str | Path | None = None,
) -> Callable[[], int]:
    client = RuntimeRegistryClient(base_url, ca_bundle=ca_bundle)
    return client.latest_version


def maybe_reexec_managed_worker_runtime() -> None:
    """Converge/re-exec inside any worker explicitly configured for the registry."""
    worker_id = os.environ.get("MLDB_RUNTIME_REGISTRY_WORKER_ID")
    if worker_id is None:
        return
    if not worker_id.strip():
        raise RuntimeRegistryError("runtime registry worker id is empty")
    raw_target = os.environ.get("MLDB_RUNTIME_REGISTRY_VERSION")
    root = os.environ.get("MLDB_RUNTIME_REGISTRY_ROOT")
    base_url = os.environ.get("MLDB_V2_RUNTIME_REGISTRY_URL")
    ca_bundle = os.environ.get("MLDB_RUNTIME_REGISTRY_CA_BUNDLE")
    if raw_target is None or root is None or base_url is None:
        raise RuntimeRegistryError(
            f"runtime-registry environment is incomplete for worker {worker_id!r}"
        )
    try:
        target = int(raw_target)
    except ValueError as error:
        raise RuntimeRegistryError("runtime registry target version is invalid") from error
    if target <= 0:
        raise RuntimeRegistryError("runtime registry target version is invalid")

    materializer = ManagedRuntimeMaterializer(
        root=root,
        registry=RuntimeRegistryClient(base_url, ca_bundle=ca_bundle),
    )
    active = os.environ.get("MLDB_RUNTIME_REGISTRY_ACTIVE_VERSION")
    if active == str(target):
        raw_fd = os.environ.get("MLDB_RUNTIME_REGISTRY_LOCK_FD")
        if raw_fd is None:
            raise RuntimeRegistryError("managed runtime re-exec lost its shared lock")
        try:
            os.fstat(int(raw_fd))
        except (OSError, ValueError) as error:
            raise RuntimeRegistryError("managed runtime re-exec lock is invalid") from error
        if Path(sys.prefix).resolve() != materializer.venv.resolve():
            raise RuntimeRegistryError("managed runtime re-exec did not use managed venv")
        return

    report, lock_handle = materializer.prepare(target, hold_shared_lock=True)
    if lock_handle is None:
        raise RuntimeRegistryError("managed runtime shared lock was not retained")
    env = os.environ.copy()
    env["MLDB_RUNTIME_REGISTRY_ACTIVE_VERSION"] = str(report.version)
    env["MLDB_RUNTIME_REGISTRY_LOCK_FD"] = str(lock_handle.fileno())
    managed_python = str(materializer.managed_python)
    os.execve(managed_python, [managed_python, *sys.argv], env)
