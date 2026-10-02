from __future__ import annotations

import json
from pathlib import Path

import pytest

from mldb_v2.src.catalog.architecture import Architecture
from mldb_v2.src.evaluation.evaluation_protocol import EvaluationProtocol
from mldb_v2.src.runtime_registry import (
    ManagedRuntimeMaterializer,
    RuntimeRegistryError,
    RuntimeRegistrySnapshot,
    _snapshot_sha256,
    maybe_reexec_managed_worker_runtime,
)
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.study.study import Study
from mldb_v2.src.training.train_protocol import TrainProtocol


INDEX = "https://pypi.org/simple"


def _snapshot(version: int, packages: dict[str, str]) -> RuntimeRegistrySnapshot:
    dependencies = ",\n    ".join(
        json.dumps(f"{name}=={package_version}")
        for name, package_version in sorted(packages.items())
    )
    pyproject = (
        "[project]\n"
        'name = "mldb-global-runtime"\n'
        'version = "0.0.0"\n'
        'requires-python = ">=3.11,<3.12"\n'
        "dependencies = [\n    "
        + dependencies
        + "\n]\n"
    ).encode()
    lock_lines = [
        "version = 1",
        'requires-python = "==3.11.*"',
        "",
        "[[package]]",
        'name = "mldb-global-runtime"',
        'source = { virtual = "." }',
    ]
    for name, package_version in sorted(packages.items()):
        lock_lines += [
            "",
            "[[package]]",
            f"name = {json.dumps(name)}",
            f"version = {json.dumps(package_version)}",
            f'source = {{ registry = "{INDEX}" }}',
        ]
    uv_lock = ("\n".join(lock_lines) + "\n").encode()
    return RuntimeRegistrySnapshot(
        version=version,
        snapshot_sha256=_snapshot_sha256(pyproject, uv_lock),
        pyproject_toml=pyproject,
        uv_lock=uv_lock,
    )


class _Registry:
    def __init__(self, *snapshots: RuntimeRegistrySnapshot) -> None:
        self.snapshots = {snapshot.version: snapshot for snapshot in snapshots}
        self.calls: list[int] = []

    def get(self, version: int) -> RuntimeRegistrySnapshot:
        self.calls.append(version)
        return self.snapshots[version]


def _marker(materializer: ManagedRuntimeMaterializer, snapshot: RuntimeRegistrySnapshot) -> None:
    materializer._write_atomic_json(
        materializer.marker,
        {
            "version": snapshot.version,
            "snapshot_sha256": snapshot.snapshot_sha256,
        },
    )


def test_same_version_verifies_but_skips_uv_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _snapshot(1, {"foo": "1.0", "torch": "2.5.1+cu124"})
    materializer = ManagedRuntimeMaterializer(
        root=tmp_path,
        registry=_Registry(target),  # type: ignore[arg-type]
    )
    _marker(materializer, target)
    state = {
        "foo": {"version": "1.0", "path": "/opt/conda/lib/python3.11/site-packages"},
        "torch": {
            "version": "2.5.1+cu124",
            "path": "/opt/conda/lib/python3.11/site-packages",
        },
    }
    monkeypatch.setattr(materializer, "_ensure_venv", lambda: None)
    monkeypatch.setattr(materializer, "_installed", lambda: dict(state))
    monkeypatch.setattr(
        materializer,
        "_uv",
        lambda _args: (_ for _ in ()).throw(AssertionError("same version must not invoke uv")),
    )

    report, lock = materializer.prepare(1)

    assert lock is None
    assert report.changed is False
    assert report.installed == ()
    assert report.removed == ()
    assert json.loads(materializer.marker.read_text())["version"] == 1


def test_version_change_applies_only_required_package_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _snapshot(1, {"bar": "1.0", "foo": "1.0", "torch": "2.5.1+cu124"})
    target = _snapshot(2, {"baz": "1.0", "foo": "2.0", "torch": "2.5.1+cu124"})
    materializer = ManagedRuntimeMaterializer(
        root=tmp_path,
        registry=_Registry(previous, target),  # type: ignore[arg-type]
    )
    _marker(materializer, previous)
    local = str((materializer.venv / "lib/python3.11/site-packages").resolve())
    state: dict[str, dict[str, str]] = {
        "bar": {"version": "1.0", "path": local},
        "foo": {"version": "1.0", "path": local},
        "torch": {
            "version": "2.5.1+cu124",
            "path": "/opt/conda/lib/python3.11/site-packages",
        },
    }
    calls: list[tuple[str, ...]] = []

    def fake_uv(arguments):
        args = tuple(arguments)
        calls.append(args)
        if args[1] == "uninstall":
            for name in args[4:]:
                state.pop(name, None)
            return
        assert args[1] == "install"
        requirements = [
            arg for arg in args
            if "==" in arg and not arg.startswith("http")
        ]
        for requirement in requirements:
            name, version = requirement.split("==", 1)
            state[name.lower()] = {"version": version, "path": local}

    monkeypatch.setattr(materializer, "_ensure_venv", lambda: None)
    monkeypatch.setattr(materializer, "_installed", lambda: dict(state))
    monkeypatch.setattr(materializer, "_uv", fake_uv)

    report, _lock = materializer.prepare(2)

    flattened = [argument for call in calls for argument in call]
    assert "torch==2.5.1+cu124" not in flattened
    assert "bar" in flattened
    assert "baz==1.0" in flattened
    assert "foo==2.0" in flattened
    assert report.installed == ("baz==1.0", "foo==2.0")
    assert report.removed == ("bar",)
    assert json.loads(materializer.marker.read_text())["version"] == 2


def test_failed_convergence_leaves_prior_marker_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _snapshot(1, {"foo": "1.0"})
    target = _snapshot(2, {"foo": "2.0"})
    materializer = ManagedRuntimeMaterializer(
        root=tmp_path,
        registry=_Registry(previous, target),  # type: ignore[arg-type]
    )
    _marker(materializer, previous)
    state = {
        "foo": {
            "version": "1.0",
            "path": str((materializer.venv / "lib/python3.11/site-packages").resolve()),
        }
    }
    monkeypatch.setattr(materializer, "_ensure_venv", lambda: None)
    monkeypatch.setattr(materializer, "_installed", lambda: dict(state))
    monkeypatch.setattr(
        materializer,
        "_uv",
        lambda _args: (_ for _ in ()).throw(RuntimeRegistryError("synthetic install failure")),
    )

    with pytest.raises(RuntimeRegistryError, match="synthetic install failure"):
        materializer.prepare(2)

    marker = json.loads(materializer.marker.read_text())
    assert marker["version"] == 1
    assert marker["snapshot_sha256"] == previous.snapshot_sha256


def test_authoring_schemas_do_not_gain_runtime_dependency_declarations() -> None:
    forbidden = {
        "packages",
        "dependencies",
        "repository",
        "repositories",
        "environment",
        "runtime_environment",
        "runtime_registry_version",
    }
    for schema in (Study, StudyPlan, Architecture, TrainProtocol, EvaluationProtocol):
        assert forbidden.isdisjoint(schema.__annotations__)



def test_worker_without_registry_opt_in_skips_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MLDB_RUNTIME_REGISTRY_WORKER_ID",
        "MLDB_RUNTIME_REGISTRY_VERSION",
        "MLDB_RUNTIME_REGISTRY_ROOT",
        "MLDB_V2_RUNTIME_REGISTRY_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    maybe_reexec_managed_worker_runtime()


def test_any_named_worker_opts_into_registry_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLDB_RUNTIME_REGISTRY_WORKER_ID", "old-gpu3090")
    monkeypatch.delenv("MLDB_RUNTIME_REGISTRY_VERSION", raising=False)
    monkeypatch.delenv("MLDB_RUNTIME_REGISTRY_ROOT", raising=False)
    monkeypatch.delenv("MLDB_V2_RUNTIME_REGISTRY_URL", raising=False)
    with pytest.raises(RuntimeRegistryError, match="old-gpu3090"):
        maybe_reexec_managed_worker_runtime()
