from __future__ import annotations

import hashlib
import json
import os
import subprocess
from io import StringIO
from pathlib import Path

import pytest

import mldb_v2.src.api.composition as composition
import mldb_v2.src.cli.main as cli_main
import mldb_v2.tests.test_api_application_integration as app_fx
import mldb_v2.tests.test_api_execution as execution_fx
import mldb_v2.tests.test_cli_main as main_fx
from mldb_v2.src.api.application import Application
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.tests.test_study_source_pinning import _commit_all, _training_repo


REPO = Path(__file__).resolve().parents[2]


def _run(
    argv: list[str],
    *,
    application: object | None = None,
    default_backend: str | None = None,
) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    kwargs: dict[str, object] = {
        "argv": argv,
        "stdout": stdout,
        "stderr": stderr,
    }
    if application is not None:
        kwargs["application_factory"] = lambda: application
        kwargs["default_backend"] = default_backend
    status = cli_main.main(**kwargs)  # type: ignore[arg-type]
    return status, stdout.getvalue(), stderr.getvalue()


def _terminalize(repository_root: Path, calls: list[str]):
    def advance(*, study_result):
        calls.append(str(study_result))
        execution_fx._complete_result(repository_root, str(study_result))
        return {
            "study_result": study_result,
            "status": "completed",
            "changed": True,
            "admitted": [],
            "finalized_results": [],
            "finalized_models": [],
            "active": [],
            "terminal": True,
        }

    return advance


class _DictTransport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = dict(objects)

    def read_bytes(self, uri: str) -> bytes:
        return self._objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("CLI integration verification must not publish object bytes")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _production_valid_training_repo(tmp_path: Path) -> tuple[Path, str, dict[str, bytes]]:
    repository_root, study_id, _commit, _sources = _training_repo(tmp_path)
    root = repository_root / "mldb_data"
    objects: dict[str, bytes] = {}

    for local, payload in (("train-v1", b"train-data"), ("eval-v1", b"eval-data")):
        object_name = f"{local}.bin"
        manifest = (
            json.dumps(
                {
                    "path": object_name,
                    "bytes": len(payload),
                    "sha256": _sha(payload),
                    "split": "train",
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        manifest_path = root / "data-ns" / "corpora" / f"{local}.manifest.jsonl"
        manifest_path.write_bytes(manifest)
        definition_path = root / "data-ns" / "corpora" / f"{local}.yaml"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        definition["manifest"]["sha256"] = _sha(manifest)
        definition["manifest"]["entries"] = 1
        definition_path.write_text(
            json.dumps(definition, separators=(",", ":")), encoding="utf-8"
        )
        objects[f"s3://bucket/{local}/{object_name}"] = payload

    companions = {
        root / "arch-ns" / "architectures" / "arch-a-v1.py": (
            b"import torch.nn as nn\n\ndef build():\n    return nn.Identity()\n"
        ),
        root / "arch-ns" / "architectures" / "arch-z-v1.py": (
            b"import torch.nn as nn\n\ndef build():\n    return nn.Identity()\n"
        ),
        root / "proto-ns" / "train_protocols" / "train-v1.py": (
            b"def train(context):\n    return context.model\n"
        ),
        root / "proto-ns" / "evaluation_protocols" / "eval-v1.py": (
            b"def evaluate(context):\n    return None\n"
        ),
    }
    for companion_path, companion in companions.items():
        companion_path.write_bytes(companion)
        definition_path = companion_path.with_suffix(".yaml")
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        definition["implementation"]["sha256"] = _sha(companion)
        definition_path.write_text(
            json.dumps(definition, separators=(",", ":")), encoding="utf-8"
        )

    for namespace, domain, local in (
        ("arch-ns", "architectures", "arch-a-v1"),
        ("arch-ns", "architectures", "arch-z-v1"),
        ("proto-ns", "train_protocols", "train-v1"),
        ("proto-ns", "evaluation_protocols", "eval-v1"),
    ):
        directory = repository_root / "mldb_v2" / "tests" / namespace / domain / local
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test_asset.py").write_text(
            "def test_asset():\n    assert True\n", encoding="utf-8"
        )

    _commit_all(repository_root, "production-valid CLI fixture")
    return repository_root, study_id, objects


def _real_application(tmp_path: Path) -> tuple[Path, str, app_fx.FakeBackend, Application]:
    repository_root, study_id, objects = _production_valid_training_repo(tmp_path)
    backend = app_fx.FakeBackend()
    application = Application(
        repository_root=repository_root,
        backend_registry=app_fx._registry(backend),
        object_bytes=_ObjectByteAccess(_DictTransport(objects)),
    )
    return repository_root, study_id, backend, application


def test_production_composition_read_only_commands_do_not_activate_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mldb_data").mkdir()
    monkeypatch.setenv("MLDB_REPO_ROOT", str(tmp_path))

    def fail_activation(*args, **kwargs):
        raise AssertionError("read-only CLI composition must not activate backend or S3")

    monkeypatch.setattr(BackendRegistry, "resolve", fail_activation)
    monkeypatch.setattr(composition, "_create_s3_transport", fail_activation)

    commands = (
        ["ps", "--json"],
        ["get", "tasks", "--json"],
        ["status", "--json"],
        ["watch", "--json"],
    )
    for argv in commands:
        status, stdout, stderr = _run(argv)
        assert status == 0, (argv, stdout, stderr)
        assert stderr == ""
        assert "CLI application construction is not configured" not in stdout
        assert "CLI application construction is not configured" not in stderr


def test_production_composition_uses_public_seam_only() -> None:
    source = (REPO / "mldb_v2/src/cli/main.py").read_text(encoding="utf-8")
    assert "from mldb_v2.src.api import ApplicationInterface, compose_application" in source
    for forbidden in (
        "BackendRegistry",
        "BackendConfig",
        "_ObjectByteAccess",
        "CanonicalRepositoryResolver",
        "StudyDriver",
        "clearml_backend",
        "s3_transport",
    ):
        assert forbidden not in source


def test_default_backend_resolution_explicit_configured_and_missing() -> None:
    explicit = main_fx.FakeApplication()
    status, _, stderr = _run(
        ["run", "demo/study-v1", "--backend", "fake"],
        application=explicit,
        default_backend="ignored",
    )
    assert status == 0
    assert stderr == ""
    assert explicit.calls == [
        ("run_study", {"study": "demo/study-v1", "backend": "fake"})
    ]

    configured = main_fx.FakeApplication()
    status, _, stderr = _run(
        ["run", "demo/study-v1"],
        application=configured,
        default_backend="fake",
    )
    assert status == 0
    assert stderr == ""
    assert configured.calls == [
        ("run_study", {"study": "demo/study-v1", "backend": "fake"})
    ]

    missing = main_fx.FakeApplication()
    status, stdout, stderr = _run(
        ["run", "demo/study-v1"], application=missing, default_backend=None
    )
    assert status != 0
    assert stdout == ""
    assert stderr.startswith("invalid_request:")
    assert missing.calls == []


def test_plan_uses_real_w007_application(tmp_path: Path) -> None:
    _repo, study_id, backend, application = _real_application(tmp_path)
    status, stdout, stderr = _run(["plan", study_id], application=application)
    assert status == 0
    assert stderr == ""
    assert "study_plan" in stdout or "plan" in stdout
    assert backend.admit_calls == []
    assert backend.observe_calls == []
    assert backend.collect_calls == []


def test_run_and_resume_keep_foreground_progression_in_application(tmp_path: Path) -> None:
    repository_root, study_id, backend, application = _real_application(tmp_path)
    run_calls: list[str] = []
    application._execution._advance_one_pass = _terminalize(repository_root, run_calls)

    status, stdout, stderr = _run(
        ["run", study_id, "--backend", "fake"], application=application
    )
    assert status == 0
    assert stderr == ""
    assert stdout
    assert len(run_calls) == 1

    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"],
        backend="fake",
        execution_key=execution_fx.FRESH_UUID_B.hex,
    )
    resume_calls: list[str] = []
    application._execution._advance_one_pass = _terminalize(repository_root, resume_calls)
    status, stdout, stderr = _run(["resume", source["id"]], application=application)
    assert status == 0
    assert stderr == ""
    assert stdout
    assert resume_calls == [source["id"]]
    assert backend.admit_calls == []
    assert backend.collect_calls == []


def test_rerun_uses_real_application_and_exact_source_plan(tmp_path: Path) -> None:
    repository_root, study_id, _backend, application = _real_application(tmp_path)
    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    calls: list[str] = []
    application._execution._advance_one_pass = _terminalize(repository_root, calls)

    status, stdout, stderr = _run(["rerun", source["id"]], application=application)
    assert status == 0
    assert stderr == ""
    assert stdout
    assert len(calls) == 1
    created = application.get_study_result(study_result=calls[0])["study_result"]
    assert created["plan"] == source["plan"]


def test_cancel_delegates_only_to_real_application_entrypoint(tmp_path: Path) -> None:
    _repo, study_id, backend, application = _real_application(tmp_path)
    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )

    status, stdout, stderr = _run(["cancel", source["id"]], application=application)
    assert status == 0
    assert stderr == ""
    assert "cancelling" in stdout
    assert backend.cancel_calls == [source["id"]]
    assert backend.admit_calls == []
    assert backend.collect_calls == []


def test_advance_calls_real_application_exactly_once(tmp_path: Path) -> None:
    _repo, study_id, backend, application = _real_application(tmp_path)
    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    original = application.advance_study
    calls: list[str] = []

    def counted(*, study_result):
        calls.append(str(study_result))
        return original(study_result=study_result)

    application.advance_study = counted  # type: ignore[method-assign]
    status, stdout, stderr = _run(["advance", source["id"]], application=application)
    assert status == 0
    assert stderr == ""
    assert stdout
    assert calls == [source["id"]]
    assert backend.admit_calls


def test_exact_status_and_watch_are_read_only_on_real_application(tmp_path: Path) -> None:
    repository_root, study_id, backend, application = _real_application(tmp_path)
    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    namespace, local_id = str(source["id"]).split("/", 1)
    result_path = repository_root / "mldb_data" / namespace / "study_results" / f"{local_id}.yaml"
    before = result_path.read_bytes()

    for argv in (["status", source["id"], "--json"], ["watch", source["id"], "--json"]):
        status, stdout, stderr = _run(list(argv), application=application)
        assert status == 0
        assert stderr == ""
        rendered = json.loads(stdout)
        assert rendered["study_result"]["id"] == source["id"]
        assert result_path.read_bytes() == before

    assert backend.admit_calls == []
    assert backend.collect_calls == []
    assert backend.cancel_calls == []


def test_idless_status_watch_and_get_remain_discovery(tmp_path: Path) -> None:
    _repo, _study_id, backend, application = _real_application(tmp_path)
    for argv in (
        ["get", "tasks", "--json"],
        ["status", "--json"],
        ["watch", "--json"],
    ):
        status, stdout, stderr = _run(argv, application=application)
        assert status == 0
        assert stderr == ""
        json.loads(stdout)
    assert backend.admit_calls == []
    assert backend.collect_calls == []
    assert backend.cancel_calls == []


def test_unsupported_logs_surface_public_capability_error(tmp_path: Path) -> None:
    _repo, study_id, _backend, application = _real_application(tmp_path)
    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    status, stdout, stderr = _run(["logs", source["id"]], application=application)
    assert status != 0
    assert stdout == ""
    assert stderr.startswith("unsupported_capability:")


def test_structured_exact_and_list_shapes_are_preserved(tmp_path: Path) -> None:
    _repo, study_id, _backend, application = _real_application(tmp_path)
    plan = application.plan_study(study=study_id)
    source = application.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )

    status, list_json, stderr = _run(["ps", "--all", "--json"], application=application)
    assert status == 0 and stderr == ""
    listing = json.loads(list_json)
    assert set(listing) == {"items", "issues"}
    assert isinstance(listing["items"], list)

    status, exact_json, stderr = _run(["status", source["id"], "--json"], application=application)
    assert status == 0 and stderr == ""
    exact = json.loads(exact_json)
    assert isinstance(exact, dict)
    assert exact["study_result"]["id"] == source["id"]


def test_repository_command_resolves_from_outside_cwd(tmp_path: Path) -> None:
    command = str(REPO / "mldb.cmd")
    help_result = subprocess.run(
        ["cmd", "/c", command, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "usage: mldb" in help_result.stdout
    assert help_result.stderr == ""

    read_only = subprocess.run(
        ["cmd", "/c", command, "ps", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert read_only.returncode == 0
    assert read_only.stderr == ""
    listing = json.loads(read_only.stdout)
    assert set(listing) == {"items", "issues"}
