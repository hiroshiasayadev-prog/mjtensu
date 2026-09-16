from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mldb_v2.src.api._query import (
    ReadOnlyQueryService,
    _InvalidQueryRequest,
    _UnsupportedQueryCapability,
)
from mldb_v2.src.common.ids import EntityKind


EXECUTION_KEY = "1234567812344abc9234567890abcdef"
STUDY_RESULT_ID = f"demo/run-{EXECUTION_KEY}"
PLAN_ID = "demo/study-v1-plan-0123456789abcdef"
STUDY_ID = "demo/study-v1"
SOURCE_COMMIT = "a" * 40

_SCHEMA = {
    EntityKind.TASK: "mjtensu.mldb-v2/task/v1",
    EntityKind.CORPUS: "mjtensu.mldb-v2/corpus/v1",
    EntityKind.ARCHITECTURE: "mjtensu.mldb-v2/architecture/v1",
    EntityKind.TRAIN_PROTOCOL: "mjtensu.mldb-v2/train-protocol/v1",
    EntityKind.EVALUATION_PROTOCOL: "mjtensu.mldb-v2/evaluation-protocol/v1",
    EntityKind.STUDY: "mjtensu.mldb-v2/study/v1",
    EntityKind.MODEL: "mjtensu.mldb-v2/model/v1",
    EntityKind.STUDY_RESULT: "mjtensu.mldb-v2/study-result/v1",
}
_DOMAIN = {
    EntityKind.TASK: "tasks",
    EntityKind.CORPUS: "corpora",
    EntityKind.ARCHITECTURE: "architectures",
    EntityKind.TRAIN_PROTOCOL: "train_protocols",
    EntityKind.EVALUATION_PROTOCOL: "evaluation_protocols",
    EntityKind.STUDY: "studies",
    EntityKind.MODEL: "models",
    EntityKind.STUDY_RESULT: "study_results",
}
_DEFINITION_KINDS = (
    EntityKind.TASK,
    EntityKind.CORPUS,
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
    EntityKind.STUDY,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _namespace(root: Path, namespace: str) -> None:
    _write_json(
        root / namespace / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": namespace,
            "name": namespace,
            "description": "test namespace",
        },
    )


def _entity(
    root: Path,
    namespace: str,
    kind: EntityKind,
    local_id: str,
    *,
    extra: dict[str, object] | None = None,
) -> Path:
    value: dict[str, object] = {
        "schema": _SCHEMA[kind],
        "id": f"{namespace}/{local_id}",
    }
    if extra:
        value.update(extra)
    path = root / namespace / _DOMAIN[kind] / f"{local_id}.yaml"
    _write_json(path, value)
    return path


def _install_definitions(root: Path, namespace: str, *, status: str = "sealed") -> None:
    _namespace(root, namespace)
    for kind in _DEFINITION_KINDS:
        local_id = {
            EntityKind.TASK: "task-v1",
            EntityKind.CORPUS: "corpus-v1",
            EntityKind.ARCHITECTURE: "arch-v1",
            EntityKind.TRAIN_PROTOCOL: "train-v1",
            EntityKind.EVALUATION_PROTOCOL: "eval-v1",
            EntityKind.STUDY: "study-v1",
        }[kind]
        extra: dict[str, object] = {"status": status}
        if kind is EntityKind.CORPUS:
            extra["manifest"] = {"file": f"{local_id}.manifest.jsonl"}
        path = _entity(root, namespace, kind, local_id, extra=extra)
        if kind in {
            EntityKind.ARCHITECTURE,
            EntityKind.TRAIN_PROTOCOL,
            EntityKind.EVALUATION_PROTOCOL,
        }:
            path.with_suffix(".py").write_text(
                "raise RuntimeError('listing imported executable companion')\n",
                encoding="utf-8",
            )
        if kind is EntityKind.CORPUS:
            path.with_name(f"{local_id}.manifest.jsonl").write_text("{}\n", encoding="utf-8")


def _slot(disposition: str, result: str | None = None, reason: str | None = None) -> dict[str, object]:
    return {"disposition": disposition, "result": result, "reason": reason}


def _evaluation(
    coordinate: str,
    disposition: str,
    *,
    result_id: str | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "coordinate": coordinate,
        "stage": f"stage-{coordinate}",
        "disposition": disposition,
        "result": result_id,
        "reason": reason,
    }


def _study_result(*, all_pending: bool = False) -> dict[str, object]:
    if all_pending:
        trials = [
            {
                "trial": "trial-0001",
                "training": _slot("pending"),
                "evaluations": [
                    _evaluation("eval-0001", "pending"),
                    _evaluation("eval-0002", "pending"),
                ],
            },
            {
                "trial": "trial-0002",
                "training": None,
                "evaluations": [_evaluation("eval-0001", "pending")],
            },
        ]
    else:
        trials = [
            {
                "trial": "trial-0001",
                "training": _slot(
                    "completed",
                    f"{STUDY_RESULT_ID}-trial-0001-train",
                ),
                "evaluations": [
                    _evaluation(
                        "eval-0001",
                        "failed",
                        result_id=f"{STUDY_RESULT_ID}-trial-0001-eval-0001",
                    ),
                    _evaluation("eval-0002", "pending"),
                ],
            },
            {
                "trial": "trial-0002",
                "training": None,
                "evaluations": [
                    _evaluation("eval-0001", "skipped", reason="upstream_failed")
                ],
            },
        ]
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": STUDY_RESULT_ID,
        "execution_key": EXECUTION_KEY,
        "plan": PLAN_ID,
        "study": STUDY_ID,
        "source_commit": SOURCE_COMMIT,
        "backend": "fake",
        "created_at": "2026-09-13T00:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": trials,
    }


def _install_study_result(root: Path, *, all_pending: bool = False) -> Path:
    _namespace(root, "demo")
    result = _study_result(all_pending=all_pending)
    path = root / "demo" / "study_results" / f"run-{EXECUTION_KEY}.yaml"
    _write_json(path, result)
    return path


def _stage_key(trial: str, kind: str, coordinate: str | None) -> dict[str, object]:
    return {
        "study_result": STUDY_RESULT_ID,
        "plan": PLAN_ID,
        "trial": trial,
        "kind": kind,
        "coordinate": coordinate,
        "source_commit": SOURCE_COMMIT,
    }


def _active(stage_key: dict[str, object], execution_id: str) -> dict[str, object]:
    return {
        "state": "active",
        "stage_key": copy.deepcopy(stage_key),
        "backend": "fake",
        "execution_ids": [execution_id],
    }


class GuardBackend:
    def __init__(self) -> None:
        self.observations: dict[tuple[str, str, str | None], dict[str, object] | None] = {}
        self.observe_calls: list[dict[str, object]] = []
        self.admit_calls = 0
        self.collect_calls = 0
        self.cancel_calls = 0
        self.advance_calls = 0

    @staticmethod
    def _token(stage_key: dict[str, object]) -> tuple[str, str, str | None]:
        return (
            str(stage_key["trial"]),
            str(stage_key["kind"]),
            None if stage_key["coordinate"] is None else str(stage_key["coordinate"]),
        )

    def observe(self, *, stage_key):
        self.observe_calls.append(copy.deepcopy(stage_key))
        return copy.deepcopy(self.observations.get(self._token(stage_key)))

    def admit(self, *, stage_input):
        self.admit_calls += 1
        raise AssertionError("query must never admit")

    def collect(self, *, stage_key):
        self.collect_calls += 1
        raise AssertionError("query must never collect")

    def cancel_study(self, *, study_result):
        self.cancel_calls += 1
        raise AssertionError("query must never cancel")

    def advance(self, *args, **kwargs):
        self.advance_calls += 1
        raise AssertionError("query must never advance")


class LogBackend(GuardBackend):
    def __init__(self) -> None:
        super().__init__()
        self.logs: dict[str, tuple[str, ...]] = {}
        self.log_calls: list[tuple[str, str]] = []

    def read_task_logs(self, *, study_result, task_id):
        self.log_calls.append((str(study_result), task_id))
        chunks = self.logs.get(task_id)
        if chunks is None:
            return None
        return SimpleNamespace(chunks=chunks)


def _service(root: Path, backend: GuardBackend | None = None, **kwargs) -> ReadOnlyQueryService:
    return ReadOnlyQueryService(
        mldb_data_root=root,
        backend=backend or GuardBackend(),
        **kwargs,
    )


def test_runtime_query_interface_is_exact_frozen_runtime_mirror() -> None:
    root = Path(__file__).parents[1]
    frozen = (root / "skeleton/api/query_interface.py").read_text(encoding="utf-8")
    runtime = (root / "src/api/query_interface.py").read_text(encoding="utf-8")
    assert runtime == frozen.replace("mldb_v2.skeleton.", "mldb_v2.src.")
    assert "mldb_v2.skeleton" not in runtime


def test_list_resource_definitions_aggregate_namespace_and_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _install_definitions(root, "beta", status="draft")
    _install_definitions(root, "alpha", status="sealed")
    _entity(root, "alpha", EntityKind.MODEL, "model-v1")

    service = _service(root)
    aggregate = service.list_entities(resource="definitions")
    assert [item["id"] for item in aggregate["items"]] == [
        "alpha/task-v1",
        "alpha/corpus-v1",
        "alpha/arch-v1",
        "alpha/train-v1",
        "alpha/eval-v1",
        "alpha/study-v1",
        "beta/task-v1",
        "beta/corpus-v1",
        "beta/arch-v1",
        "beta/train-v1",
        "beta/eval-v1",
        "beta/study-v1",
    ]
    assert "alpha/model-v1" not in [item["id"] for item in aggregate["items"]]

    sealed = service.list_entities(resource="definitions", status="sealed")
    assert len(sealed["items"]) == 6
    assert all(str(item["id"]).startswith("alpha/") for item in sealed["items"])

    beta = service.list_entities(resource=EntityKind.TASK, namespace="beta", status="draft")
    assert [item["id"] for item in beta["items"]] == ["beta/task-v1"]


def test_unsupported_filter_resource_combination_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "demo")
    with pytest.raises(_InvalidQueryRequest, match="unsupported"):
        _service(root).list_entities(resource=EntityKind.MODEL, status="sealed")


def test_get_entity_is_exact_typed_resolution(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "demo")
    _entity(root, "demo", EntityKind.TASK, "task-v1", extra={"status": "sealed"})
    result = _service(root).get_entity(kind=EntityKind.TASK, entity_id="demo/task-v1")
    assert result["id"] == "demo/task-v1"
    with pytest.raises(FileNotFoundError):
        _service(root).get_entity(kind=EntityKind.TASK, entity_id="demo/missing-v1")


def test_list_study_results_filters_time_and_limit(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "demo")
    for suffix, status, created_at, study in (
        ("1" * 32, "submitted", "2026-09-11T00:00:00Z", "demo/study-v1"),
        ("2" * 32, "failed", "2026-09-12T00:00:00Z", "demo/study-v1"),
        ("3" * 32, "failed", "2026-09-13T00:00:00Z", "demo/other-v1"),
    ):
        _entity(
            root,
            "demo",
            EntityKind.STUDY_RESULT,
            f"run-{suffix}",
            extra={"study": study, "status": status, "created_at": created_at},
        )

    listing = _service(root).list_study_results(
        namespace="demo",
        study="demo/study-v1",
        statuses={"failed"},
        created_at_from=datetime(2026, 9, 11, 12, tzinfo=timezone.utc),
        created_at_to=datetime(2026, 9, 12, 12, tzinfo=timezone.utc),
        limit=1,
    )
    assert [item["id"] for item in listing["items"]] == [f"demo/run-{'2' * 32}"]
    with pytest.raises(_InvalidQueryRequest, match="timezone-aware"):
        _service(root).list_study_results(created_at_from=datetime(2026, 9, 1))


def test_get_study_result_progress_uses_only_canonical_slots(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _install_study_result(root)
    view = _service(root).get_study_result(study_result=STUDY_RESULT_ID)
    assert view["progress"] == {
        "training": {
            "planned": 1,
            "pending": 0,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "skipped": 0,
        },
        "evaluations": {
            "planned": 3,
            "pending": 1,
            "completed": 0,
            "failed": 1,
            "cancelled": 0,
            "skipped": 1,
        },
        "total": {
            "planned": 4,
            "pending": 1,
            "completed": 1,
            "failed": 1,
            "cancelled": 0,
            "skipped": 1,
        },
    }
    assert view["study_result"]["trials"][1]["training"] is None


def test_observe_queries_exact_pending_stage_keys_in_canonical_order_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    path = _install_study_result(root, all_pending=True)
    before = path.read_bytes()
    backend = GuardBackend()
    expected = [
        _stage_key("trial-0001", "training", None),
        _stage_key("trial-0001", "evaluation", "eval-0001"),
        _stage_key("trial-0001", "evaluation", "eval-0002"),
        _stage_key("trial-0002", "evaluation", "eval-0001"),
    ]
    for index, key in enumerate(expected, start=1):
        backend.observations[backend._token(key)] = _active(key, f"exec-{index}")

    observed = _service(root, backend).observe_study(study_result=STUDY_RESULT_ID)
    assert backend.observe_calls == expected
    assert [item["stage_key"] for item in observed["backend_observations"]] == expected
    assert path.read_bytes() == before
    assert observed["progress"]["training"]["pending"] == 1
    assert observed["progress"]["evaluations"]["pending"] == 3
    assert backend.admit_calls == backend.collect_calls == backend.cancel_calls == backend.advance_calls == 0


def test_read_backend_logs_supported_via_optional_capability(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _install_study_result(root, all_pending=True)
    backend = LogBackend()
    training = _stage_key("trial-0001", "training", None)
    evaluation = _stage_key("trial-0001", "evaluation", "eval-0001")
    backend.observations[backend._token(training)] = _active(training, "exec-train")
    backend.observations[backend._token(evaluation)] = _active(evaluation, "exec-eval")
    backend.logs = {"exec-train": ("train-a", "train-b"), "exec-eval": ("eval-a",)}

    chunks = list(
        _service(root, backend).read_backend_logs(
            request={
                "study_result": STUDY_RESULT_ID,
                "trial": "trial-0001",
                "coordinate": None,
                "failed_only": False,
                "follow": False,
            }
        )
    )
    assert chunks == [
        {"execution_id": "exec-train", "text": "train-a"},
        {"execution_id": "exec-train", "text": "train-b"},
        {"execution_id": "exec-eval", "text": "eval-a"},
    ]
    assert backend.log_calls == [
        (STUDY_RESULT_ID, "exec-train"),
        (STUDY_RESULT_ID, "exec-eval"),
    ]
    assert backend.admit_calls == backend.collect_calls == backend.cancel_calls == backend.advance_calls == 0


def test_read_backend_logs_unsupported_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _install_study_result(root)
    with pytest.raises(_UnsupportedQueryCapability, match="log access"):
        list(
            _service(root).read_backend_logs(
                request={
                    "study_result": STUDY_RESULT_ID,
                    "trial": None,
                    "coordinate": None,
                    "failed_only": False,
                    "follow": False,
                }
            )
        )


def test_diagnose_is_read_only_and_probes_are_injected(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    root.mkdir()
    backend = GuardBackend()
    calls: list[str] = []

    def probe():
        calls.append("probe")
        return {"name": "object_store", "status": "warning", "diagnostic": None}

    checks = _service(root, backend, diagnostic_probes=(probe,)).diagnose()
    assert checks == [
        {"name": "repository", "status": "ok", "diagnostic": None},
        {
            "name": "backend_logs",
            "status": "unsupported",
            "diagnostic": {
                "code": "unsupported_capability",
                "message": "backend log access is unavailable",
            },
        },
        {"name": "object_store", "status": "warning", "diagnostic": None},
    ]
    assert calls == ["probe"]
    assert backend.observe_calls == []
    assert backend.admit_calls == backend.collect_calls == backend.cancel_calls == backend.advance_calls == 0


def test_listing_does_not_import_executable_definition_companions(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _install_definitions(root, "demo")
    listing = _service(root).list_entities(resource="definitions")
    assert len(listing["items"]) == 6
    assert listing["issues"] == ()
