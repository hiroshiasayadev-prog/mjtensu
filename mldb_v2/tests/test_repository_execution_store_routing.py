from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from mldb_v2.src.common.ids import EntityKind, StudyResultId
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.repository.listing import CanonicalRepositoryListing
from mldb_v2.src.repository.mutation_coordination import StudyResultMutationCoordinator
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver


class _FakeExecutionStore:
    def __init__(self, documents: dict[tuple[EntityKind, str], dict[str, object]] | None = None):
        self.documents = dict(documents or {})
        self.locked: list[str] = []

    def resolve(self, *, kind: EntityKind, entity_id: object) -> dict[str, object]:
        key = (kind, str(entity_id))
        if key not in self.documents:
            raise FileNotFoundError(str(entity_id))
        return dict(self.documents[key])

    def create_immutable(self, *, kind, entity_id, document):
        key = (kind, str(entity_id))
        candidate = dict(document)
        existing = self.documents.get(key)
        if existing is not None and existing != candidate:
            raise ValueError("conflict")
        self.documents[key] = candidate
        return dict(candidate)

    def replace(self, *, kind, entity_id, replacement, transition_validator, require_statuses=None):
        key = (kind, str(entity_id))
        current = self.documents[key]
        if require_statuses is not None and current.get("status") not in require_statuses:
            raise ValueError("status")
        transition_validator(current, replacement)
        self.documents[key] = dict(replacement)
        return dict(replacement)

    def list_records(
        self,
        *,
        kind=None,
        namespace=None,
        study_id=None,
        statuses=None,
        created_at_from=None,
        created_at_to=None,
        limit=None,
    ):
        records = []
        for (record_kind, _entity_id), document in sorted(
            self.documents.items(), key=lambda item: item[0][1]
        ):
            if kind is not None and record_kind is not kind:
                continue
            if namespace is not None and str(document["id"]).split("/", 1)[0] != namespace:
                continue
            if study_id is not None and document.get("study") != study_id:
                continue
            if statuses is not None and document.get("status") not in statuses:
                continue
            records.append(dict(document))
        return records[:limit] if limit is not None else records

    @contextmanager
    def advisory_lock(self, identity: str):
        self.locked.append(identity)
        yield


class _NoopValidator:
    def validate(self, **_kwargs):
        return None


def _write_namespace(root: Path) -> None:
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "namespace.yaml").write_text(
        json.dumps(
            {
                "schema": "mjtensu.mldb-v2/namespace/v1",
                "id": "demo",
                "name": "Demo",
                "description": "demo",
            }
        ),
        encoding="utf-8",
    )


def test_resolver_routes_execution_record_to_store_but_definition_to_filesystem(tmp_path):
    root = tmp_path / "mldb_data"
    _write_namespace(root)
    tasks = root / "demo" / "tasks"
    tasks.mkdir()
    task = {
        "schema": "mjtensu.mldb-v2/task/v1",
        "id": "demo/task-a",
    }
    (tasks / "task-a.yaml").write_text(json.dumps(task), encoding="utf-8")

    result = {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": "demo/run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "study": "demo/study-a",
        "status": "submitted",
    }
    store = _FakeExecutionStore(
        {(EntityKind.STUDY_RESULT, str(result["id"])): result}
    )
    resolver = CanonicalRepositoryResolver(root, execution_store=store)

    assert resolver.resolve(kind=EntityKind.TASK, entity_id="demo/task-a") == task
    assert resolver.resolve(
        kind=EntityKind.STUDY_RESULT, entity_id=result["id"]
    ) == result


def test_listing_reads_execution_records_from_store(tmp_path):
    root = tmp_path / "mldb_data"
    _write_namespace(root)
    result = {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": "demo/run-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "study": "demo/study-a",
        "status": "completed",
        "created_at": "2026-10-22T00:00:00Z",
    }
    store = _FakeExecutionStore(
        {(EntityKind.STUDY_RESULT, str(result["id"])): result}
    )

    listing = CanonicalRepositoryListing(root, execution_store=store).list_study_results(
        namespace="demo", study="demo/study-a", statuses={"completed"}
    )
    assert listing["items"] == (result,)


def test_writer_routes_immutable_execution_record_to_store(tmp_path):
    root = tmp_path / "mldb_data"
    _write_namespace(root)
    store = _FakeExecutionStore()
    writer = CanonicalRepositoryWriter(
        tmp_path,
        record_validator=_NoopValidator(),
        execution_store=store,
    )
    plan = {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "id": "demo/plan-a",
    }

    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id="demo/plan-a",
        document=plan,
    ) == plan
    assert not (root / "demo" / "study_plans" / "plan-a.yaml").exists()
    assert store.resolve(kind=EntityKind.STUDY_PLAN, entity_id="demo/plan-a") == plan


def test_mutation_coordinator_uses_execution_store_advisory_lock(tmp_path):
    store = _FakeExecutionStore()
    coordinator = StudyResultMutationCoordinator(tmp_path, execution_store=store)
    result_id = StudyResultId("demo/run-cccccccccccccccccccccccccccccccc")

    with coordinator.acquire(study_result_id=result_id):
        pass

    assert store.locked == [str(result_id)]
