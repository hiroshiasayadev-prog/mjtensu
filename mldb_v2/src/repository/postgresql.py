"""PostgreSQL persistence for MLDB execution records.

Definitions stay Git/filesystem-owned.  This module stores only formal execution
records: StudyPlan, StudyResult, TrainingResult, Model and EvaluationResult.
The complete canonical document remains the authority; relational columns are
query projections only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from mldb_v2.src.common.ids import EntityKind, _validate_typed_reference
from mldb_v2.src.repository.execution_store import EXECUTION_RECORD_KINDS


_SCHEMA_STATEMENTS = (
    "CREATE SCHEMA IF NOT EXISTS mldb",
    """
    CREATE TABLE IF NOT EXISTS mldb.canonical_records (
        kind text NOT NULL,
        id text NOT NULL,
        namespace text NOT NULL,
        schema_id text NOT NULL,
        document jsonb NOT NULL,
        document_sha256 character(64) NOT NULL,
        status text,
        study_id text,
        study_result_id text,
        plan_id text,
        source_commit text,
        created_at timestamptz,
        revision bigint NOT NULL DEFAULT 1,
        stored_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT canonical_records_pk PRIMARY KEY (kind, id),
        CONSTRAINT canonical_records_kind_ck CHECK (
            kind = ANY (ARRAY[
                'study_plan'::text,
                'study_result'::text,
                'training_result'::text,
                'model'::text,
                'evaluation_result'::text
            ])
        ),
        CONSTRAINT canonical_records_revision_ck CHECK (revision >= 1),
        CONSTRAINT canonical_records_sha256_ck CHECK (
            document_sha256 ~ '^[0-9a-f]{64}$'::text
        ),
        CONSTRAINT canonical_records_typed_id_ck CHECK (
            id LIKE (namespace || '/%')
        )
    )
    """,
    "CREATE INDEX IF NOT EXISTS canonical_records_namespace_kind_idx ON mldb.canonical_records(namespace, kind)",
    "CREATE INDEX IF NOT EXISTS canonical_records_study_idx ON mldb.canonical_records(study_id) WHERE study_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS canonical_records_study_result_idx ON mldb.canonical_records(study_result_id) WHERE study_result_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS canonical_records_plan_idx ON mldb.canonical_records(plan_id) WHERE plan_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS canonical_records_status_created_idx ON mldb.canonical_records(status, created_at DESC) WHERE status IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS canonical_records_document_gin_idx ON mldb.canonical_records USING gin(document)",
)


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(document: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _projection(kind: EntityKind, document: Mapping[str, object]) -> dict[str, object | None]:
    entity_id = _validate_typed_reference(document.get("id"))
    namespace = entity_id.split("/", 1)[0]
    schema_id = document.get("schema")
    if type(schema_id) is not str or not schema_id:
        raise ValueError("canonical execution record schema is missing")

    study_id: str | None = None
    study_result_id: str | None = None
    plan_id: str | None = None

    raw_study = document.get("study")
    if type(raw_study) is str:
        study_id = raw_study

    raw_result = document.get("study_result")
    if type(raw_result) is str:
        study_result_id = raw_result
    elif kind is EntityKind.STUDY_RESULT:
        study_result_id = entity_id

    raw_plan = document.get("plan")
    if type(raw_plan) is str:
        plan_id = raw_plan
    elif kind is EntityKind.STUDY_PLAN:
        plan_id = entity_id

    status = document.get("status")
    if status is not None and type(status) is not str:
        status = None

    source_commit = document.get("source_commit")
    if source_commit is not None and type(source_commit) is not str:
        source_commit = None

    created_at = document.get("created_at")
    if created_at is not None and type(created_at) is not str:
        created_at = None

    return {
        "kind": kind.value,
        "id": entity_id,
        "namespace": namespace,
        "schema_id": schema_id,
        "document_sha256": _sha256(document),
        "status": status,
        "study_id": study_id,
        "study_result_id": study_result_id,
        "plan_id": plan_id,
        "source_commit": source_commit,
        "created_at": created_at,
    }


class PostgresExecutionRecordStore:
    """Small transactional store; canonical validation remains above this layer."""

    def __init__(
        self,
        dsn: str,
        *,
        connect: Callable[..., Any] | None = None,
        pool_min_size: int = 0,
        pool_max_size: int = 8,
    ) -> None:
        if type(dsn) is not str or not dsn.strip():
            raise ValueError("PostgreSQL DSN must be a non-empty string")
        if type(pool_min_size) is not int or pool_min_size < 0:
            raise ValueError("pool_min_size must be a non-negative integer")
        if type(pool_max_size) is not int or pool_max_size < 1:
            raise ValueError("pool_max_size must be a positive integer")
        if pool_min_size > pool_max_size:
            raise ValueError("pool_min_size must not exceed pool_max_size")
        self._dsn = dsn
        self._connect_override = connect
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._resolved_pool: Any | None = None

    def _pool(self):
        if self._resolved_pool is not None:
            return self._resolved_pool
        try:
            from psycopg_pool import ConnectionPool  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "psycopg[pool] is required to activate PostgreSQL execution-record storage"
            ) from error
        self._resolved_pool = ConnectionPool(
            conninfo=self._dsn,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            open=True,
        )
        return self._resolved_pool

    @contextmanager
    def _connection(self):
        if self._connect_override is not None:
            with self._connect_override(self._dsn) as connection:
                yield connection
            return
        with self._pool().connection() as connection:
            yield connection

    def ensure_schema(self) -> None:
        """Create the MLDB-owned schema/indexes without changing record semantics."""
        with self._connection() as connection:
            with connection.cursor() as cursor:
                for statement in _SCHEMA_STATEMENTS:
                    cursor.execute(statement)

    def resolve(self, *, kind: EntityKind, entity_id: object) -> dict[str, object]:
        if kind not in EXECUTION_RECORD_KINDS:
            raise ValueError(f"PostgreSQL store does not own {kind.value}")
        typed_id = _validate_typed_reference(entity_id)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT document FROM mldb.canonical_records WHERE kind = %s AND id = %s",
                    (kind.value, typed_id),
                )
                row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(f"entity not found: {kind.value} {typed_id}")
        document = row[0]
        if type(document) is str:
            document = json.loads(document)
        if type(document) is not dict:
            raise ValueError("stored canonical document is malformed")
        return document

    def create_immutable(
        self,
        *,
        kind: EntityKind,
        entity_id: object,
        document: Mapping[str, object],
    ) -> dict[str, object]:
        if kind not in EXECUTION_RECORD_KINDS:
            raise ValueError(f"PostgreSQL store does not own {kind.value}")
        typed_id = _validate_typed_reference(entity_id)
        record = dict(document)
        projection = _projection(kind, record)
        if projection["id"] != typed_id:
            raise ValueError("canonical document id does not match requested identity")

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO mldb.canonical_records (
                        kind,id,namespace,schema_id,document,document_sha256,
                        status,study_id,study_result_id,plan_id,source_commit,created_at
                    ) VALUES (
                        %s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s::timestamptz
                    )
                    ON CONFLICT (kind,id) DO NOTHING
                    RETURNING document
                    """,
                    (
                        projection["kind"],
                        projection["id"],
                        projection["namespace"],
                        projection["schema_id"],
                        json.dumps(record, ensure_ascii=False, allow_nan=False),
                        projection["document_sha256"],
                        projection["status"],
                        projection["study_id"],
                        projection["study_result_id"],
                        projection["plan_id"],
                        projection["source_commit"],
                        projection["created_at"],
                    ),
                )
                inserted = cursor.fetchone()
                if inserted is not None:
                    return record
                cursor.execute(
                    "SELECT document, document_sha256 FROM mldb.canonical_records WHERE kind=%s AND id=%s FOR UPDATE",
                    (kind.value, typed_id),
                )
                existing = cursor.fetchone()
                if existing is None:
                    raise RuntimeError("canonical insert conflict disappeared")
                existing_doc = existing[0]
                if type(existing_doc) is str:
                    existing_doc = json.loads(existing_doc)
                if existing[1].strip() == projection["document_sha256"] and existing_doc == record:
                    return dict(existing_doc)
                raise ValueError("lifecycle conflict: canonical identity already exists")

    def replace(
        self,
        *,
        kind: EntityKind,
        entity_id: object,
        replacement: Mapping[str, object],
        transition_validator: Callable[[Mapping[str, object], Mapping[str, object]], None],
        require_statuses: frozenset[str] | None = None,
    ) -> dict[str, object]:
        if kind not in EXECUTION_RECORD_KINDS:
            raise ValueError(f"PostgreSQL store does not own {kind.value}")
        typed_id = _validate_typed_reference(entity_id)
        record = dict(replacement)
        projection = _projection(kind, record)
        if projection["id"] != typed_id:
            raise ValueError("canonical document id does not match requested identity")

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT document, revision FROM mldb.canonical_records WHERE kind=%s AND id=%s FOR UPDATE",
                    (kind.value, typed_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise FileNotFoundError(f"entity not found: {kind.value} {typed_id}")
                current = row[0]
                if type(current) is str:
                    current = json.loads(current)
                if current == record:
                    return dict(current)
                if require_statuses is not None and current.get("status") not in require_statuses:
                    raise ValueError("lifecycle conflict: current status is not replaceable")
                transition_validator(current, record)
                cursor.execute(
                    """
                    UPDATE mldb.canonical_records
                       SET namespace=%s, schema_id=%s, document=%s::jsonb,
                           document_sha256=%s, status=%s, study_id=%s,
                           study_result_id=%s, plan_id=%s, source_commit=%s,
                           created_at=%s::timestamptz, revision=revision+1,
                           stored_at=clock_timestamp()
                     WHERE kind=%s AND id=%s AND revision=%s
                    """,
                    (
                        projection["namespace"],
                        projection["schema_id"],
                        json.dumps(record, ensure_ascii=False, allow_nan=False),
                        projection["document_sha256"],
                        projection["status"],
                        projection["study_id"],
                        projection["study_result_id"],
                        projection["plan_id"],
                        projection["source_commit"],
                        projection["created_at"],
                        kind.value,
                        typed_id,
                        row[1],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("concurrent canonical record update")
        return record

    def list_records(
        self,
        *,
        kind: EntityKind | None = None,
        namespace: str | None = None,
        study_id: str | None = None,
        statuses: frozenset[str] | None = None,
        created_at_from: datetime | None = None,
        created_at_to: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            if kind not in EXECUTION_RECORD_KINDS:
                return []
            clauses.append("kind = %s")
            params.append(kind.value)
        if namespace is not None:
            clauses.append("namespace = %s")
            params.append(namespace)
        if study_id is not None:
            clauses.append("study_id = %s")
            params.append(study_id)
        if statuses:
            clauses.append("status = ANY(%s)")
            params.append(list(sorted(statuses)))
        if created_at_from is not None:
            clauses.append("created_at >= %s")
            params.append(created_at_from)
        if created_at_to is not None:
            clauses.append("created_at <= %s")
            params.append(created_at_to)
        sql = "SELECT document FROM mldb.canonical_records"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            document = row[0]
            if type(document) is str:
                document = json.loads(document)
            if type(document) is not dict:
                raise ValueError("stored canonical document is malformed")
            result.append(document)
        return result

    @contextmanager
    def advisory_lock(self, identity: str) -> Iterator[None]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (identity,))
                try:
                    yield
                finally:
                    cursor.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (identity,)
                    )
