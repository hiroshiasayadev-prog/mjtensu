"""Storage port for formal MLDB execution records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol

from mldb_v2.src.common.ids import EntityKind


EXECUTION_RECORD_KINDS = frozenset(
    {
        EntityKind.STUDY_PLAN,
        EntityKind.STUDY_RESULT,
        EntityKind.TRAINING_RESULT,
        EntityKind.MODEL,
        EntityKind.EVALUATION_RESULT,
    }
)


class ExecutionRecordStore(Protocol):
    def resolve(
        self, *, kind: EntityKind, entity_id: object
    ) -> dict[str, object]: ...

    def create_immutable(
        self,
        *,
        kind: EntityKind,
        entity_id: object,
        document: Mapping[str, object],
    ) -> dict[str, object]: ...

    def replace(
        self,
        *,
        kind: EntityKind,
        entity_id: object,
        replacement: Mapping[str, object],
        transition_validator: Callable[
            [Mapping[str, object], Mapping[str, object]], None
        ],
        require_statuses: frozenset[str] | None = None,
    ) -> dict[str, object]: ...

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
    ) -> list[dict[str, object]]: ...

    def advisory_lock(self, identity: str) -> AbstractContextManager[None]: ...
