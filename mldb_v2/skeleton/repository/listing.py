"""Deterministic canonical repository inventory/query boundary."""

from datetime import datetime
from typing import Collection, Protocol, Sequence, TypedDict

from mldb_v2.skeleton.common.diagnostic import Diagnostic
from mldb_v2.skeleton.common.ids import EntityKind, NamespaceId, StudyId
from mldb_v2.skeleton.repository.resolution import CanonicalDocument
from mldb_v2.skeleton.results.study_result import StudyResultStatus


class CanonicalListing(TypedDict):
    items: Sequence[CanonicalDocument]
    issues: Sequence[Diagnostic]


class CanonicalRepositoryListing(Protocol):
    """Broad read-only inventory, distinct from exact resolution."""

    def list_entities(
        self,
        *,
        kind: EntityKind | None = None,
        namespace: NamespaceId | None = None,
    ) -> CanonicalListing: ...

    def list_study_results(
        self,
        *,
        namespace: NamespaceId | None = None,
        study: StudyId | None = None,
        statuses: Collection[StudyResultStatus] | None = None,
        created_at_from: datetime | None = None,
        created_at_to: datetime | None = None,
        limit: int | None = None,
    ) -> CanonicalListing: ...
