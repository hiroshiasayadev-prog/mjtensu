"""Typed exact canonical repository resolution boundary."""

from typing import Mapping, Protocol, TypeAlias

from mldb_v2.skeleton.common.ids import EntityKind, NamespaceId, TypedEntityId

CanonicalEntityId: TypeAlias = NamespaceId | TypedEntityId
CanonicalDocument: TypeAlias = Mapping[str, object]


class CanonicalRepositoryResolver(Protocol):
    """Resolve only the fixed canonical path for one exact typed identity."""

    def resolve(
        self,
        *,
        kind: EntityKind,
        entity_id: CanonicalEntityId,
    ) -> CanonicalDocument: ...
