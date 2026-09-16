"""Canonical record creation and Study Result transition boundary."""

import os
from typing import Literal, Protocol, TypeAlias

from mldb_v2.skeleton.common.ids import (
    EntityKind,
    EvaluationResultId,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TrainingResultId,
)
from mldb_v2.skeleton.repository.resolution import CanonicalDocument

ImmutableCanonicalKind: TypeAlias = Literal[
    EntityKind.STUDY_PLAN,
    EntityKind.TRAINING_RESULT,
    EntityKind.MODEL,
    EntityKind.EVALUATION_RESULT,
]
ImmutableCanonicalId: TypeAlias = (
    StudyPlanId | TrainingResultId | ModelId | EvaluationResultId
)


class CanonicalRecordValidator(Protocol):
    """Owning-domain validation port for complete immutable canonical records."""

    def validate(
        self,
        *,
        kind: ImmutableCanonicalKind,
        entity_id: ImmutableCanonicalId,
        document: CanonicalDocument,
    ) -> None: ...


class CanonicalRepositoryWriter(Protocol):
    """Validated canonical writes with idempotence/lifecycle semantics from the specs."""

    def __init__(
        self,
        repository_root: str | os.PathLike[str],
        *,
        record_validator: CanonicalRecordValidator,
    ) -> None: ...

    def create_immutable(
        self,
        *,
        kind: ImmutableCanonicalKind,
        entity_id: ImmutableCanonicalId,
        document: CanonicalDocument,
    ) -> CanonicalDocument: ...

    def create_study_result(
        self,
        *,
        entity_id: StudyResultId,
        document: CanonicalDocument,
    ) -> CanonicalDocument: ...

    def replace_nonterminal_study_result(
        self,
        *,
        entity_id: StudyResultId,
        replacement: CanonicalDocument,
    ) -> CanonicalDocument: ...
