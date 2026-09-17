"""Backend-neutral Study execution identity and observation values."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from mldb_v2.src.common.ids import StudyPlanId, StudyResultId
from mldb_v2.src.results.study_result import StudyResult
from mldb_v2.src.study.plan import StudyPlan


class StudyExecutionKey(TypedDict):
    """Exact canonical identity of one backend-owned Study execution."""

    study_result: StudyResultId
    plan: StudyPlanId
    source_commit: str


class BackendStudyExecutionObservation(TypedDict):
    """Backend-neutral observation of one native Study execution container."""

    state: Literal["active", "terminal"]
    status: Literal["active", "completed", "failed", "cancelled"]
    key: StudyExecutionKey
    backend: str
    execution_id: str


class StudyExecutionBackendPort(Protocol):
    """Optional W011 capability for backend-native Study execution containers."""

    def ensure_study_execution(
        self, *, plan: StudyPlan, study_result: StudyResult
    ) -> BackendStudyExecutionObservation:
        ...

    def observe_study_execution(
        self, *, plan: StudyPlan, study_result: StudyResult
    ) -> BackendStudyExecutionObservation | None:
        ...
