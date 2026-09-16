"""Pure semantic readiness boundary for immutable Study Plans.

Training-source trials expose training first and evaluations only after canonical completed Model
lineage exists. Existing-Model trials expose evaluations immediately after Model integrity is
accepted. Evaluation siblings do not block each other. Cancelling admits nothing new. Terminal
training failure/cancellation yields only the stable upstream skip reasons.

Admission ownership is operational and stays outside this semantic readiness boundary. The Study
driver handles cancellation by observing deterministic backend ownership before closing never-admitted
pending stages as study-cancelled.
"""

from typing import Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.skeleton.common.ids import EvaluationCoordinateId, TrialId
from mldb_v2.skeleton.results.study_result import StudyResult
from mldb_v2.skeleton.study.plan import StudyPlan


class TrainingStageRef(TypedDict):
    kind: Literal["training"]
    trial: TrialId


class EvaluationStageRef(TypedDict):
    kind: Literal["evaluation"]
    trial: TrialId
    coordinate: EvaluationCoordinateId


PlannedStageRef: TypeAlias = TrainingStageRef | EvaluationStageRef
UpstreamSkipReason: TypeAlias = Literal["upstream_failed", "upstream_cancelled"]


class ReadinessSkip(TypedDict):
    stage: PlannedStageRef
    reason: UpstreamSkipReason


class ExecutionReadiness(TypedDict):
    ready: list[PlannedStageRef]
    skipped: list[ReadinessSkip]


class ExecutionReadinessResolver(Protocol):
    def derive(
        self,
        *,
        plan: StudyPlan,
        result: StudyResult,
    ) -> ExecutionReadiness: ...
