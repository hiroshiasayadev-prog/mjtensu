"""MLDB v2 terminal Evaluation Result public shape."""

from typing import Literal, Mapping, TypedDict

from mldb_v2.skeleton.common.diagnostic import Diagnostic
from mldb_v2.skeleton.common.ids import (
    CorpusId,
    EvaluationCoordinateId,
    EvaluationProtocolId,
    EvaluationResultId,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TaskId,
    TrialId,
)
from mldb_v2.skeleton.common.parameters import ResolvedPublicParameters
from mldb_v2.skeleton.results.attempt_summary import AttemptSummary
from mldb_v2.skeleton.storage.artifact_reference import ArtifactRef


class EvaluationArtifactRef(ArtifactRef):
    format: str
    schema: str


class EvaluationResultPayload(TypedDict):
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, EvaluationArtifactRef]


class EvaluationResult(TypedDict):
    schema: Literal["mjtensu.mldb-v2/evaluation-result/v1"]
    id: EvaluationResultId
    study_result: StudyResultId
    plan: StudyPlanId
    trial: TrialId
    coordinate: EvaluationCoordinateId
    stage: str
    model: ModelId
    task: TaskId
    corpus: CorpusId
    evaluation_protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters
    source_commit: str
    attempts: list[AttemptSummary]
    status: Literal["completed", "failed", "cancelled"]
    diagnostic: Diagnostic | None
    result: EvaluationResultPayload | None
