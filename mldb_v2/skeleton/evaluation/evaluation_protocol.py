"""MLDB v2 reusable Evaluation Protocol definition shape."""

from typing import Literal, Mapping, NotRequired, TypeAlias, TypedDict

from mldb_v2.skeleton.common.ids import EvaluationProtocolId, TaskId
from mldb_v2.skeleton.common.parameters import PublicParameterDeclarations
from mldb_v2.skeleton.verification.executable_integrity import ExecutableSource


class EvaluationProtocolImplementation(TypedDict):
    entrypoint: Literal["evaluate"]
    sha256: NotRequired[str]
    sources: NotRequired[list[ExecutableSource]]


class EvaluationMetricDeclaration(TypedDict):
    type: Literal["integer", "number"]
    required: bool
    description: NotRequired[str]
    preference: NotRequired[Literal["higher", "lower", "neutral"]]


class EvaluationArtifactDeclaration(TypedDict):
    format: str
    schema: str
    required: bool
    description: NotRequired[str]

EvaluationMetricDeclarations: TypeAlias = Mapping[str, EvaluationMetricDeclaration]
EvaluationArtifactDeclarations: TypeAlias = Mapping[str, EvaluationArtifactDeclaration]


class EvaluationProtocol(TypedDict):
    schema: Literal["mjtensu.mldb-v2/evaluation-protocol/v1"]
    id: EvaluationProtocolId
    status: Literal["draft", "sealed"]
    task: TaskId
    name: str
    description: str
    implementation: EvaluationProtocolImplementation
    parameters: PublicParameterDeclarations
    metrics: EvaluationMetricDeclarations
    artifacts: EvaluationArtifactDeclarations
