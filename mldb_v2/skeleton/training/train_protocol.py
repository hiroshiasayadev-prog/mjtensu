"""MLDB v2 reusable Train Protocol definition shape."""

from typing import Literal, NotRequired, TypedDict

from mldb_v2.skeleton.common.ids import TaskId, TrainProtocolId
from mldb_v2.skeleton.common.parameters import PublicParameterDeclarations
from mldb_v2.skeleton.verification.executable_integrity import ExecutableSource


class TrainProtocolImplementation(TypedDict):
    entrypoint: Literal["train"]
    sha256: NotRequired[str]
    sources: NotRequired[list[ExecutableSource]]


class TrainProtocol(TypedDict):
    schema: Literal["mjtensu.mldb-v2/train-protocol/v1"]
    id: TrainProtocolId
    status: Literal["draft", "sealed"]
    task: TaskId
    name: str
    description: str
    implementation: TrainProtocolImplementation
    parameters: PublicParameterDeclarations
