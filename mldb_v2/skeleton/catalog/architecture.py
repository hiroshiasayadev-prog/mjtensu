"""MLDB v2 Architecture public shapes."""

from typing import Literal, NotRequired, TypeAlias, TypedDict

from mldb_v2.skeleton.common.ids import ArchitectureId, TaskId
from mldb_v2.skeleton.common.parameters import PublicParameterValue
from mldb_v2.skeleton.verification.executable_integrity import ExecutableSource

# ``kind`` is required and non-empty by the Architecture format; remaining keys stay open.
ArchitectureInterfaceEndpoint: TypeAlias = dict[str, PublicParameterValue]
ArchitectureParameters: TypeAlias = dict[str, PublicParameterValue]


class ArchitectureImplementation(TypedDict):
    framework: Literal["pytorch"]
    entrypoint: Literal["build"]
    sha256: NotRequired[str]
    sources: NotRequired[list[ExecutableSource]]


class ArchitectureInterface(TypedDict):
    input: ArchitectureInterfaceEndpoint
    output: ArchitectureInterfaceEndpoint


class ArchitectureStructure(TypedDict):
    summary: str
    traits: NotRequired[list[str]]


class Architecture(TypedDict):
    schema: Literal["mjtensu.mldb-v2/architecture/v1"]
    id: ArchitectureId
    status: Literal["draft", "sealed"]
    task: TaskId
    name: str
    family: str
    description: str
    implementation: ArchitectureImplementation
    interface: ArchitectureInterface
    structure: ArchitectureStructure
    parameters: NotRequired[ArchitectureParameters]
