"""MLDB v2 immutable browser/deployment runtime-model catalog shape."""

from typing import Literal, Mapping, NotRequired, TypedDict

from mldb_v2.skeleton.common.ids import RuntimeModelId


class ArtifactRef(TypedDict):
    uri: str
    bytes: int
    sha256: str


class RuntimeModel(TypedDict):
    schema: Literal["mjtensu.mldb-v2/runtime-model/v1"]
    id: RuntimeModelId
    name: str
    description: str
    role: Literal["detector", "tile-classifier", "red-five-classifier"]
    format: Literal["onnx"]
    runtime_spec: str
    artifact: ArtifactRef
    provenance: NotRequired[Mapping[str, object]]
