"""MLDB v2 Corpus public shapes."""

from typing import Literal, NotRequired, TypeAlias, TypedDict

from mldb_v2.skeleton.common.ids import CorpusId, TaskId
from mldb_v2.skeleton.common.parameters import PublicParameterValue

# ``kind`` is required and non-empty by the Corpus format; remaining keys are domain-specific.
CorpusRepresentation: TypeAlias = dict[str, PublicParameterValue]
CorpusSplits: TypeAlias = dict[str, int]
CorpusBuilderParameters: TypeAlias = dict[str, PublicParameterValue]


class CorpusStorage(TypedDict):
    root_uri: str


class CorpusManifest(TypedDict):
    file: str
    sha256: NotRequired[str]
    entries: NotRequired[int]


class CorpusBuilder(TypedDict):
    entrypoint: Literal["build"]
    sha256: NotRequired[str]
    parameters: NotRequired[CorpusBuilderParameters]


class Corpus(TypedDict):
    schema: Literal["mjtensu.mldb-v2/corpus/v1"]
    id: CorpusId
    status: Literal["draft", "sealed"]
    task: TaskId
    description: str
    storage: CorpusStorage
    manifest: CorpusManifest
    representation: CorpusRepresentation
    splits: CorpusSplits
    builder: NotRequired[CorpusBuilder]
