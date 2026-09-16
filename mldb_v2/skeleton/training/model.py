"""MLDB v2 immutable learned Model identity shape."""

from typing import Literal, TypedDict

from mldb_v2.skeleton.common.ids import ModelId, TrainingResultId


class Model(TypedDict):
    schema: Literal["mjtensu.mldb-v2/model/v1"]
    id: ModelId
    training_result: TrainingResultId
