"""Public signatures for registered immutable MLDB Corpora.

This module models the catalog-side Corpus metadata boundary only. A Corpus is a
registered immutable materialized data asset bound to one Task. Mutable annotation
stores, builder execution, canonical filesystem resolution, SQLite access/querying,
sample iteration, batching, preprocessing, augmentation, and worker-local caching are
outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from ..common.errors import ValidationReport
from ..common.ids import CorpusId, TaskId


@dataclass(frozen=True, slots=True)
class CorpusArtifact:
    """Identity and integrity metadata for the materialized Corpus artifact.

    ``format`` is the metadata-declared artifact format. Corpus v1 fixes it to
    ``"sqlite"``; a future artifact format requires a later schema contract rather
    than widening this v1 signature in place.

    ``sha256`` is the authoritative digest recorded for the immutable artifact.
    ``bytes`` is the optional recorded byte size for the same artifact. This object
    contains metadata only: no path, file object, database connection, or integrity
    calculation result is embedded here.
    """

    format: Literal["sqlite"]
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class CorpusDataSpec:
    """Selector for the concrete physical-data contract inside a Corpus artifact.

    ``schema`` remains an opaque versioned schema identifier because concrete data
    schemas are extensible and are not one closed MLDB-wide enum. ``table`` names the
    canonical sample table selected by that schema.

    Concrete columns, row types, payload decoding, SQL types, and query behavior belong
    to the selected data-schema contract and its runtime implementation, not to this
    generic catalog representation.
    """

    schema: str
    table: str


@dataclass(frozen=True, slots=True)
class Corpus:
    """Registered immutable Corpus metadata.

    The object represents the authoritative catalog metadata for one frozen materialized
    Corpus; it does not represent a mutable annotation database or an opened runtime
    resource.

    ``task`` binds the Corpus to exactly one Task by ``TaskId`` without depending on the
    Wave 1-A Task object contract.

    ``representation`` is intentionally schema-directed metadata rather than a generic
    image/tensor dataclass. The selected ``data.schema`` defines which representation
    keys are required and how they are interpreted. For example, the initial image-
    classification schema requires image-specific keys, but those keys are not universal
    Corpus fields.

    ``builder_parameters`` records the effective materialization parameters that are not
    recoverable from builder source alone. Their value vocabulary is deliberately not
    equated with the shared public-protocol parameter contract: current Corpus Design
    Records do not define a common typed parameter language for builders.

    ``splits`` is the normalized split inventory, mapping each Corpus-defined non-empty
    split name to its sample count. Split names are not a global enum.

    ``statistics`` and ``origin`` are optional descriptive metadata whose detailed shape
    is not fixed by the generic Corpus contract.
    """

    schema: Literal["mjtensu.mldb/corpus/v1"]
    id: CorpusId
    task: TaskId
    artifact: CorpusArtifact
    data: CorpusDataSpec
    representation: Mapping[str, object]
    builder_parameters: Mapping[str, object]
    splits: Mapping[str, int]
    description: str | None = None
    statistics: object | None = None
    origin: object | None = None


def validate_corpus(corpus: Corpus) -> ValidationReport:
    """Validate static Corpus metadata without resolving or opening runtime assets.

    For the current ``mjtensu.mldb/corpus/v1`` contract, validation covers outer
    metadata-local invariants such as the schema identifier, ``sqlite`` artifact-format
    requirement, SHA-256 metadata shape, optional artifact byte-size validity,
    data-schema/table metadata, and non-empty Corpus-defined split names with valid
    sample counts. Concrete ``data.schema``-specific representation rules are not added
    to the generic Corpus contract here.

    This function does not resolve the referenced Task, derive canonical sibling paths,
    read files, calculate artifact hashes or byte sizes, inspect SQLite tables/columns,
    compare split counts with rows, validate image-classification targets/class indices,
    execute the builder, or create a runtime Corpus handle. Those checks require
    repository/runtime or concrete data-schema resources outside this static domain
    boundary.
    """

    ...
