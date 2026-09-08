"""Runtime representations for MLDB Catalog entities.

The same public handle types support two legal construction sources. Canonical typed
resolvers produce handles whose repository-provenance paths and execution-resource
paths are canonical repository locations. A distributed Worker may instead construct
the same handles from Controller-selected validated metadata plus exact immutable bytes
after verifying and materializing those bytes locally; repository-provenance paths may
then be unavailable and Worker-local execution paths are not canonical locations.

Handle construction itself performs no resolution, filesystem access, integrity
calculation, executable loading, SQLite opening, or compatibility validation. Callers
must construct these values only from resources that already satisfied canonical
resolution or the assigned Worker materialization/integrity contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..catalog.architecture import Architecture
from ..catalog.corpus import Corpus
from ..catalog.task import Task


@dataclass(frozen=True, slots=True)
class TaskHandle:
    """Validated Task metadata plus optional repository provenance.

    ``metadata_path`` is the authoritative canonical Task YAML location for a
    resolver-produced handle. It is ``None`` for a Worker-materialized handle because
    the validated Task metadata is assigned by value and no synthetic YAML path is
    required or permitted as canonical provenance.

    Task has no sibling executable or materialized artifact in the current
    repository contract, so no additional resource abstraction is exposed here.
    """

    metadata: Task
    metadata_path: Path | None


@dataclass(frozen=True, slots=True)
class CorpusHandle:
    """Validated immutable Corpus plus its execution artifact and provenance.

    For a resolver-produced handle, ``metadata_path`` and ``builder_path`` identify the
    canonical YAML and sibling Python builder, and ``artifact_path`` is the canonical
    immutable SQLite artifact. For a Worker-materialized handle, ``metadata_path`` and
    ``builder_path`` are ``None`` while ``artifact_path`` is the Worker-local copy of
    the exact Controller-selected artifact bytes after required integrity verification.
    That local path is an execution resource, not canonical repository provenance.

    The Corpus builder is an authoring/materialization source and is not required for
    Training or Evaluation execution. The handle contains locations only; it does not
    contain an opened SQLite connection, cursor, iterator, DataLoader, cache, or builder
    execution state.
    """

    metadata: Corpus
    metadata_path: Path | None
    artifact_path: Path
    builder_path: Path | None


@dataclass(frozen=True, slots=True)
class ArchitectureHandle:
    """Validated Architecture plus the Python implementation used for execution.

    For a resolver-produced handle, ``metadata_path`` is the authoritative canonical
    YAML and ``implementation_path`` is the canonical sibling Python file. For a
    Worker-materialized handle, ``metadata_path`` is ``None`` and
    ``implementation_path`` is the Worker-local copy of the exact Controller-selected
    implementation bytes after integrity verification. The latter is not a canonical
    repository location.

    Importing the implementation, exposing its declared ``build`` callable,
    constructing ``torch.nn.Module``, and loading learned state belong to later
    executable-loader or Model-runtime boundaries.
    """

    metadata: Architecture
    metadata_path: Path | None
    implementation_path: Path
