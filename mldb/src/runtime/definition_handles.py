"""Runtime representations for reusable MLDB definitions.

Train Protocol and Evaluation Protocol handles are shared by canonical resolver output
and distributed Worker execution. Resolver-produced handles carry canonical repository
provenance and implementation paths; Worker-materialized handles carry Controller-
selected validated metadata plus verified local implementation paths and no synthetic
repository provenance. Study remains Controller-side and resolver-produced only.

Handle construction itself performs no resolution, filesystem access, YAML parsing,
integrity calculation, executable loading, compatibility judgment, public-parameter
resolution, Study materialization, or execution. Callers must construct these values
only from resources already accepted by the applicable canonical-resolution or Worker
materialization/integrity contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..evaluation.protocol import EvaluationProtocol
from ..study.definition import Study
from ..training.protocol import TrainProtocol


@dataclass(frozen=True, slots=True)
class TrainProtocolHandle:
    """Validated Train Protocol plus the Python implementation used for execution.

    For resolver-produced handles, ``metadata_path`` is the authoritative canonical
    YAML and ``implementation_path`` is the canonical sibling Python file. Successful
    resolution establishes requested/recorded/canonical identity agreement and declared
    upstream-reference validity. For Worker-materialized handles, ``metadata_path`` is
    ``None`` and ``implementation_path`` is the verified Worker-local copy of the exact
    Controller-selected implementation bytes; that path is not canonical provenance.

    This handle is not by itself an execution-readiness proof. Sealed-state requirements,
    implementation integrity before executable consumption, concrete
    Task/Corpus/Architecture compatibility, public-parameter resolution, executable
    loading, and execution remain applicable runtime responsibilities.
    """

    metadata: TrainProtocol
    metadata_path: Path | None
    implementation_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationProtocolHandle:
    """Validated Evaluation Protocol plus the Python implementation used for execution.

    For resolver-produced handles, ``metadata_path`` is the authoritative canonical
    YAML and ``implementation_path`` is the canonical sibling Python file. Successful
    resolution establishes requested/recorded/canonical identity agreement and declared
    upstream-reference validity. For Worker-materialized handles, ``metadata_path`` is
    ``None`` and ``implementation_path`` is the verified Worker-local copy of the exact
    Controller-selected implementation bytes; that path is not canonical provenance.

    This handle is not by itself an execution-readiness proof. Sealed-state requirements,
    implementation integrity before executable consumption, concrete Task/Model/Corpus
    compatibility, public-parameter resolution, executable loading, result validation,
    and execution remain applicable runtime responsibilities.
    """

    metadata: EvaluationProtocol
    metadata_path: Path | None
    implementation_path: Path


@dataclass(frozen=True, slots=True)
class StudyHandle:
    """Successfully resolved Study metadata and its canonical YAML location.

    ``metadata`` is the entity-specifically validated Study definition and
    ``metadata_path`` is the authoritative canonical Study YAML location.

    Successful resolution establishes agreement among the requested typed Study ID,
    ``metadata.id``, and canonical repository identity/location, together with the
    resolver-owned validation of declared upstream references applicable to the Study.
    Resolved Corpus, Architecture, Protocol, and Model dependencies are intentionally
    not embedded into this handle.

    This handle is not an execution-readiness proof. A valid ``draft`` Study may
    resolve successfully. Requiring ``sealed`` state for execution, cross-asset Task
    compatibility, applicable integrity checks, public-parameter resolution, existing
    Model lineage readiness, Study preflight, plan materialization, Run allocation,
    queue state, and execution remain later runtime responsibilities.
    """

    metadata: Study
    metadata_path: Path
