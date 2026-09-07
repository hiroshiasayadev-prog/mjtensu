"""Definition sealing lifecycle mutation boundary for MLDB reusable definitions.

This module fixes the lower-level application/domain operations that perform the one
supported reusable-definition lifecycle mutation, ``draft -> sealed``, for Architecture,
Train Protocol, Evaluation Protocol, and Study. A later public Controller
``seal_definition`` operation may dispatch to these entity-specific functions and
project their small result without owning a second sealing implementation.

Executable-definition sealing composes only freeze-existing typed resolution,
entity-specific metadata validation, canonical repository layout, mandatory asset pytest
verification, exact implementation-byte hashing, and complete canonical YAML replacement.
The replacement is a lifecycle patch of the authoritative authored YAML, not a
serialization of the normalized domain value as though that value represented every
legal authored field. Study sealing follows the same authored-metadata preservation rule
while composing its typed resolver and metadata validator; it deliberately does not
reuse Study execution preflight or invent an executable pytest/hash requirement.

There is no generic ``seal(kind, id)`` function here. Task and Corpus have no operation
in this module because their current lifecycle contracts do not define this transition.
The module also introduces no DefinitionManager, lifecycle state machine, registry,
hash service, YAML repository/codec API, transaction abstraction, Run allocation,
Study execution, Queue/Worker behavior, or model/result persistence.

Semantic YAML parsing/serialization and SHA-256 calculation are implementation-private.
For a draft mutation, the implementation retains the exact canonical YAML text, parses a
private full authored mapping from that snapshot, and verifies agreement between that
snapshot and the normalized definition obtained from the frozen typed resolver. Only the
lifecycle-owned fields are changed in the private authored representation: ``status`` for
all four kinds, plus ``implementation.sha256`` for executable definitions. Every other
legal authored field must remain semantically unchanged, including advisory public-
parameter metadata and supplemental human-readable metadata intentionally absent from
normalized execution values. Byte-identical YAML text, comments, whitespace, quoting,
and key order are not required to survive serialization. Every authoritative metadata
destination is derived only through the corresponding typed
:class:`~mldb.skeleton.repository.layout.RepositoryLayout` method and committed with
:class:`~mldb.skeleton.repository.ports.FilesystemPort.replace_text`.

Concrete implementations must privately serialize same-definition Controller sealing or
other coordinated mutation. This is not a public lock API and does not establish a
filesystem-wide authoring lock. For executable definitions, implementations must compare
the exact canonical implementation bytes from the pre-pytest snapshot with a post-pytest
read and with a final read immediately before metadata replacement; any observed
disagreement refuses sealing. The metadata snapshot must likewise still agree with the
snapshot associated with the resolved draft immediately before replacement so an
observed concurrent authoring mutation is refused rather than overwritten. V1 file-based
authoring requires the target definition and implementation not to be concurrently
mutated by an uncoordinated external editor/process during sealing. An external
change-and-restore (ABA) that is not observable through these reads is outside the v1
synchronization contract; stronger ABA-proofing would require a separate authoring/
concurrency contract rather than a hidden CAS, generation, watcher, or filesystem lock.
The byte snapshot is only an integrity/stability observation: asset-specific pytest still
exercises the canonical asset through the frozen verification runner and the normal
resolution/executable-loading boundary. Sealing does not substitute a temporary copied
implementation or an alternate ad-hoc import convention for the pytest target.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from ..common.ids import (
    ArchitectureId,
    EvaluationProtocolId,
    StudyId,
    TrainProtocolId,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from .sealing import PytestRunner


SealResult: TypeAlias = Literal["sealed", "already_sealed"]
"""Definitive result of one supported definition-sealing request.

``"sealed"`` means this call validated the current canonical draft, satisfied every
sealing gate applicable to that definition kind, and durably replaced its canonical YAML
with the validated sealed form.

``"already_sealed"`` is returned only for a canonical definition that was already
sealed when examined inside this operation's same-definition exclusion and still
satisfies the integrity contract applicable to that sealed definition. For executable
assets this includes exact agreement between the recorded ``implementation.sha256`` and
the current canonical sibling Python bytes. The replay path never runs pytest merely to
re-seal history and never rewrites sealed metadata to follow changed implementation
bytes.

This is intentionally only the lifecycle outcome needed by a later thin Controller
projection. Definition kind, ID, and public ``status=sealed`` are already known from the
entity-specific operation selected by that caller and are not duplicated in another
result DTO or mutation hierarchy.
"""


def seal_architecture(
    architecture_id: ArchitectureId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> SealResult:
    """Seal one exact canonical Architecture or validate an already-sealed replay.

    The implementation enters the private same-Architecture sealing exclusion before
    observing mutable lifecycle state. For draft-mutation snapshot agreement it retains
    the exact text from ``layout.architecture_metadata_path(architecture_id)``
    immediately before performing the fresh frozen
    :func:`mldb.skeleton.runtime.resolution.resolve_architecture`. Immediately after
    resolution it re-reads that canonical metadata path and requires the exact text to
    be unchanged before using the retained text as the authored snapshot. The private
    parser/normalizer for that snapshot must agree with the resolver-produced normalized
    Architecture. Successful resolution supplies canonical identity/path agreement,
    Architecture static metadata validation, required sibling-file existence, and
    referenced Task validity already owned by the frozen resolver. No second resolver,
    registry, public YAML representation, or direct YAML path derivation is permitted.

    If the resolved Architecture is already ``SEALED``, this function does not invoke
    pytest and does not construct or persist replacement metadata. It reads the exact
    current bytes from
    ``layout.architecture_implementation_path(architecture_id)`` through
    ``filesystem.read_bytes(...)``, calculates SHA-256 privately, and requires exact
    agreement with the already-recorded ``metadata.implementation.sha256``. Resolver
    success plus this sealed implementation-integrity agreement is sufficient for
    ``"already_sealed"``. Missing hash metadata, a hash mismatch, or any resolution/
    integrity failure is an operation failure; the sealed YAML must not be rewritten to
    match current bytes.

    If the resolved Architecture is ``DRAFT``, sealing follows this exact order:

    1. retain the freshly resolved draft Architecture, canonical sibling paths, and the
       agreed exact authored YAML text snapshot described above;
    2. parse that snapshot into an implementation-private full authored mapping and
       retain every authored field represented by it;
    3. read and retain the exact implementation-byte snapshot from
       ``layout.architecture_implementation_path(architecture_id)``;
    4. derive only ``layout.architecture_test_dir(architecture_id)`` and call the frozen
       ``verify_architecture_tests(...)`` with that exact ID/path, ``filesystem``, and
       ``runner``;
    5. require the verification result's ``successful`` property to be true; a missing
       directory, incomplete pytest invocation, zero collected tests, zero passed tests,
       any failure, or any error refuses sealing;
    6. after successful pytest, read the same canonical implementation path again and
       require byte-for-byte equality with the pre-pytest snapshot;
    7. calculate the lowercase SHA-256 hex digest from that stable tested byte snapshot;
    8. construct a new normalized Architecture domain value equal to the resolved draft
       in every field except ``status=SEALED`` and
       ``implementation.sha256=<calculated digest>``;
    9. apply the frozen ``validate_architecture_metadata(...)`` to that constructed
       sealed normalized value and require a valid report;
    10. in the private full authored mapping change only ``status`` from ``draft`` to
        ``sealed`` and ``implementation.sha256`` to the calculated digest; all other
        legal authored metadata remains semantically unchanged;
    11. immediately before commit, re-read both canonical inputs: require the metadata
        text to equal the original agreed authored snapshot and the implementation bytes
        to equal the same tested byte snapshot; any observed disagreement refuses
        sealing rather than overwriting the competing mutation; and
    12. privately serialize the complete patched authored mapping and commit it only to
        ``layout.architecture_metadata_path(architecture_id)`` using
        ``filesystem.replace_text(...)``. Serialization may change YAML presentation but
        must not change the semantics of any non-owned authored field.

    Successful return after step 12 is exactly ``"sealed"``. Pytest therefore always
    precedes hash establishment and metadata persistence, and the persisted digest is
    derived from the same implementation bytes whose stability was maintained across the
    pytest gate. No implementation bytes, test files, Task metadata, Run state, or other
    repository entity are mutated by this operation.
    """

    ...


def seal_train_protocol(
    protocol_id: TrainProtocolId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> SealResult:
    """Seal one exact canonical Train Protocol or validate an idempotent sealed replay.

    Under the private same-Protocol sealing exclusion, retain the exact canonical text
    from ``layout.train_protocol_metadata_path(protocol_id)``, then resolve ``protocol_id``
    through the frozen ``resolve_train_protocol(...)`` operation and re-read that metadata
    path. Draft mutation proceeds only when the text is unchanged and the private
    parser/normalizer of that retained full authored snapshot agrees with the
    resolver-produced normalized Train Protocol. Resolver success establishes canonical
    requested/recorded identity, static Train Protocol validation, sibling existence, and
    referenced Task validity; this boundary does not duplicate them or expose a raw YAML
    type publicly.

    An already-``SEALED`` protocol returns ``"already_sealed"`` only after the exact
    current bytes at ``layout.train_protocol_implementation_path(protocol_id)`` are read
    through ``filesystem``, privately SHA-256 hashed, and found equal to the persisted
    ``metadata.implementation.sha256``. The replay path does not rerun asset pytest and
    never updates a sealed hash to follow changed bytes.

    A ``DRAFT`` protocol uses the exact sealing order below:

    1. retain the resolved normalized draft and the agreed exact authored YAML text
       snapshot described above, then parse that snapshot into an implementation-private
       full authored mapping;
    2. read the exact canonical sibling implementation bytes as the pre-pytest snapshot;
    3. derive only ``layout.train_protocol_test_dir(protocol_id)`` and invoke the frozen
       ``verify_train_protocol_tests(...)`` gate for that exact ID/path;
    4. require ``verification.successful`` before any sealed metadata is persisted;
    5. re-read the canonical sibling after pytest and require byte-for-byte equality with
       the pre-pytest snapshot;
    6. calculate SHA-256 from that stable tested snapshot;
    7. construct a sealed normalized Train Protocol by changing only ``status=SEALED``
       and ``implementation.sha256`` to that digest, then require it to pass the frozen
       ``validate_train_protocol_metadata(...)``;
    8. in the private full authored mapping change only ``status`` and
       ``implementation.sha256``. Preserve every other legal authored field semantically,
       including each parameter declaration's advisory ``description``, ``type``,
       ``minimum``, ``maximum``, ``suggested``, and any supplemental human-readable
       metadata intentionally absent from the normalized execution subset;
    9. immediately before metadata replacement, require the canonical metadata text to
       equal the original agreed authored snapshot and the canonical implementation bytes
       to equal the same tested snapshot; any observed disagreement refuses sealing; and
    10. privately serialize the complete patched authored mapping and commit it only to
        ``layout.train_protocol_metadata_path(protocol_id)`` through
        ``filesystem.replace_text(...)``. YAML presentation may change, but non-owned
        authored semantics must not.

    Only successful completion of the canonical metadata replacement returns
    ``"sealed"``. Pytest runner details, YAML codec details, SHA-256 helper choice, and
    same-definition exclusion mechanics remain private implementation concerns and do
    not create new public services.
    """

    ...


def seal_evaluation_protocol(
    protocol_id: EvaluationProtocolId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> SealResult:
    """Seal one exact Evaluation Protocol or validate an already-sealed replay.

    Inside the private same-Protocol sealing exclusion, retain the exact canonical text
    from ``layout.evaluation_protocol_metadata_path(protocol_id)``, perform the fresh
    frozen ``resolve_evaluation_protocol(...)`` call, then re-read the same metadata path.
    Draft mutation proceeds only when the text is unchanged and the private
    parser/normalizer of the retained full authored snapshot agrees with the
    resolver-produced normalized Evaluation Protocol. Successful resolution establishes
    the canonical identity, metadata-local validation, canonical sibling presence, and
    referenced Task validity already owned by the resolver.

    For an already-``SEALED`` Evaluation Protocol, no pytest is invoked and no metadata
    replacement occurs. The operation reads the exact bytes at
    ``layout.evaluation_protocol_implementation_path(protocol_id)``, privately computes
    SHA-256, and returns ``"already_sealed"`` only when that digest exactly equals the
    persisted ``metadata.implementation.sha256``. A mismatch is an integrity failure,
    never permission to rewrite immutable metadata.

    For a ``DRAFT`` Evaluation Protocol, the mandatory order is:

    1. retain the resolved normalized draft and the agreed exact authored YAML text
       snapshot, then parse that snapshot into an implementation-private full authored
       mapping;
    2. read and retain the exact pre-pytest canonical implementation bytes;
    3. derive only ``layout.evaluation_protocol_test_dir(protocol_id)`` and call the
       frozen ``verify_evaluation_protocol_tests(...)`` gate;
    4. require ``verification.successful`` before lifecycle/hash persistence;
    5. re-read the canonical implementation after pytest and require exact equality with
       the pre-pytest snapshot;
    6. calculate SHA-256 from that stable tested snapshot;
    7. construct a sealed normalized Evaluation Protocol differing from the resolved
       draft only by ``status=SEALED`` and the calculated ``implementation.sha256``, then
       require the frozen ``validate_evaluation_protocol_metadata(...)`` to accept it;
    8. in the private full authored mapping change only ``status`` and
       ``implementation.sha256``. Preserve every other legal authored field semantically,
       including all advisory public-parameter metadata that normalized
       ``PublicParameterDeclaration`` intentionally does not represent;
    9. immediately before commit, require the canonical metadata text to equal the
       original agreed authored snapshot and the canonical implementation bytes to equal
       the same tested snapshot; any observed disagreement refuses sealing; and
    10. privately serialize the complete patched authored mapping and replace only
        ``layout.evaluation_protocol_metadata_path(protocol_id)`` through
        ``filesystem.replace_text(...)``. YAML presentation may change, but non-owned
        authored semantics must not.

    Successful replacement returns ``"sealed"``. The operation does not load or invoke
    ``evaluate``, validate concrete Evaluation results, import artifacts, allocate Runs,
    or touch Queue/Worker state.
    """

    ...


def seal_study(
    study_id: StudyId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> SealResult:
    """Seal one exact canonical Study without promoting execution preflight into sealing.

    Inside the private same-Study sealing exclusion, retain the exact canonical text from
    ``layout.study_metadata_path(study_id)``, perform a fresh frozen
    ``resolve_study(study_id, layout, filesystem)``, then re-read the same metadata path.
    Draft mutation proceeds only when the text is unchanged and the private
    parser/normalizer of the retained full authored snapshot agrees with the
    resolver-produced normalized Study. The resolver remains the complete
    reference-validity boundary used here: it establishes Study canonical identity and
    static metadata validity, and requires successful typed resolution of the direct
    Study references appropriate to its model-source mode. Consequently sealing inherits
    the validity/integrity already owned by those typed resolvers (for example immutable
    Corpus resolution and existing-Model lineage facts) without inventing another Study
    dependency traversal.

    This operation deliberately does **not** call ``preflight_study_execution(...)``.
    Current execution-preflight-only rules therefore remain deferred until Study Run
    launch: common Task agreement, executable ``SEALED`` requirements, executable
    implementation-hash verification of referenced assets, Study-to-Protocol parameter
    publication/default resolution, existing-Model learned-weight byte/hash/size
    verification beyond resolver-owned facts, and other execution-readiness checks are
    not moved into Study sealing. Asset-specific pytest is never run for Study and Study
    has no ``implementation.sha256`` field.

    If the freshly resolved Study is already ``SEALED``, resolver success is the
    applicable current Study-definition integrity check at this boundary. Return
    ``"already_sealed"`` without rewriting canonical YAML. The operation does not inspect
    or repair referenced executable hashes merely to make an already-sealed Study replay
    succeed; those remain properties of the referenced definitions and execution
    preflight.

    If the Study is ``DRAFT``:

    1. retain the freshly resolved normalized draft and the agreed exact authored YAML
       text snapshot, then parse that snapshot into an implementation-private full
       authored mapping;
    2. construct a new normalized Study equal in every field except ``status=SEALED``;
    3. require the frozen ``validate_study_metadata(...)`` to accept that sealed
       normalized value;
    4. in the private full authored mapping change only ``status`` from ``draft`` to
       ``sealed``; every other legal authored field remains semantically unchanged;
    5. immediately before replacement, re-read the canonical Study metadata and require
       its text to equal the original agreed authored snapshot. An observed concurrent
       authoring mutation refuses sealing rather than being overwritten; and
    6. privately serialize the complete patched authored mapping and commit it only to
       ``layout.study_metadata_path(study_id)`` through
       ``filesystem.replace_text(...)``. YAML presentation may change, but non-owned
       authored semantics must not.

    Successful canonical replacement returns ``"sealed"``. Study sealing allocates no
    Study Run, materializes no plan, resolves no parameter defaults, imports no Python,
    invokes no executable code, and touches no Queue/Worker/model/result persistence.
    """

    ...
