"""Public Python signatures for MLDB executable-asset loading.

This skeleton fixes the runtime boundary from a prepared executable-asset runtime handle
to its declared Python callable. It intentionally does not resolve IDs or
repository paths, parse metadata, run pytest, mutate sealing state, invoke executable
entrypoints, load Model weights, or define loader/repository abstractions.
"""

from __future__ import annotations

from ..catalog.architecture import ArchitectureBuild
from ..evaluation.interface import EvaluationEntrypoint
from ..training.protocol import TrainEntrypoint
from .catalog_handles import ArchitectureHandle
from .definition_handles import EvaluationProtocolHandle, TrainProtocolHandle


def load_architecture_build(
    architecture: ArchitectureHandle,
) -> ArchitectureBuild:
    """Expose the declared Architecture v1 ``build`` callable for a runtime asset.

    ``architecture`` is the complete runtime handoff: its validated metadata fixes the
    expected v1 ``build`` entrypoint and its required ``implementation_path`` fixes the
    exact Python file to load. For resolver output that path is the canonical sibling;
    for Worker materialization it is the integrity-verified local execution copy of the
    Controller-selected bytes. Callers must not supply a second repository, path, module
    name, entrypoint name, expected callable type, or loader object.

    A successful return guarantees that every implementation-integrity requirement
    applicable to the supplied Architecture has been accepted before executable code is
    exposed. In particular, a sealed Architecture's recorded implementation SHA-256
    matches the actual bytes at ``implementation_path`` regardless of canonical or
    Worker-local placement. Draft loading remains possible where the surrounding
    operation permits draft execution and therefore has no sealed-hash requirement.

    Success further guarantees that the selected implementation has loaded, the declared
    ``build`` entrypoint exists and is callable, and the returned object is compatible
    with the no-argument invocation shape of the frozen :class:`ArchitectureBuild`
    boundary. The callable remains contractually required to return ``torch.nn.Module``
    when invoked; this loader does not pre-invoke ``build()`` merely to expose it.
    Validation of an actual constructed result and model-family-specific behavior belongs
    to the Architecture build interface and its caller/verification context.

    Merely importing this module must not import or execute any asset-owned Python
    implementation. Dynamic import occurs only when this operation is called.

    Missing implementation files, integrity mismatch, Python import failure, missing or
    non-callable entrypoints, and incompatible entrypoint shape are operation failures.
    Concrete exception classes remain implementation-owned by the runtime contract and
    are intentionally not frozen here.
    """

    ...


def load_train_entrypoint(
    protocol: TrainProtocolHandle,
) -> TrainEntrypoint:
    """Expose the declared Train Protocol v1 ``train`` callable for a runtime asset.

    ``protocol`` is the complete runtime handoff: its validated metadata fixes the
    expected v1 ``train`` entrypoint and its required ``implementation_path`` fixes the
    exact Python file to load. For resolver output that path is the canonical sibling;
    for Worker materialization it is the integrity-verified local execution copy of the
    Controller-selected bytes. Callers must not supply a second repository, path, module
    name, entrypoint name, expected callable type, selected Task/Corpus/Architecture,
    context, or loader object.

    A successful return guarantees that every implementation-integrity requirement
    applicable to the supplied Train Protocol has been accepted before executable code
    is exposed. In particular, a sealed Train Protocol's recorded implementation
    SHA-256 matches the actual bytes at ``implementation_path`` regardless of canonical
    or Worker-local placement. Draft loading remains possible where the surrounding
    operation permits draft execution and therefore has no sealed-hash requirement.

    Success further guarantees that the selected implementation has loaded, the declared
    ``train`` entrypoint exists and is callable, and the returned object is compatible
    with the one-argument invocation shape of the frozen :class:`TrainEntrypoint`
    boundary. The callable remains contractually required to accept one ``TrainContext``
    and return ``torch.nn.Module`` when invoked; this loader does not construct a context,
    invoke ``train()``, validate its returned module, check Architecture compatibility,
    serialize canonical weights, finalize a Training Run, or create a Model.

    Merely importing this module must not import or execute any asset-owned sibling Python
    implementation. Dynamic import occurs only when this operation is called.

    Missing implementation files, integrity mismatch, Python import failure, missing or
    non-callable entrypoints, and incompatible entrypoint shape are operation failures.
    Concrete exception classes remain implementation-owned by the runtime contract and
    are intentionally not frozen here.
    """

    ...


def load_evaluation_entrypoint(
    protocol: EvaluationProtocolHandle,
) -> EvaluationEntrypoint:
    """Expose the declared Evaluation Protocol v1 ``evaluate`` callable for a runtime asset.

    ``protocol`` is the complete runtime handoff: its validated metadata fixes the
    expected v1 ``evaluate`` entrypoint and its required ``implementation_path`` fixes
    the exact Python file to load. For resolver output that path is the canonical sibling;
    for Worker materialization it is the integrity-verified local execution copy of the
    Controller-selected bytes. Callers must not supply a second repository, path, module
    name, entrypoint name, expected callable type, selected Task/Corpus/Model, context,
    or loader object.

    A successful return guarantees that every implementation-integrity requirement
    applicable to the supplied Evaluation Protocol has been accepted before executable
    code is exposed. In particular, a sealed Evaluation Protocol's recorded
    implementation SHA-256 matches the actual bytes at ``implementation_path`` regardless
    of canonical or Worker-local placement. Draft loading remains possible where the
    surrounding operation permits draft execution and therefore has no sealed-hash
    requirement.

    Success further guarantees that the selected implementation has loaded, the declared
    ``evaluate`` entrypoint exists and is callable, and the returned object is compatible
    with the one-argument invocation shape of the frozen :class:`EvaluationEntrypoint`
    boundary. The callable remains contractually required to accept one
    ``EvaluationContext`` and return ``EvaluationResult`` when invoked; this loader does
    not construct a context, load a Model, invoke ``evaluate()``, validate the returned
    result, import artifacts, or finalize an Evaluation Run.

    Merely importing this module must not import or execute any asset-owned sibling Python
    implementation. Dynamic import occurs only when this operation is called.

    Missing implementation files, integrity mismatch, Python import failure, missing or
    non-callable entrypoints, and incompatible entrypoint shape are operation failures.
    Concrete exception classes remain implementation-owned by the runtime contract and
    are intentionally not frozen here.
    """

    ...
