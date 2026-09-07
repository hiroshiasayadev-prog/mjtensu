"""Public Python signatures for MLDB Evaluation Protocol execution.

This skeleton fixes the common callable boundary between resolved Evaluation
execution and one Evaluation Protocol implementation. It intentionally does not
define Evaluation Protocol metadata, resolve public parameters, load Models,
validate returned results, import or hash artifacts, manage Evaluation Run
lifecycle, access repositories, or implement evaluation algorithms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from ..common.parameters import PublicParameterValue, ResolvedPublicParameters
from ..model.loading import ModelHandle
from ..runtime.catalog_handles import CorpusHandle, TaskHandle


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Resolved request supplied to one Evaluation Protocol ``evaluate`` entrypoint.

    ``task``, ``corpus``, ``model``, and the complete public-parameter mapping have
    already passed launch preflight before this value is constructed.

    ``model`` is the resolved learned-Model identity boundary. Protocol code that
    needs the learned ``torch.nn.Module`` can call ``load_model(context.model)``;
    this context therefore does not expose Training Run,
    Architecture, canonical weights, repository, or loader dependencies separately.

    ``parameters`` contains every published Evaluation Protocol public parameter
    exactly once after common resolution. Evaluation Run v1 has no universal seed;
    a protocol that requires caller-controlled stochastic seeding publishes and
    consumes an ordinary ``parameters["seed"]`` value instead.

    ``work_dir`` is the concrete Evaluation Run ``work/`` directory available for
    protocol-owned temporary, diagnostic, and candidate formal-result files. Writing
    a file there does not make it a canonical Evaluation Run artifact.
    """

    task: TaskHandle
    corpus: CorpusHandle
    model: ModelHandle
    parameters: ResolvedPublicParameters
    work_dir: Path


@dataclass(frozen=True, slots=True)
class UnavailableOutput:
    """Explicit unavailability report for one declared formal output.

    ``output`` identifies one declaration using ``metrics.<key>`` or
    ``artifacts.<key>``. ``type`` is a short machine-readable reason string with no
    universal MLDB v1 enum, and ``message`` is the concise human-readable explanation.

    Declaration lookup, conflict checks, required/optional behavior, and resulting
    Evaluation Run status belong to later result validation rather than this value.
    """

    output: str
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Candidate formal outputs returned by one Evaluation Protocol invocation.

    ``metrics`` contains candidate scalar metric values. Runtime result validation
    later enforces declared keys, finite ``int``/``float`` values, rejects booleans,
    NaN, and infinities, and handles missing/unavailable declarations.

    ``artifacts`` maps declared candidate artifact keys to files beneath the supplied
    ``EvaluationContext.work_dir``. These paths still identify protocol-produced work
    files; they are not canonical Evaluation Run ``artifacts/`` paths and carry no
    hash, byte-size, format, or schema metadata at this boundary.

    ``unavailable_outputs`` is an ordered sequence of explicit unavailable-output
    reports. The concrete mapping/sequence implementations and any defensive-copy
    strategy are intentionally not fixed by this public container.
    """

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]
    unavailable_outputs: Sequence[UnavailableOutput]


EvaluationEntrypoint: TypeAlias = Callable[[EvaluationContext], EvaluationResult]
"""Resolved Evaluation Protocol v1 ``evaluate`` callable.

Each sibling Evaluation Protocol implementation exposes a function named ``evaluate``
matching this callable shape. A successful call accepts exactly one
:class:`EvaluationContext` and returns one :class:`EvaluationResult` containing only
candidate formal outputs and explicit unavailability reports.

Loading the sibling implementation, validating callable presence, invoking it,
translating exceptions or invalid returns into Evaluation Run failure, validating
formal outputs, and materializing accepted artifacts belong to later runtime,
execution, and result-validation boundaries. This alias is independent of Evaluation
Protocol metadata definitions.
"""
