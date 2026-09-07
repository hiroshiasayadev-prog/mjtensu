"""Public Python signatures for MLDB Evaluation Protocol execution.

Implementation of the frozen callable boundary between resolved Evaluation execution
and one Evaluation Protocol implementation. This module intentionally contains only
value/container definitions and no repository, loading, validation, or lifecycle I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from ..common.parameters import PublicParameterValue, ResolvedPublicParameters

if TYPE_CHECKING:
    from ..model.loading import ModelHandle
    from ..runtime.catalog_handles import CorpusHandle, TaskHandle


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Resolved request supplied to one Evaluation Protocol ``evaluate`` entrypoint."""

    task: TaskHandle
    corpus: CorpusHandle
    model: ModelHandle
    parameters: ResolvedPublicParameters
    work_dir: Path


@dataclass(frozen=True, slots=True)
class UnavailableOutput:
    """Explicit unavailability report for one declared formal output."""

    output: str
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Candidate formal outputs returned by one Evaluation Protocol invocation."""

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]
    unavailable_outputs: Sequence[UnavailableOutput]


EvaluationEntrypoint: TypeAlias = Callable[[EvaluationContext], EvaluationResult]
"""Resolved Evaluation Protocol v1 ``evaluate`` callable."""
