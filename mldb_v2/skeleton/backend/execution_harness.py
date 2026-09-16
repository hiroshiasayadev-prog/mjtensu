"""Backend-neutral process entrypoint for one immutable planned stage."""

from typing import Protocol

from mldb_v2.skeleton.backend.candidate_outcome import TerminalCandidate
from mldb_v2.skeleton.backend.stage_input import StageInput


class ExecutionHarness(Protocol):
    """Execute one admitted StageInput through the common MLDB harness."""

    def __call__(self, stage_input: StageInput) -> TerminalCandidate:
        ...
