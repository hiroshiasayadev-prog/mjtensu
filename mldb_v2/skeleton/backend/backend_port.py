"""Generic MLDB v2 execution-backend port."""

from typing import Protocol

from mldb_v2.skeleton.backend.candidate_outcome import (
    BackendObservation,
    StageKey,
    TerminalCandidate,
)
from mldb_v2.skeleton.backend.stage_input import StageInput
from mldb_v2.skeleton.common.ids import StudyResultId


class BackendPort(Protocol):
    """Required backend-neutral execution capabilities only."""

    def admit(self, *, stage_input: StageInput) -> BackendObservation:
        """Idempotently admit or recover one exact ready logical stage."""
        ...

    def observe(self, *, stage_key: StageKey) -> BackendObservation | None:
        """Recover the current active/terminal observation for logical work."""
        ...

    def collect(self, *, stage_key: StageKey) -> TerminalCandidate | None:
        """Collect the terminal candidate when the logical work is terminal."""
        ...

    def cancel_study(self, *, study_result: StudyResultId) -> None:
        """Request cancellation of admitted work for one Study execution."""
        ...
