"""MLDB v2 terminal backend-attempt summary value."""

from typing import Literal, TypedDict

from mldb_v2.src.common.diagnostic import Diagnostic


class AttemptSummary(TypedDict):
    backend: str
    execution_id: str
    status: Literal["completed", "failed", "cancelled"]
    started_at: str | None
    ended_at: str | None
    diagnostic: Diagnostic | None
