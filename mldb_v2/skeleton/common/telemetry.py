"""MLDB v2 backend-neutral execution telemetry boundary."""

from __future__ import annotations

from typing import Protocol


class TelemetryReporter(Protocol):
    def report_scalar(
        self,
        *,
        group: str,
        series: str,
        value: int | float,
        step: int,
    ) -> None:
        ...
