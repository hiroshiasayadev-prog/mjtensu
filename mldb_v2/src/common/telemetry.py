"""Backend-neutral execution telemetry contract and scalar validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
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


def _validate_scalar_call(
    *,
    group: object,
    series: object,
    value: object,
    step: object,
) -> None:
    if type(group) is not str or not group:
        raise ValueError("telemetry group must be a non-empty exact string")
    if type(series) is not str or not series:
        raise ValueError("telemetry series must be a non-empty exact string")
    if type(value) not in {int, float}:
        raise ValueError("telemetry value must be an exact int or float")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("telemetry value must be finite")
    if type(step) is not int:
        raise ValueError("telemetry step must be an exact integer")
    if step < 0:
        raise ValueError("telemetry step must be non-negative")


@dataclass(frozen=True, slots=True)
class _AcceptedScalarEvent:
    group: str
    series: str
    value: int | float
    step: int


class _RecordingTelemetryReporter:
    """Validate, retain, then best-effort deliver scalar events for one attempt."""

    def __init__(
        self,
        *,
        sink: Callable[[_AcceptedScalarEvent], None] | None = None,
    ) -> None:
        self._events: list[_AcceptedScalarEvent] = []
        self._sink = sink

    @property
    def accepted_events(self) -> tuple[_AcceptedScalarEvent, ...]:
        return tuple(self._events)

    @property
    def accepted_count(self) -> int:
        return len(self._events)

    def report_scalar(
        self,
        *,
        group: str,
        series: str,
        value: int | float,
        step: int,
    ) -> None:
        _validate_scalar_call(
            group=group,
            series=series,
            value=value,
            step=step,
        )
        event = _AcceptedScalarEvent(
            group=group,
            series=series,
            value=value,
            step=step,
        )
        self._events.append(event)
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception:
                pass
