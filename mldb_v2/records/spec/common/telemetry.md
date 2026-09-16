# Contract: Execution telemetry

- **id**: `spec:mldb.v2.common.telemetry`
- **status**: draft
- **date**: 2026-09-15
- **parent**: `spec:mldb.v2.common`
- **contract_class**: `interface`

## Purpose

Execution telemetry is a backend-neutral observational projection for understanding training and
evaluation behavior while work is running or after it finishes. It is not canonical experiment
truth and is never a substitute for Training Result, Evaluation Result, Model, or Study Result.

Protocol code MUST NOT import ClearML or another backend SDK to publish telemetry. The execution
harness supplies a reporter through the callable context and the active backend may project accepted
telemetry to its native scalar/plot UI.

## Scalar reporter

```python
class TelemetryReporter(Protocol):
    def report_scalar(
        self, *, group: str, series: str, value: int | float, step: int
    ) -> None:
        ...
```

`group` and `series` are non-empty stable presentation keys. `value` is a finite integer or number;
boolean is invalid. `step` is a non-negative integer; boolean is invalid. These values are
observational labels only and MUST NOT be parsed to reconstruct canonical identity or lifecycle.

Malformed telemetry calls are Protocol contract violations. Failure of a configured backend
telemetry destination after a valid call has been accepted MUST NOT fail an otherwise valid logical
training/evaluation stage; backend delivery is best-effort projection.

## Required observability

A successful Train Protocol invocation MUST make at least one valid scalar telemetry report that is
materially useful for judging optimization progress, training quality, or another protocol-relevant
execution property. MLDB does not prescribe metric names, report cadence, epoch semantics, or a
particular loss/accuracy family. The appropriate series and cadence are chosen when the Protocol is
authored or changed and are part of its semantic review.

A successful Evaluation execution MUST likewise establish at least one useful scalar telemetry
point. The execution harness automatically projects validated numeric `EvaluationCandidate.metrics`
into telemetry; an Evaluation Protocol may additionally report intermediate/sweep telemetry through
its context reporter. Artifact-only evaluations remain valid only if the Protocol explicitly emits
at least one useful scalar observation.

Automated conformance can require at least one valid accepted scalar event, but cannot determine
whether a chosen metric is scientifically useful. That semantic judgment belongs to Protocol review.
