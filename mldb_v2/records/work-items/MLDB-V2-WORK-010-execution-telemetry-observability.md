# MLDB-V2-WORK-010: Execution telemetry and observability

- **status**: completed
- **date**: 2026-09-15
- **depends_on**: [MLDB-V2-WORK-009]
- **source_refs**: `spec:mldb.v2.common.telemetry`, `spec:mldb.v2.training.train_interface`, `spec:mldb.v2.evaluation.evaluate_interface`, `spec:mldb.v2.backend.execution_harness`, `spec:mldb.v2.backend.clearml_mapping`

## Trigger

W009 actual ClearML GPU execution completed successfully, but the completed Training/Evaluation Tasks
showed `No chart data`. The execution path preserves canonical results correctly but does not publish
scalar telemetry, so optimization behavior and evaluation summaries are not observable in the backend
UI.

## Goal

Make useful execution telemetry a formal, backend-neutral MLDB callable responsibility while keeping
canonical truth unchanged. ClearML remains only an operational projection; Protocol code never imports
or depends on ClearML directly.

The generic contract deliberately does **not** prescribe `loss`, `accuracy`, epoch cadence, or any
other fixed metric family. Which observations are scientifically useful is decided when each Train or
Evaluation Protocol is authored or changed.

## Boundary

Own the telemetry contract, callable-context seam, common execution-harness enforcement, and backend
projection. Do not make telemetry canonical, do not add queue/worker/heartbeat infrastructure, and do
not let Protocol companions import backend SDKs.

A valid telemetry call may be rejected for malformed group/series/value/step. Once a valid event is
accepted, downstream reporting failure is best-effort operational loss and MUST NOT fail an otherwise
valid logical stage.

## Task candidates

| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-010-01 | Freeze `TelemetryReporter` and updated Train/Evaluation Context Skeleton mirrors; add shape/validation tests. | W009 |
| MLDB-V2-TASK-010-02 | Implement harness-side recording/enforcement and automatic numeric Evaluation metric projection. | T010-01 |
| MLDB-V2-TASK-010-03 | Implement ClearML scalar projection with delivery failure isolated from canonical stage success. | T010-02 |
| MLDB-V2-TASK-010-04 | Update current classifier/detector Train Protocols to emit protocol-appropriate useful progress telemetry; select exact series/cadence explicitly per Protocol rather than from a generic hard-coded list. | T010-02 |
| MLDB-V2-TASK-010-05 | Run focused/fake conformance plus bounded actual ClearML GPU verification and confirm chart data appears without changing canonical result authority. | T010-03,T010-04 |

## Completion condition

- Successful training cannot complete with zero accepted scalar telemetry events.
- Successful evaluation cannot complete with zero accepted scalar telemetry events; validated numeric candidate metrics are automatically projected.
- Generic MLDB does not prescribe a universal metric name, cadence, or training-loop unit.
- Current classifier/detector Protocols have explicitly reviewed telemetry series appropriate to their own training procedures.
- Protocol companions have no ClearML/backend imports.
- ClearML displays scalar chart data for bounded actual training and evaluation Tasks.
- Telemetry delivery failure does not change canonical Training/Evaluation/Study success or artifact identity.
- Existing canonical result formats and backend authority boundaries remain unchanged.

## Closure evidence

T010-05 completed actual ClearML GPU verification on pinned source `0cc3828702bbb7630675ba3de3e62a6a0b79bfb7`. Detector and classifier Training/Evaluation Tasks completed on `bugrat-gpu0` / `NVIDIA GeForce RTX 3090`; required Training and Evaluation scalar Charts were present, canonical Result authority and artifact identity remained unchanged, and the final full suite passed `1384 passed, 3 skipped`. Execution telemetry and observability is closed.
