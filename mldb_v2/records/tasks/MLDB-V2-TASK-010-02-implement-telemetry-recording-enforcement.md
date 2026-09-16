# MLDB-V2-TASK-010-02: Implement telemetry recording and enforcement

- **status**: completed
- **date**: 2026-09-15
- **work_item**: MLDB-V2-WORK-010
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-010-01]
- **outputs**: accepted scalar recording, Training/Evaluation zero-event enforcement, validated Evaluation metric projection

## Boundary
Implement backend-neutral accepted-event recording and successful-stage observability enforcement. Do not implement ClearML delivery, protocol-specific series/cadence, canonical telemetry persistence, or candidate/result schema changes.

## Accepted-event recording
`_RecordingTelemetryReporter` validates every `report_scalar()` call through the T010-01 scalar validator before appending a private immutable `_AcceptedScalarEvent`. Accepted events preserve exact call order and values without deduplication. Malformed calls raise `ValueError` and are not counted. Recording is execution-attempt-local, non-canonical state and has no backend dependency.

## Training enforcement
The Training runtime creates one recording reporter and injects it into `TrainContext`. After `train(context)` returns, it requires `accepted_count >= 1` before trained-module canonicalization, weights serialization, or object-store publication. A malformed report remains a Protocol invocation failure; a silent Protocol is a distinct zero-event contract violation. Both are bounded by the existing common harness failure boundary.

## Evaluation projection and enforcement
The Evaluation runtime validates `EvaluationCandidate.metrics` with the existing strict `_validate_candidate_metrics()` first. Only validated metrics are then recorded automatically. The deterministic terminal-summary mapping is `group = exact Evaluation stage name`, `series = exact metric declaration key`, `value = validated metric value`, `step = 0`. This mapping adds no scientific metric semantics.

After explicit Protocol telemetry plus automatic metric projection, Evaluation requires `accepted_count >= 1` before candidate artifact validation/publication. Therefore artifact-only output without an explicit valid scalar fails before object-store publication, while explicit telemetry, automatic metrics, or both may satisfy the success contract.

## Shape and dependency preservation
Completed Training/Evaluation candidate/result shapes are unchanged; telemetry is not added to `TerminalCandidate`, canonical Result, Study Result, YAML/JSON records, or object storage. The common telemetry/runtime implementation adds no ClearML, boto3, queue, worker, or backend-SDK dependency. ClearML delivery and classifier/detector-specific telemetry remain T010-03/T010-04 ownership.

## Verification evidence — 2026-09-15
- Common telemetry + Training/Evaluation runtime focused tests: `103 passed, 1 skipped in 4.22s`.
- Final focused runtime/harness/executable regression: `258 passed, 2 skipped in 49.73s`.
- Actual harness checks cover silent/malformed Training as bounded failed candidates with no weights publication; Evaluation zero-event artifact publication is blocked at runtime.
- `py_compile` for all T010-02 changed Python: PASS.
- Dependency/import scan and genericity scan: PASS.
- `git diff --check -- mldb_v2`: PASS; owned-file trailing-whitespace scan: PASS.
- Full `mldb_v2/tests` intentionally not run per focused-test policy.
- W009 and T010-01 remain completed and unmodified by this Task. W010 remains `planned`; only T010-02 is completed.
