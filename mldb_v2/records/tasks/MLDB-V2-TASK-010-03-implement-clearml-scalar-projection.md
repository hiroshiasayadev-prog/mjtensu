# MLDB-V2-TASK-010-03: Implement ClearML scalar telemetry projection

- **status**: completed
- **date**: 2026-09-15
- **work_item**: MLDB-V2-WORK-010
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-010-02]
- **outputs**: backend-neutral scalar delivery seam, live ClearML scalar projection, delivery-failure isolation, focused conformance tests

## Boundary

Project accepted backend-neutral scalar telemetry to the current logical ClearML execution Task without changing canonical Training, Evaluation, Study, artifact, StageInput, or TerminalCandidate semantics. Protocol-specific classifier/detector telemetry remains T010-04 ownership; actual GPU/Charts verification remains T010-05.

## Backend-neutral delivery seam

`_RecordingTelemetryReporter` accepts an optional private sink callable. Each report is processed strictly as validation, accepted-event append, then best-effort sink delivery. Accepted events remain ordered, attempt-local, non-canonical, and are never deduplicated. A malformed call raises before acceptance and before sink invocation. Sink absence preserves local/fake behavior.

Training and Evaluation runtime entrypoints accept the same optional private sink and pass it only to the recording reporter. `CommonExecutionHarness` threads that sink to the selected runtime while preserving the public `call(stage_input)` shape and all StageInput/candidate schemas.

## ClearML projection

The ClearML remote harness binds a lazy `_ClearMLScalarSink` to the already-current `Task.init(...)` Task. The sink resolves `task.get_logger()` only on the first accepted event and reuses that logger. It does not create Tasks, children, Projects, or per-event flushes.
Mechanical scalar mapping uses the installed ClearML 2.1.12 Logger API exactly: MLDB `group -> title`, `series -> series`, `value -> value`, and `step -> iteration`. Generic presentation keys are not renamed or interpreted.

## Accepted-before-delivered and failure isolation

Once an event is appended, any sink/logger/reporting exception is suppressed as operational telemetry loss. Accepted-count enforcement therefore depends only on accepted events and never on ClearML delivery success. Training weights and Evaluation metrics/artifacts remain unchanged when a sink fails after acceptance. Existing final `task.flush(wait_for_uploads=True)` is preserved.

Telemetry is not added to TerminalCandidate, TrainingCandidateResult, EvaluationCandidateResult, canonical Training/Evaluation Results, Model, StudyResult, or StudyPlan. Common telemetry/runtime code has no ClearML/backend SDK dependency.

## Verification evidence — 2026-09-15

- Final focused suite (`test_common_telemetry.py`, Training/Evaluation runtime, execution harness, ClearML admission/observation/cancellation/backend conformance): **208 passed, 1 skipped in 46.97s**.
- Common execution harness focused rerun including direct live-sink threading: **45 passed in 45.12s**.
- Fake ClearML Task/Logger verifies lazy logger resolution and exact generic mappings for `optimization/cross_entropy_loss/1.25/3` and `validation/mean_iou/0.7/5`.
- Training sink-success versus sink-failure candidate result and published weights: invariant.
- Evaluation automatic metric delivery and sink-failure candidate metrics/artifacts/publication: invariant.
- `py_compile` for changed Python and focused tests: PASS.
- Common telemetry/ClearML dependency-direction scan: PASS.
- `git diff --check -- mldb_v2`: PASS.
- Owned-file trailing-whitespace scan: PASS.
- Actual GPU/Charts execution intentionally not run; deferred to T010-05.
- W010 remains `planned`; T010-03 alone is completed.
