# MLDB-V2-TASK-010-01: Freeze telemetry callable-context contract

- **status**: completed
- **date**: 2026-09-15
- **work_item**: MLDB-V2-WORK-010
- **task_type**: contract
- **depends_on**: [MLDB-V2-WORK-009]
- **outputs**: frozen/public `TelemetryReporter`, required Train/Evaluation Context telemetry field, scalar validation seam, focused compatibility tests

## Boundary
Freeze `spec:mldb.v2.common.telemetry` into Skeleton and src mirrors without implementing backend delivery, accepted-event recording, successful-stage zero-telemetry enforcement, Evaluation metric projection, or protocol-specific metric/cadence policy.

## Owned changes
- `mldb_v2/skeleton/common/telemetry.py`
- `mldb_v2/skeleton/training/train_interface.py`
- `mldb_v2/skeleton/evaluation/evaluate_interface.py`
- `mldb_v2/src/common/telemetry.py`
- `mldb_v2/src/training/train_interface.py`
- `mldb_v2/src/evaluation/evaluate_interface.py`
- `mldb_v2/src/training/runtime.py`
- `mldb_v2/src/evaluation/runtime.py`
- `mldb_v2/tests/test_common_telemetry.py`
- exact-shape compatibility updates in executable/training/evaluation runtime tests

## Contract frozen
`TelemetryReporter.report_scalar` is keyword-only for `group`, `series`, `value`, and `step`, returns `None`, and has no backend dependency. `TrainContext` and `EvaluationContext` retain frozen dataclass semantics and now require a non-optional `TelemetryReporter` field in the spec-defined position.

## Validation semantics
- `group` / `series`: exact non-empty `str`; no trimming, regex, taxonomy, or hierarchy semantics added.
- `value`: exact `int | float`, boolean rejected, finite only, no coercion.
- `step`: exact non-negative `int`, boolean rejected, no coercion.
- Malformed calls raise bounded `ValueError` from the private validation seam.

## Compatibility wiring
Training and evaluation runtimes inject a private `_ValidatingDiscardingTelemetryReporter`. It validates valid protocol calls and discards them. It does not record accepted events, persist telemetry, import ClearML/backend code, enforce minimum event counts, or project Evaluation metrics. This preserves existing runtime execution until T010-02 replaces/extends the seam.

## Verification evidence — 2026-09-15
- Focused contract + executable loading + training/evaluation runtime: `190 passed, 1 skipped in 6.27s`.
- Backend execution harness + executable asset regression: `56 passed, 1 skipped in 49.43s`.
- `py_compile` for all changed Python: PASS.
- Telemetry dependency/import scan for ClearML, boto3, backend, queue, worker imports: PASS.
- Full `mldb_v2/tests` intentionally not run; T010-01 requires focused/relevant regression only.
- W009 was not reopened or modified. W010 remains `planned`; only T010-01 is completed.
