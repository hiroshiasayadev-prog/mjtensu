# MLDB-V2-TASK-011-05: Verify actual ClearML Pipeline closure

- **status**: planned
- **date**: 2026-09-17
- **work_item**: MLDB-V2-WORK-011
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-011-04]
- **outputs**: actual ClearML Pipeline/UI/GPU evidence, interruption/recovery/cancel verification, W011 closure

## Goal

Verify the amended execution mapping on the real self-hosted ClearML deployment without changing scientific Protocol semantics.

## Work

- Run a bounded multi-trial Study through one ClearML Pipeline Run on queue `default` / available GPU worker.
- Confirm child training/evaluation Tasks execute through `CommonExecutionHarness` and are grouped/navigable in Pipeline UI.
- Confirm selected summary metrics/artifacts appear at Pipeline level while detailed telemetry remains on child Tasks.
- Verify dependent Evaluation cannot run before accepted TrainingResult/Model lineage.
- Exercise local interruption plus `resume` recovery of the same Pipeline identity.
- Exercise one bounded retry/failure path and Study cancellation without duplicate logical work.
- Confirm source pinning, canonical Result/Model authority, and S3 artifact integrity remain unchanged.
- Run focused W011 tests plus the required aggregate regression before closure.

## Closure rule

Do not close W011 from mocked tests alone. Actual Pipeline UI and real backend execution evidence are required.
