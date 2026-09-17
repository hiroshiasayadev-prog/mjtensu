# MLDB-V2-TASK-011-05: Verify actual ClearML Pipeline closure

- **status**: in_progress
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

## Actual verification evidence

- Completed real GPU Pipeline StudyResult: `tile-classifier/run-9b98462eef4a489fbb9e52a7ad692cb4`.
- One controller Task `1eee926262e346d4b12c419ef6c3ac50` owned four child nodes: two training Tasks and two angle-robustness Evaluation Tasks.
- Both training Tasks completed before their dependent Evaluation Tasks were released; canonical TrainingResult/Model acceptance remained the semantic gate.
- Local execution interruption plus `resume` recovered the same controller and child Task identities; replay-safe parent binding prevented duplicate logical work.
- Final completed controller summary exposed 8/8 selected Study metrics as controller single values while detailed `angle-robustness` / `angle-sweep` telemetry remained on child Evaluation Tasks.
- ClearML Fileserver returned 404 for the optional Study-summary artifact; the projection was repaired to controller configuration + single-value metrics so a reporting-storage failure cannot block canonical Study progress.
- ClearML API-server 2.17+ native Pipeline discovery requires hidden `mldb/<namespace>/.pipelines/<study-local-id>` subprojects. The W011 controller was migrated there and its Project now carries `hidden,pipeline`; the controller remains type `controller` with system tag `pipeline`.
- A native `PipelineController` probe produced the same Project/type/system-tag shape; the temporary probe Task/Project was deleted after comparison.
- Actual cancellation StudyResult: `tile-classifier/run-6ef38740e3bf447ea650c1940513a289` -> canonical `cancelled`; controller and two admitted training Tasks -> ClearML `stopped`; dependent Evaluations -> canonical skipped without duplicate Tasks.
- Focused/aggregate W011 backend regression after the repairs: `95 passed` across Pipeline, Study driver, admission, observation, and cancellation tests.

Human-visible Pipeline-page confirmation is still required before T011-05/W011 closure. The earlier `No pipelines to show` observation occurred before native Pipeline-subproject placement was repaired and the existing controller was migrated.
