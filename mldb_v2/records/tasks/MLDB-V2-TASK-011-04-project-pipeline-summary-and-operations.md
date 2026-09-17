# MLDB-V2-TASK-011-04: Project Pipeline summary and operational controls

- **status**: completed
- **date**: 2026-09-17
- **work_item**: MLDB-V2-WORK-011
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-011-03]
- **outputs**: Pipeline stage grouping, selected summary projection, retry/cancel/log integration

## Goal

Make ClearML useful as the operational UI for one Study without moving experiment truth out of canonical MLDB.

## Work

- Group child Tasks by stable Pipeline stages suitable for collapsed Study-level navigation.
- Preserve child Task names/metadata sufficient to identify trial, stage kind, evaluation stage, architecture/model, Plan, Study Result, and source commit without parsing display names for identity.
- Mirror only explicitly selected Study-summary metrics/artifacts/models to the Pipeline/controller.
- Keep detailed Protocol telemetry on the child Task that produced it.
- Delegate physical retry/liveness and active-child cancellation to ClearML; map retries back to the same MLDB logical stage attempt history.
- Keep `mldb logs/status/watch/cancel` generic and backed by capability-checked operational observations.

## Verification

Focused tests must cover grouping metadata, bounded summary projection, retry attempt ordering, Pipeline cancellation, log lookup, and the rule that telemetry/UI failures cannot rewrite a valid canonical outcome.

## Completion evidence

- Native Pipeline nodes retain stable `stage` grouping and exact child/parent binding metadata.
- Canonical Study state is projected as one bounded `mldb-study-summary` artifact/configuration on the controller; projection failure is explicitly best-effort and cannot change accepted canonical outcomes.
- Controller status mirrors canonical Study lifecycle for UI navigation without becoming formal authority.
- Pipeline cancellation targets the controller in addition to exact active child Tasks; log lookup remains the existing capability-checked child-Task path.
- Ordered retry attempt histories remain accepted by the ClearML observation boundary; no MLDB-side retry scheduler was introduced. Actual ClearML server retry behavior remains part of T011-05 closure verification.

Verification:

- Pipeline + Study progression focused: **39 passed**
- Pipeline/cancellation/observation/Study operational set: **79 passed**
- W011 + Application/Admission integration join: **104 passed**
- changed runtime `py_compile`: PASS
- `git diff --check -- mldb_v2`: PASS
