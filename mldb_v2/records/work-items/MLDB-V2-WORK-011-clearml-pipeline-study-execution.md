# MLDB-V2-WORK-011: ClearML Pipeline Study execution

- **status**: in_progress
- **date**: 2026-09-17
- **depends_on**: [MLDB-V2-WORK-010]
- **source_refs**: `spec:mldb.v2.backend.backend_port`, `spec:mldb.v2.backend.clearml_mapping`, `spec:mldb.v2.api.study_driver`, `spec:mldb.v2.study.execution_readiness`

## Goal

Repair the post-W010 ClearML execution mapping so MLDB remains the authority for experiment semantics and canonical history while ClearML owns the operational mechanics it is designed to provide: one Pipeline Run per Study Result, child Task scheduling, dependency execution, queue/worker placement, retry/liveness, cancellation, logs, and UI grouping.

This is an approved post-closure Specification amendment. W005/W006 completion evidence remains historical evidence for the previous flat-Task mapping and must not be rewritten as if Pipeline support already existed.

## Boundary

MLDB continues to own Study/Plan identity, trial/evaluation semantics, source pinning, runtime Model lineage, formal result acceptance, and canonical Training/Evaluation/Study Results. ClearML Pipeline state is an operational projection and never becomes canonical truth.

The common execution harness remains the only training/evaluation domain execution entrypoint. No model-family-specific runners are introduced.

## Task candidates

| task | responsibility | dependency |
|---|---|---|
| T011-01 | Amend formal Specs/manuals for the StudyResult -> Pipeline Run mapping and responsibility split. | W010 |
| T011-02 | Add the backend-neutral Study-execution capability and ClearML PipelineController projection while preserving CommonExecutionHarness stage execution. | T011-01 |
| T011-03 | Rework Study progression so ClearML schedules physical steps while MLDB performs semantic gating, canonical acceptance, and downstream release after accepted training/model lineage. | T011-02 |
| T011-04 | Project Pipeline stage grouping plus selected Study-summary metrics/artifacts; integrate backend retry/cancel/recovery without duplicate MLDB scheduling. | T011-03 |
| T011-05 | Verify actual ClearML Pipeline UI and RTX 3090 execution end to end, including interruption/resume, retry/cancel, source pinning, and canonical authority. | T011-04 |

## Completion condition

- One MLDB Study Result maps to exactly one recoverable ClearML Pipeline Run/controller identity.
- Planned training/evaluation work appears as Pipeline step Tasks grouped for human navigation rather than an unstructured flat Task list.
- ClearML owns physical step scheduling, queue/worker placement, operational retry/liveness, and cancellation; MLDB does not duplicate those mechanisms.
- No dependent Evaluation starts before the required Training Result and Model lineage have passed MLDB canonical acceptance.
- ClearML cache/reuse cannot silently replace a fresh MLDB execution unless a future explicit MLDB contract authorizes that behavior.
- Pipeline/child status, telemetry, logs, and UI remain projections; canonical MLDB records remain authoritative.
- Existing public experiment authoring semantics, source pinning, Protocol execution, and result contracts remain model-family neutral.
