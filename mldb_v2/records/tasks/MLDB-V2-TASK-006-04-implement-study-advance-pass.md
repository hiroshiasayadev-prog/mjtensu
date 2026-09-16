# MLDB-V2-TASK-006-04: Implement one idempotent Study advancement pass

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-006
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-006-01, MLDB-V2-TASK-006-02, MLDB-V2-TASK-006-03, MLDB-V2-WORK-005]
- **outputs**: composed ResultAcceptor runtime and one-pass `advance_study` reconciliation, focused integration tests

## Start gate clarification — 2026-09-13
T006-01/02/03 and W004 are completed. T005-01/02/03/04 are also completed and the frozen generic `BackendPort` surface is stable. Therefore T006-04 implementation may start before T005-05/W005 formal closure by testing against a fake BackendPort. `depends_on: W005` remains the W006 integration/completion gate; T006-04 must not import ClearML-specific modules. W007 owns all `mldb_v2/src/api/`, so this Task must implement the internal one-pass primitive outside that package.

## Goal
Implement one resumable/idempotent `advance_study` pass that reconciles backend ownership with immutable Plan semantics and canonical history in the frozen order.

## Work
- Compose the public ResultAcceptor boundary over completed T006-01/T006-02 persistence-free acceptance implementations.
- Validate current StudyResult/Plan/source, observe/recover pending StageKeys, collect terminal candidates, then delegate to result acceptance.
- Persist immutable children before parent disposition: TrainingResult then Model, or EvaluationResult, then re-resolve and replace the non-terminal StudyResult under short mutation coordination.
- Apply T006-03 upstream skips; when cancelling, distinguish never-admitted pending work from admitted work using backend ownership, skip only never-admitted as `study_cancelled`, and request backend cancellation for admitted work.
- Derive newly-ready StageInputs after canonical reconciliation and idempotently admit them through the selected BackendPort; then terminalize only when every planned slot is non-pending.
- Do not hold the repository mutation lock during backend calls, artifact IO, polling, or waits; after lock acquisition always re-resolve canonical state before mutation.
- Backend unavailable is transient; do not global-fail solely for temporary backend reachability. Do not choose queue/GPU/retry policy or create alternate state machines.
- Do not implement public `mldb_v2/src/api/` application methods; W007 will wrap this internal primitive.

## Done condition
Repeated one-pass calls can recover ownership, accept/persist children, skip/admit newly-ready work, and close StudyResult without duplicate logical admission or parent-before-child history.

## Verification
Focused one-pass integration tests only: no-op active pass, terminal collection, child-before-parent ordering, readiness/admission, cancellation ownership split, stale caller re-resolution, admission ambiguity recovery, and backend unavailable. No broad regression; no commit/stage/push.

## Completion evidence — 2026-09-13
- Added runtime `ResultAcceptor` composition in `mldb_v2/src/verification/result_acceptance.py`, delegating to completed T006-01 `_accept_training_candidate` and T006-02 `_accept_evaluation_result` without copying acceptance logic.
- Added `AcceptedResultRecordValidator`, a bounded kind-dispatched `CanonicalRecordValidator` for accepted TrainingResult, Model, and EvaluationResult persistence/replay. It reuses owning TrainingResult/Model validators, validates canonical Model lineage, and reuses the Evaluation acceptance metric/artifact/protocol contract without persistence-time object-byte fetches.
- Added internal `mldb_v2.src.study.study_driver.advance_study`. It consumes only generic `BackendPort`, follows the frozen observe -> collect -> accept -> child persist -> parent update -> skip -> readiness -> admit -> closure order, and returns deterministic `StudyAdvanceResult` fields for W007 wrapping.
- Child-before-parent recovery is implemented: TrainingResult -> optional Model -> StudyResult slot, and EvaluationResult -> StudyResult slot. Exact existing children are validated and replayed; interrupted-after-child cases finish the missing parent update without rollback/delete.
- Added the minimal backward-compatible `_materialize_stage_input` helper extraction in `execution_readiness.py` for already-admitted/cancelling pending work; `_build_stage_input` remains ready-only for new admission.
- Cancelling separates never-admitted pending work (`skipped: study_cancelled`) from admitted active/terminal ownership; admitted work receives backend cancellation and terminal candidates are still formally collected/accepted. No new admission occurs while cancelling.
- StudyResult mutation uses `StudyResultMutationCoordinator`; every mutation lock re-resolves canonical state before reconciliation. Backend observe/collect/admit/cancel and acceptance object-byte IO occur outside the StudyResult mutation lock.
- Focused verification: `test_result_acceptance.py` **4 passed**, `test_study_driver.py` **28 passed**. T006-01/02/03/04 focused join (`test_training_result_acceptance.py`, `test_evaluation_result_acceptance.py`, `test_execution_readiness.py`, plus both T006-04 files) **111 passed in 4.53s**.
- Changed Python `py_compile` PASS; exact-symbol import smoke PASS; dependency scan PASS with no ClearML, Skeleton, or `src/api` dependency. Full `mldb_v2/tests` intentionally not run; broad/concurrency adversarial remains T006-05 scope.
- No commit/add/stash/reset/clean/restore/push performed. T005-05/W005 remains the W006 integration/closure gate and is not a blocker to this completed T006-04 implementation task.
