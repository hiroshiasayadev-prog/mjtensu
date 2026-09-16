# MLDB-V2-TASK-006-05: Verify Study driver cancellation, recovery, and concurrency closure

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-006
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-006-04]
- **outputs**: W006 closure verification across acceptance/readiness/driver concurrency and recovery

## Goal
Close W006 by proving the one-pass driver and formal acceptance rules remain correct across cancellation, interrupted persistence, repeated/concurrent callers, backend unavailability, failed/cancelled training, and existing-Model Studies.

## Work
- Verify child-before-parent recovery when interruption occurs after TrainingResult, after Model, after EvaluationResult, and before/after StudyResult replacement.
- Verify concurrent/repeated `advance_study` callers re-resolve under StudyResult mutation coordination and cannot lose dispositions or duplicate immutable child records/logical backend admission.
- Cover cancelling with never-admitted vs admitted stages, failed/cancelled training skip propagation, evaluation sibling independence, existing Models, terminal closure rules, and global progression failure handling.
- Verify backend `completed` never bypasses T006-01/T006-02 acceptance and no skipped stage receives a synthetic child Result.
- Keep read-only query/API presentation, CLI, scheduling policy, and model-family execution outside this Task.

## Done condition
Every planned stage deterministically closes completed/failed/cancelled/skipped under the frozen rules, recovery is resumable, and terminal StudyResult history is immutable and internally complete.

## Verification
Run focused W006 closure suites first, then relevant W003/W004/W005 integration and broad `mldb_v2/tests`; include concurrency/interruption tests. Do not modify Work Item status or `tasks/index.md`; no commit/stage/push.
## Completion evidence — 2026-09-14
- Added `mldb_v2/tests/test_study_driver_closure.py` covering interrupted child-before-parent recovery, terminal immutability, all backend calls outside StudyResult mutation coordination, mixed cancellation, transient cancel recovery, concurrent admission/terminal reconciliation, sibling independence, and existing-Model completion/failure/replay.
- Adversarial verification found one blocking defect: when a deterministic canonical TrainingResult already existed, a newly accepted different valid terminal document with the same ID could be ignored in favor of the existing child, bypassing the canonical exact-replay/lifecycle-conflict rule. Trigger: existing `failed` TrainingResult plus newly accepted `cancelled` TrainingResult at the same deterministic ID.
- Repaired only `mldb_v2/src/study/study_driver.py`: prepared TrainingResult/EvaluationResult documents now pass through `CanonicalRepositoryWriter.create_immutable()` even when that deterministic child already resolves; same-ID different-content conflicts are converted to bounded `_UnrecoverableProgressionError`, while exact replay remains idempotent.
- T006-05 focused closure: **15 passed**.
- T006-01..05 focused join: **126 passed**.
- Relevant W003/W004/W005 integration: **340 passed, 1 skipped**.
- Full `python -m pytest mldb_v2/tests -q`: **1254 passed, 3 skipped**.
- `py_compile` PASS; exact-symbol import smoke PASS; forbidden dependency/genericity scan PASS; whitespace scan PASS. No catch-all progression failure conversion was introduced.
- Backend `observe`, `collect`, `admit`, and `cancel_study` remain outside StudyResult mutation coordination; concurrent/repeated callers re-resolve canonical state under the short mutation lock and backend deterministic ownership remains the logical-admission authority.
- No commit/add/stash/reset/clean/restore/push performed. W006 work-item status and `tasks/index.md` were intentionally not modified; W006 closure belongs to the coordinator.
