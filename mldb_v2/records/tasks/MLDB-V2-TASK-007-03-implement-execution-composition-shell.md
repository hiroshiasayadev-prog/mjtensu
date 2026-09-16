# MLDB-V2-TASK-007-03: Implement execution composition shell

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-007
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-003, MLDB-V2-TASK-006-01, MLDB-V2-TASK-006-02, MLDB-V2-TASK-006-03]
- **outputs**: start/advance/run/resume/rerun/cancel application-side composition against injected W006 driver seams

## Start gate clarification
This Task may start before T006-04/T006-05 and before W006 closure. It MUST use injected fake/protocol/callable seams for one-pass advancement and cancellation request handling. W006 completion is the concrete integration/closure gate, not the contract-first implementation gate.

## Exclusive ownership
May modify only:
- `mldb_v2/src/api/_execution.py`
- `mldb_v2/tests/test_api_execution.py`
- this Task record

Do not modify any other `src/api/**`, W006 implementation/tests, backend implementation, CLI, Specs/Skeletons, or `records/tasks/index.md`.

## Goal
Implement only Application-side execution composition. Never reimplement readiness, acceptance, child-before-parent persistence, ownership reconciliation, cancellation progression, or terminalization rules owned by W006.

## Work
- `start_study` validates the exact immutable Plan, builds the initial canonical StudyResult topology mechanically from that Plan, and persists it through W001 `CanonicalRepositoryWriter.create_study_result`; it performs zero backend admission.
- Same execution key + same Plan/backend is idempotent; conflicting reuse is lifecycle conflict. Do not allocate retry/backend IDs into canonical state.
- `advance_study` delegates exactly one call to an injected W006 one-pass primitive and returns its exact semantic summary; do not derive progression decisions locally.
- `run_study` composes plan -> fresh UUID4 execution key -> start -> repeated injected advance calls until terminal. `resume_study` loops the same primitive on one existing non-terminal execution.
- `rerun_study` resolves the source StudyResult and its exact referenced immutable Plan, chooses source backend unless explicitly overridden, allocates a fresh execution key, and never recompiles the current Study definition.
- `cancel_study` delegates cancellation-request handling to an injected narrow seam. Do not encode W006 cancellation ownership/progression rules here.
- Foreground looping may use an injected wait/sleep policy, but must contain no alternate readiness/terminalization state machine.
- Keep all lower/application failures bounded for T007-04 public error mapping.

## Focused verification
Use fake injected driver/cancel seams. Cover start durability-before-admission, idempotent/conflicting execution keys, exact initial topology including existing-Model trials, one-pass delegation, run/resume loop behavior, rerun exact-Plan semantics, fresh UUID4 allocation, cancel delegation, and proof that W006 semantics are not duplicated. No full suite.

## Completion condition
Execution composition obeys frozen Application semantics while all progression/cancellation state authority remains behind the injected W006 seam.

No commit/stage/stash/reset/clean/restore/push.
## Completion evidence — 2026-09-13
- Added `mldb_v2/src/api/_execution.py` as a thin execution-composition shell with injected planning, one-pass advancement, cancellation-request, wait, UUID4, and clock seams.
- `start_study` resolves and validates the exact immutable StudyPlan, mechanically mirrors Plan trial/evaluation order into an initial `submitted` StudyResult, allocates `created_at` once, and persists only through `CanonicalRepositoryWriter.create_study_result`; no advancement/admission/backend call occurs.
- Same execution key + same Plan/backend replays the existing canonical StudyResult without reallocating `created_at`; different Plan/backend reuse is a bounded lifecycle conflict, including concurrent-create replay recovery.
- `advance_study` performs exactly one injected one-pass call and returns that semantic response unchanged. No W006 study-driver, readiness, result-acceptance, BackendPort, ClearML, queue, retry, terminalization, or ownership implementation is imported or duplicated.
- `run_study` is plan -> fresh UUID4 key -> start -> injected advance loop; `resume_study` reuses the existing execution identity; both loop only on the injected `terminal` flag and optional wait seam.
- `rerun_study` resolves the source StudyResult and its exact referenced immutable Plan, never calls current-Study planning, inherits the source backend unless explicitly overridden, and always allocates a fresh UUID4 execution key.
- `cancel_study` delegates once to the injected cancellation-request seam and only validates/adapts the frozen three-field response shape; no cancellation progression is implemented locally.
- Focused verification: `mldb_v2/tests/test_api_execution.py` -> **11 passed in 8.67s**; changed Python `py_compile` PASS; untracked-file whitespace checks PASS. Full `mldb_v2/tests` intentionally not run.
- No commit/add/stage/stash/reset/clean/restore/push performed. Concrete production W006/W005 binding remains T007-04 ownership.

## Focused repair evidence — 2026-09-14
- T007-05 adversarial closure found that malformed execution request identities/values were escaping as ordinary `ValueError` and were therefore publicly misclassified as `validation_failed` instead of `invalid_request`.
- Repaired only `src/api/_execution.py`: public execution request ingress now converts malformed Plan/Study/StudyResult identities, caller-supplied execution keys, and caller-supplied backend names into the existing bounded `_ApplicationBoundaryError` with `code=invalid_request`.
- Split start composition so `run_study` / `rerun_study` can use an already-validated internal start path. Internally generated UUID failures and canonical backend values are not relabeled as caller request failures.
- Preserved category boundaries: well-formed missing identity -> `not_found`; execution-key reuse conflict -> `lifecycle_conflict`; corrupt canonical Plan/domain validation -> `validation_failed`.
- Added adversarial coverage for malformed inputs across `start_study`, `advance_study`, `run_study`, `resume_study`, `rerun_study`, and `cancel_study`, including explicit rerun backend override and internal UUID generation failure.
- Focused `test_api_execution.py`: **14 passed**. Public T007-04 Application integration: **16 passed**.
- `py_compile` and changed-file whitespace/BOM checks: PASS. No W006 progression/cancellation logic was introduced.
- No commit/add/stage/stash/reset/clean/restore/checkout/worktree/push performed. T007-05 remains untouched/planned for restarted closure verification.
