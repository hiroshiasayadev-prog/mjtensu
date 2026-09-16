# MLDB-V2-TASK-005-02: Implement ClearML deterministic admission

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-005
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-005-01]
- **outputs**: ClearML admission mapping/ownership recovery/common-harness launch helper and focused tests

## Start gate clarification — 2026-09-13
T005-01 is completed. W004 T004-03/T004-04 are completed and the actual common harness implementation now exists at `mldb_v2.src.backend.execution_harness.CommonExecutionHarness`. T005-02 may therefore start before formal W004 closure, consuming that W004-owned module read-only. W004 closure remains a W005 integration/completion gate, not an implementation-start gate.

## Goal
Implement idempotent ClearML admission for one exact ready StageInput using deterministic logical ownership metadata and the common W004 execution harness.

## Work
- Map Namespace to ClearML Project `mldb/<namespace>`; keep Study/Plan/StudyResult/trial/stage/source identity in searchable metadata/configuration, never in a Project-per-Study convention.
- Derive deterministic ownership from exact StudyResult + trial + kind + evaluation coordinate; recover an existing matching Task before creating new logical work, including ambiguous create-response recovery.
- Preserve required ClearML metadata, canonical refs, public parameters, source commit, and the exact StageInput transport without parsing human Task names.
- Remote execution must enter the generic W004 `CommonExecutionHarness` boundary at the Plan-pinned source snapshot; do not invent a model-family runner.
- Keep endpoint/credentials/queue operational and ClearML Task IDs opaque provenance only.
- Keep this Task admission-only so T005-03 observation/recovery and T005-04 cancellation/logs can remain file-disjoint and later run in parallel.

## Exclusive implementation ownership
- `mldb_v2/src/backend/_clearml_admission.py`
- `mldb_v2/tests/test_clearml_admission.py`
- this Task record status/evidence only

## Non-goals
Do not implement observation/collection, cancellation/logs, readiness, result acceptance, Study mutation, retry/heartbeat/GPU policy, or canonical writes. Do not modify W004-owned harness/value files or T005-01 generic registry/config files.

## Done condition
Repeated admission of one exact StageInput resolves one logical ClearML ownership and remote execution is configured only for the generic W004 harness with no duplicate logical work or canonical-config leakage.

## Verification
Focused mocked ClearML admission tests only: project/metadata mapping, ownership replay/ambiguous create recovery, no Task-name parsing, exact StageInput transport, generic harness launch, pinned source identity, config secrecy, and no canonical mutation. No broad regression; no commit/stage/push.

## Completion Evidence — 2026-09-13
- Added `mldb_v2/src/backend/_clearml_admission.py` with admission-only `ClearMLAdmissionService.admit()` plus an SDK-optional `ClearMLAdmissionClient` seam. No ClearML SDK import is required for module import or focused tests.
- Namespace maps exactly to ClearML Project `mldb/<namespace>`. Study identity is recovered from the exact single Study pin in `StageInput.pins`; Study/Plan/StudyResult/trial/stage/ref/source identity is projected as searchable metadata, not encoded as Project-per-Study.
- Deterministic ownership key is `mldb-v2-stage:` plus SHA-256 of canonical JSON containing exactly StudyResult ID, trial ID, stage kind, and evaluation coordinate (`null` for training). Human Task name is presentation-only and is never parsed for ownership/recovery.
- Admission searches ownership before create, validates exact searchable metadata plus exact canonical StageInput transport, recovers one exact existing Task, rejects duplicate/mismatched ownership, and re-searches up to a bounded count after missing/ambiguous create responses. A server-side create followed by lost response is recovered without issuing a second create.
- Exact StageInput transport is canonical JSON (`mldb.stage_input`) and round-trips through `_restore_stage_input()` without adding endpoint, credentials, queue, worker, SDK objects, or ClearML Task ID. Queue remains only in the operational remote-launch description.
- Remote launch is pinned to the input `source_commit` and the actual W004 symbol `mldb_v2.src.backend.execution_harness.CommonExecutionHarness`; no training/evaluation/model-family runner is introduced.
- Required training/evaluation metadata, canonical refs, evaluation stage/coordinate, Model/Architecture where applicable, source commit, ownership key, and canonical public parameters are covered by focused tests.
- Focused T005-02 pytest: **14 passed**. Focused T005-01 + T005-02 join: **30 passed**. Full `mldb_v2/tests` was intentionally not run.
- Changed Python `py_compile`: PASS. Import smoke: PASS. AST/dependency/genericity scan: PASS. Direct whitespace scan and `git diff --check --` on owned Python paths: PASS.
- W004-owned harness/value modules, T005-01 registry/config modules, Frozen Specs/Skeletons, Work Item status, `tasks/index.md`, canonical data, and all other scope-external paths were not modified. No commit/add/stash/reset/clean/restore/push was performed.
- No T005-02 implementation blocker remains. W005 integration/completion remains gated by formal W004 closure and later T005-03/T005-04/T005-05 work.
