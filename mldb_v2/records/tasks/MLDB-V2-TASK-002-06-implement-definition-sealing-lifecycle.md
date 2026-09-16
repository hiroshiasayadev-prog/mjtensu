# MLDB-V2-TASK-002-06: Implement definition sealing lifecycle

- **status**: completed
- **date**: 2026-09-11
- **work_item**: MLDB-V2-WORK-002
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-002-01, MLDB-V2-TASK-002-02, MLDB-V2-TASK-002-03, MLDB-V2-TASK-002-04, MLDB-V2-TASK-002-05]
- **outputs**: `mldb_v2/src/verification/definition_lifecycle.py` sealing implementation, focused lifecycle tests

## Goal
Implement the frozen `draft -> sealed` lifecycle for Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, and Study after all validation/verification gates are satisfied.

## Work
- Implement `DefinitionSealer` for the six sealable definition kinds; Namespace has no seal lifecycle.
- Build the exact sealed representation from the verified draft, adding only derived sealed integrity fields required by the owning formats: executable sibling SHA, Corpus manifest SHA/entry count, and optional Corpus builder SHA where applicable.
- Reuse T002-03 integrity evidence and T002-05 verification; do not rerun alternate validators or invent a second hashing/import path.
- Persist one complete sealed document atomically using W001 repository locking/atomic-replacement primitives, with no Git stage/commit/push side effect.
- Prevent TOCTOU sealing of stale draft bytes: the document authorized by verification must still be the document transitioned under the mutation boundary.
- Enforce sealed immutability/no return to draft and preserve identity/path; semantic or executable change after sealing requires a new revision.
- Keep StudyPlan creation, backend work, result acceptance, and concrete S3 transport implementation outside this Task. If an already-sealed `seal` replay behavior is not fixed by the frozen contract, report the ambiguity rather than inventing lifecycle semantics.

## Done condition
Every sealable definition can transition from a verified draft to one atomic immutable sealed canonical representation with all required derived integrity fields; failed gates or concurrent/stale mutation leave canonical bytes unchanged.

## Verification
Cover each definition kind, Namespace rejection, failed validation/verification, integrity-field derivation, stale/concurrent draft protection, atomic failure preservation, post-seal mutation rejection, no Git mutation, public shape conformance, py_compile/import, and `git diff --check`.

## Completion evidence ? 2026-09-12
- Added private `_RepositoryDefinitionSealer` in `mldb_v2/src/verification/_definition_sealing.py`; frozen public `definition_lifecycle.py` remains an exact Skeleton mirror.
- The sealer supports exactly Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, and Study. It derives one exact canonical target, serializes the transition with W001 `_process_file_lock` under `.local/mldb_v2/definition_seal_locks/`, verifies the current draft, obtains T002-03/T002-05 private sealing evidence, validates the complete proposed sealed mapping through the owning parser, rechecks exact authorized bytes, and performs one W001 `_atomic_replace_record`.
- Task/Study change only `status`. Executable definitions additionally receive the exact companion `implementation.sha256`; authored `implementation.sources` is preserved. Corpus receives exact manifest SHA/entry count and builder SHA only when a builder exists. T002-06 contains no duplicate file hashing path.
- Deterministic stale-draft and atomic-write-failure probes preserve newer/original bytes. Successful sealed definitions cannot be returned to draft or overwritten by a later seal transition.
- Frozen records do not define already-sealed replay success vs conflict. This implementation rejects already-sealed `seal()` as a lifecycle conflict without mutation; the ambiguity does not block mandatory first-time `draft -> sealed` completion.
- Focused `test_definition_sealing.py`: **28 passed**. T002-05 Study/lifecycle/sealing regression: **97 passed**. W002 T002-01..06 regression: **334 passed, 1 skipped**. Full `mldb_v2/tests`: **730 passed, 1 skipped**.
- Explicit all-six-kind seal/validate/verify smoke, deterministic stale probe, and injected atomic failure probe: **3 passed**. Changed Python py_compile/import smoke PASS; frozen public Skeleton exact comparison PASS.
- Actual repository read-only smoke: every current reusable definition validates locally; current Studies still fail full verification only at the expected `referenced_definition_not_sealed` gate. Canonical listing stayed **18 items / 0 issues** before/after; canonical-domain `__pycache__` stayed **0 -> 0**. No current example was sealed.
- T002-06 dependency scan is clean for mldb v1, Skeleton runtime imports, ClearML, boto/minio, `tools/`, StudyPlan/grid expansion, canonical-weight runtime, subprocess/Git commands, and direct `hashlib` use. `git diff --check -- mldb_v2` PASS.
- Final Git status retains the pre-existing tracked-flat `mldb_data` deletions, namespace-first untracked `mldb_data` migration, and untracked `mldb_v2` tree; no commit/stage/stash/restore/push was performed. W002 remains `planned` for coordinator closure review.
