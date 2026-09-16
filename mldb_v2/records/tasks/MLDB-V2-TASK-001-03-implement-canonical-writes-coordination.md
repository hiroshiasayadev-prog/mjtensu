# MLDB-V2-TASK-001-03: Implement canonical writes and coordination

- **status**: completed
- **date**: 2026-09-09
- **work_item**: MLDB-V2-WORK-001
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-001-01]
- **outputs**: `mldb_v2/src/repository/canonical_writes.py`, `mutation_coordination.py`, focused tests

## Goal
Implement atomic/idempotent canonical record writes and short per-StudyResult mutation coordination exactly as frozen by repository contracts.

## Work
- Implement complete-file atomic replacement so readers never observe partial canonical YAML/manifest content.
- Implement immutable Plan/TrainingResult/Model/EvaluationResult create with exact-existing idempotence and differing-content lifecycle conflict.
- Implement nonterminal StudyResult replacement only through valid next representations.
- Implement process-safe repository-local StudyResult locking outside canonical `mldb_data/` history.
- Ensure lock ownership never spans backend polling, GPU work, sleeps, large artifact transfer, or scheduling.

## Done condition
Concurrent/repeated canonical mutations cannot lose valid updates, overwrite immutable history, or turn repository coordination into queue/lease/liveness infrastructure.

## Verification
Test exact-existing replay, conflicting same-ID content, atomic replacement behavior, terminal immutability, concurrent StudyResult mutation, and stale-process-safe lock recovery semantics.

## Evidence

### Finding 2 / StudyResult closure repair
- 2026-09-09 integrated review: 16 focused canonical-write/coordination tests PASS.
- Review repaired two contract bugs before completion: execution keys now validate actual UUID4 version/variant, and terminal StudyResult closure rejects failed/cancelled semantics encoded under the wrong top-level status.
- Atomic replacement, exact replay, immutable conflict behavior, process-safe lock release/crash behavior, py_compile and `git diff --check` pass.

### Finding 3 / immutable validator enforcement repair — 2026-09-10
- `mldb_v2/src/repository/canonical_writes.py` now exposes the frozen `CanonicalRecordValidator` port and requires `record_validator` as a keyword-only, non-optional `CanonicalRepositoryWriter` constructor dependency; omitted and `None` validators cannot bypass enforcement.
- `create_immutable` preserves repository-owned kind/identity/path/JSON checks, validates the exact proposed logical document through the injected port before create/replay success, and validates an existing immutable record through the same `(kind, entity_id)` port before equality may return idempotently.
- Validator failures propagate unchanged and leave canonical bytes untouched; valid differing content remains a lifecycle conflict. No StudyPlan/TrainingResult/Model/EvaluationResult domain validator or schema/status table was copied into the repository layer, and dedicated StudyResult validation/transition behavior remains separate.
- Focused verification command over canonical writes, mutation coordination, and W001 integrated probes: `40 passed`.
- Full `mldb_v2/tests` suite: `210 passed`.
- All `mldb_v2/src/repository/*.py` modules pass `py_compile`; public runtime signatures match the frozen Skeleton; scans found no Skeleton runtime import, mldb v1 import, ClearML/boto/minio dependency, or immutable-domain validator duplication; `git diff --check -- mldb_v2` passes.

## T005 Finding 3 contract-repair note — 2026-09-10

The original frozen `create_immutable` Skeleton exposed only `(kind, entity_id, document)` and did not
freeze who proves complete kind-specific canonical validity. T005 therefore found that the current
implementation can persist incomplete or wrong-kind immutable records while still satisfying its
local JSON/id checks.

The repaired frozen contract now requires a non-optional `CanonicalRecordValidator` dependency for
the generic immutable write boundary. The repository passes `kind`, `entity_id`, and the exact
document to that port; owning Study/Training/Evaluation validators or their composition layer own
format semantics. Repository code must not copy StudyPlan, TrainingResult, Model, or
EvaluationResult validators.

Exact T003 implementation repair required after this contract update:

1. require `record_validator` when constructing the concrete `CanonicalRepositoryWriter`; provide no
   implicit/pass-through/accept-all production default;
2. in `create_immutable`, run the configured validator for the proposed record before any create or
   replay can succeed;
3. when the path already exists, validate the existing record through the same port for the same kind
   and identity before exact-content equality may be returned as idempotent replay;
4. preserve repository-owned supported-kind, typed-ID/path, namespace, atomic-write, locking,
   exact-replay, and same-ID conflict semantics;
5. a validation failure must leave canonical storage unchanged and must never be reclassified as an
   idempotent replay or lifecycle conflict;
6. repository-focused tests may inject a strict/recording fake validator to exercise write mechanics,
   but the previous `{schema,id,value}` pseudo-Plan is not evidence of StudyPlan conformance. Domain
   integration must use the owning real/composite validator.

This Finding 3 repair does not redesign the dedicated StudyResult create/replace path. T005 Finding 2
remains the separate owner of StudyResult closure/status repair. No `src/` change was made by this
contract-repair Task, so the existing T003 implementation evidence predates and does not satisfy the
new frozen validator dependency.

## T005 re-verification reopen — 2026-09-10
- W001-RV02 remains open: status: failed currently permits a mixed global_failure + study_cancelled closure. T003 is reopened until cancellation-only study_cancelled semantics are excluded from failed closure while preserving the dedicated cancelling -> cancelled path.

## T005 W001-RV02 StudyResult closure repair — 2026-09-10
- `_validate_study_result_record` now rejects `status: failed` whenever any skipped stage uses `reason: study_cancelled`; the existing requirements for a non-null diagnostic and at least one `global_failure` skip remain unchanged.
- The repair is deliberately narrow: failed closure still accepts already-terminal ordinary `completed`, `failed`, `cancelled`, `upstream_failed`, and `upstream_cancelled` stage semantics when the frozen invariants otherwise hold. The `cancelling -> cancelled` transition is unchanged, and cancelled closure remains valid both with `skipped: study_cancelled` and with all active children closing as `cancelled` results.
- The T005 RV02 integrated probe remains intact and now passes. Focused canonical-write/mutation/W001 verification: `46 passed`; full `mldb_v2/tests`: `231 passed`.
- All 6 `mldb_v2/src/repository/*.py` modules pass `py_compile`; normalized public method parameter shape matches the frozen canonical-writes Skeleton; `git diff --check -- mldb_v2` passes.
- T003 is returned to `completed`. T005/W001 status is intentionally unchanged and remains owned by independent re-verification.

## Final T005 re-verification reopen — 2026-09-10
- W001-RV04 remains open: non-terminal `submitted` / `cancelling` StudyResult records can still be accepted with zero planned stage slots.
- W001-RV05 remains open: `created_at` currently accepts broader ISO-8601 forms than the frozen RFC3339 UTC-`Z` lexical contract.
- W001-RV06 remains open: StudyResult identity is required by the frozen format to live in the source Study namespace, and Plan identity is tied to that Study namespace; the writer does not yet reject cross-namespace Study/Plan references.
- T003 is reopened for these three focused validation repairs. V02/V03/RV02 behavior must not regress.
- T005/W001 remain `planned` until another independent verification closes the findings.


## T005 W001-RV04/RV05/RV06 StudyResult validation repair ? 2026-09-10
- RV04: `submitted` / `cancelling` now require at least one `pending` planned stage even when the collected disposition topology is empty; valid pending topology and existing-Model `training: null` with planned Evaluations remain valid.
- RV05: `created_at` now requires the exact RFC3339 UTC-`Z` lexical shape `YYYY-MM-DDTHH:MM:SS[.fraction]Z` before calendar parsing. Week/ordinal/reduced-time/space/offset/missing-zone/lowercase-zone forms and invalid calendar dates are rejected; fractional precision is not fixed to three digits.
- RV06: StudyResult ID namespace, referenced Study namespace, and referenced Plan namespace must match. The check is identity-only and performs no Plan hash/prefix/content lookup or cross-namespace dependency ban.
- Existing UUID4/RFC variant, closure/status, cancellation, global-failure, immutable-slot/topology, validator injection, validation order, and atomic write semantics remain covered by regression tests; T005 RV04/RV05/RV06 probes were preserved unchanged.
- T003 canonical-write unit tests: `49 passed`. Requested focused canonical-write/mutation/W001 verification: `68 passed`. Full `mldb_v2/tests`: `267 passed`.
- All 6 `mldb_v2/src/repository/*.py` modules pass `py_compile`; normalized `CanonicalRecordValidator` / `CanonicalRepositoryWriter` public parameter shapes match the frozen Skeleton; `git diff --check -- mldb_v2` passes.
- Final repository status remains `?? mldb_v2/`, the pre-existing untracked-tree state. No commit was created. T005/W001 status was not changed.

## Final adversarial T005 reopen — 2026-09-10
- W001-RV07 remains open: `_canonical_path` only checks for `namespace.yaml` existence and does not validate that namespace metadata ID matches the directory before canonical writes.
- W001-RV08 remains open: StudyResult trial/evaluation validation derives `trial-{index:04d}` / `eval-{index:04d}` and therefore accepts index 10000+, outside the frozen four-digit `trial-NNNN` / `eval-NNNN` grammar.
- T003 is reopened for these two narrow repository/StudyResult validation repairs. Existing CanonicalRecordValidator, StudyResult closure, UUID4, RFC3339, namespace source identity, atomicity, and coordination behavior must remain unchanged.
- T005/W001 remain `planned` until independent re-verification closes the findings.

## W001-RV07/RV08 repair closure ? 2026-09-11
- RV07: canonical write path derivation now reuses repository Namespace validation via `_read_namespace_document`; malformed Namespace YAML, wrong schema, invalid/mismatched Namespace ID, and missing `namespace.yaml` cannot authorize immutable or StudyResult writes.
- RV08: StudyResult trials/evaluation coordinates now validate the actual authored IDs through `_validate_trial_id` / `_validate_evaluation_coordinate_id` before contiguous-order comparison, so `trial-0000`, `trial-10000`, `eval-0000`, `eval-10000`, and malformed spellings are rejected.
- Fresh coordinator verification: canonical-write/mutation/integrated focused set **94 passed**; full `mldb_v2/tests` after the parallel T003/T004 repairs **341 passed** before the storage coordinator hardening below.
- Manual adversarial probes confirm Namespace ID mismatch, `trial-10000`, and `eval-10000` are rejected. All `mldb_v2/src` Python files py_compile and actual canonical listing remains **18 items / 0 issues**.
- T003 is returned to `completed`; T005/W001 remain `planned` for independent verification.
