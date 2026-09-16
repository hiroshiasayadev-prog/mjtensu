# MLDB-V2-WORK-009: End-to-end conformance

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002, MLDB-V2-WORK-003, MLDB-V2-WORK-004, MLDB-V2-WORK-005, MLDB-V2-WORK-006, MLDB-V2-WORK-007, MLDB-V2-WORK-008]
- **source_refs**: `spec:mldb.v2`, all frozen Skeleton modules, and repository examples under `mldb_data/`

## Goal
Independently verify that the complete implementation is contract-conformant from authored definitions through planning, backend execution, canonical acceptance/history, discovery, and CLI behavior.

## Boundary
Own end-to-end/conformance tests and review evidence. Do not redesign or silently repair implementation contracts inside the final review task; named findings go back to the owning Work Item.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-009-01 | Run static Specification↔Skeleton↔src public-shape/import/dependency conformance and forbidden-infrastructure scans. | W001-W008 |
| MLDB-V2-TASK-009-02 | Run repository/example conformance for classifier, rotated detector, Evaluation grids, and custom crop-quality protocol. | T01 |
| MLDB-V2-TASK-009-03 | Run deterministic local/fake-backend lifecycle tests covering success, failure, cancellation, retry history, interruption, and existing-Model reevaluation. | T01 |
| MLDB-V2-TASK-009-04 | Run actual ClearML + S3-compatible end-to-end smoke with one bounded Study and verify canonical/backend separation. | T02,T03 |
| MLDB-V2-TASK-009-05 | Perform independent integrated review and record PASS or NEEDS REVISION with unresolved findings. | T01-T04 |

## Completion condition
- No unresolved Specification/Skeleton/src contract mismatch remains.
- Generic examples demonstrate that MLDB core is not classifier-, detector-, metric-, or single-Evaluation-condition specific.
- Canonical Git records remain sufficient to reconstruct experiment conditions/results without treating ClearML state as authority.
- ClearML owns operational queue/agent/retry/liveness and MLDB does not reimplement them.
- Final review records PASS with all focused/actual-backend verification evidence, or names exact blocking findings without self-closing them.

## Closure — 2026-09-15

**Final review: PASS.** W009's authored-definition → planning → backend execution → canonical acceptance/history → query/CLI path is conformant with no unresolved Specification/Skeleton/src mismatch found.

- **T009-01 / static-integrated conformance:** exact public-shape/Skeleton mirror, dependency-hygiene, genericity/secret-boundary, repository, API, backend, and CLI tests are included in the full `mldb_v2/tests` regression. `git diff --check -- mldb_v2` is clean.
- **T009-02 / repository examples:** classifier and rotated-detector executable examples, multi-architecture detector definitions, Evaluation grids, and `rotated-fcos/detector-crop-quality-v1` are covered by executable-integrity/definition-loading tests. The classifier rotation-robustness example now fully verifies because its dependencies are sealed. The detector spatial-screen example validates structurally and is intentionally bounded by `referenced_definition_not_sealed` while its additional architectures/crop-quality protocol remain draft; this is lifecycle state, not a contract mismatch.
- **T009-03 / deterministic lifecycle:** closure coverage includes interrupted training/evaluation recovery, cancellation drain/retry, competing callers, terminal replay, mixed evaluation sibling outcomes, failed candidates, and existing-Model reevaluation without duplicate parent mutation.
- **T009-04 / actual ClearML + S3, detector:** `rotated-fcos/run-d839f1ef2f3b4febae61f7f645f12d7e` completed from source `22afb1fc9445d86876ad60391bc7fe352b7d9662`. Training execution `0583a84318f547d4b6181c7104b0a56f` and evaluation execution `db08069eece44b5ba3147773a5e1ad09` both completed. Ten-epoch weights were immutably published to S3 (910414 bytes, SHA-256 `009cd317e966617274eb581c3a2906094bb0f372c61017debc938b065e4be305`); canonical evaluation recorded precision 0.8208333, recall 0.9949495, F1 0.8995434, mean rotated IoU 0.7615188.
- **T009-04 / actual ClearML + S3, classifier:** `tile-classifier/run-22f4df1c723949f6b87a7c688db01574` completed from the same source commit. Training execution `78a03f6225964a8e964241d5e44ad46e` and evaluation execution `c6d4c271a1474e9c97e068c01f58522c` both completed. Ten-epoch weights were immutably published to S3 (1504834 bytes, SHA-256 `cb140fc5ab8dbe0f5ea17db0849c99606e0fcc55d7169c78e5e3ea8c87dc3a53`); canonical angle-robustness evaluation over 0/15/30/45 degrees recorded manual accuracy@0 0.9244444, JP accuracy@0 0.9297059, manual angle mean 0.5277778, JP angle mean 0.5846691.
- These actual runs preserve the required authority split: ClearML owns queue/worker/liveness/execution IDs; MLDB canonical Study/Training/Evaluation results and immutable S3 artifact identities are sufficient to reconstruct the accepted experiment result without treating ClearML state as canonical truth.
- **T009-05 / final integrated review:** the stale current-example lifecycle expectation was corrected after classifier dependencies became sealed; focused `test_definition_lifecycle.py` passed **31/31**. Final full `python -m pytest mldb_v2/tests -q` passed **1344 passed, 3 skipped in 348.94s**. No production-code change was required for this final finding.

## Post-closure amendment — 2026-09-15

The W009 PASS above records conformance against the contract frozen and implemented at closure time. The later execution-telemetry/observability requirement discovered from actual ClearML `No chart data` behavior is an approved post-closure Specification amendment tracked by `MLDB-V2-WORK-010`; it does not retroactively reopen or rewrite W009 evidence.
