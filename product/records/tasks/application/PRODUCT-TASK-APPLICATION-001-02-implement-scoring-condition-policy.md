# PRODUCT-TASK-APPLICATION-001-02: Implement scoring condition policy

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: implementation
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production scoring-condition policy implementation
  - PRODUCT-TASK-APPLICATION-001-02

## Goal

Implement the one shared condition-normalization and control-availability policy consumed by Application state updates and UI controls.

## Work

- Implement pure normalization for win-method-dependent, riichi-dependent, seat-wind-dependent, and mutually exclusive situational conditions.
- Implement pure availability derivation from the same rule table rather than duplicating logic.
- Preserve the accepted rule that enabling a special condition does not mutate its prerequisites; changing a prerequisite later clears impossible dependent specials.
- Keep structure-dependent scoring requirements at the Scoring boundary rather than adding a second hand validator here.
- Add exhaustive/table-driven tests for all ordinary condition transitions and idempotence.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| condition normalization | Implement the accepted shared rule table for Ron/Tsumo, Riichi/Ippatsu, seat wind/Tenhou/Chiihou, and special-condition exclusions. | Every rule-table transition normalizes to one deterministic draft and a second normalization is idempotent. | Table-driven policy tests. |
| control availability | Derive selectable/enabled condition states from the same policy implementation. | UI availability and stored normalization cannot disagree for the same condition draft. | Availability/normalization cross-check tests. |
| prerequisite direction | Keep specials dependent on existing prerequisites rather than changing win method/seat wind/riichi automatically when a special is selected. | Selecting an available special changes only that special; changing its prerequisite later clears it when impossible. | Transition-sequence tests. |
| scoring boundary | Do not implement structure/yaku/scoring-engine requirements in this policy. | Structure-dependent invalidity remains a ScoringService outcome rather than condition-policy normalization. | Focused boundary tests plus source review. |

## Done condition

One deterministic shared condition policy implements normalization and availability for the accepted condition rule table, passes exhaustive transition/idempotence tests, and does not duplicate structure-aware scoring validation.

## Verification

- Run table-driven condition normalization tests.
- Run availability-versus-normalization consistency tests.
- Run prerequisite-change sequence tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.system.contracts.scoring_condition_policy` is the normative implementation contract.
- `spec:product.system.contracts.testing_strategy` requires shared policy behavior to be exhaustively deterministic rather than UI-only E2E logic.
- Execution results are recorded here when the Task is performed.
