# PRODUCT-TASK-APPLICATION-001-05: Verify Application contracts

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: verification
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-04
- **outputs**:
  - PRODUCT-TASK-APPLICATION-001-05

## Goal

Execute one objective Application acceptance gate covering session defaults/transitions, shared condition policy, correction semantics, store ownership, and fake-service scoring orchestration.

## Work

- Execute the complete focused Application test suite from A01 through A04.
- Verify initial session defaults and winning-tile selection behavior.
- Verify condition normalization/availability and idempotence.
- Verify correction temporary-invalid/validated-commit semantics and TileInstanceId preservation.
- Verify result invalidation and Result-origin correction routing state needed by UI.
- Verify the Zustand state contains no prohibited lifecycle runtime resources.
- Record expected/observed results and one PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined Application contract check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result |
|---|---|
| complete focused Application test suite | PASS |
| initial session conditions/winning-tile defaults | exact accepted state |
| condition policy normalization/availability matrix | PASS |
| correction command/identity/commit matrix | PASS |
| stale result invalidation/recalculation state transitions | PASS |
| Zustand runtime-resource isolation | PASS |
| strict typecheck/lint/architecture gate | PASS |

The overall result is PASS only when every required check is PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` defines the minimum Application test requirements.
- Exact command outputs and the final verdict are recorded here when executed.
