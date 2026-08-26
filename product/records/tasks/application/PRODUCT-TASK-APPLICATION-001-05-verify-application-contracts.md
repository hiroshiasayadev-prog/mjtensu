# PRODUCT-TASK-APPLICATION-001-05: Verify Application contracts

- **status**: completed
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

| check | expected result | observed result |
|---|---|---|
| complete focused Application test suite | PASS | PASS: 7 files / 98 tests passed |
| initial session conditions/winning-tile defaults | exact accepted state | PASS: Tsumo / East round / East seat / Riichi none / all situational false; rightmost completed-hand TileInstanceId selected initially |
| condition policy normalization/availability matrix | PASS | PASS: table-driven transitions, exhaustive availability/normalization consistency, and idempotence passed |
| correction command/identity/commit matrix | PASS | PASS: temporary-invalid drafts, add/replace/remove/move semantics, TileInstanceId lifecycle, targeted validation, and validated commit passed |
| stale result invalidation/recalculation state transitions | PASS | PASS: score-relevant mutations clear latestResult; Result-origin correction recalculates when ready and routes to Conditions without restoring stale result otherwise |
| Zustand runtime-resource isolation | PASS | PASS: store state-shape coverage plus architecture gate confirm only semantic scoring-session data is stored |
| strict typecheck/lint/architecture gate | PASS | PASS: strict app/test typecheck passed; architecture lint OK (37 source files checked) |

The overall result is PASS only when every required check is PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` defines the minimum Application test requirements.
- The predefined contract coverage is already present in:
  - `product/frontend/test/scoring-session-service.test.ts` for initial defaults, winning-tile identity/selection, structure replacement, stale-result invalidation, and fake ScoringService orchestration.
  - `product/frontend/test/scoring-condition-policy.test.ts` for normalization, availability consistency, prerequisite transitions, and exhaustive idempotence.
  - `product/frontend/test/correction-draft-service.test.ts` for permissive temporary-invalid drafts, command semantics, TileInstanceId lifecycle, targeted validation, and validated commit behavior.
  - `product/frontend/test/application-store.test.ts` for cross-page session ownership, service delegation, restart/disposal, result invalidation propagation, and Zustand runtime-resource isolation.
  - `product/frontend/test/tile-correction-ui.test.tsx` and `product/frontend/test/result-page.test.tsx` for Result-origin correction state preservation, stale-result removal, immediate recalculation when ready, and Conditions fallback when repair is required.
  - `product/frontend/test/architecture-boundaries.test.ts` plus `npm run lint` for the mechanically enforced architecture/runtime-resource boundary.
- Execute the objective acceptance gate from `product/frontend`:

  ```text
  npm test -- scoring-session-service.test.ts scoring-condition-policy.test.ts correction-draft-service.test.ts application-store.test.ts tile-correction-ui.test.tsx result-page.test.tsx architecture-boundaries.test.ts
  npm run typecheck
  npm run lint
  ```

- Objective acceptance gate executed on 2026-08-26 from `product/frontend`:
  - `npm test -- scoring-session-service.test.ts scoring-condition-policy.test.ts correction-draft-service.test.ts application-store.test.ts tile-correction-ui.test.tsx result-page.test.tsx architecture-boundaries.test.ts`: PASS, 7 files / 98 tests.
  - `npm run typecheck`: PASS (`tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.test.json --noEmit`).
  - `npm run lint`: PASS, `Architecture import boundaries: OK (37 source files checked)`.
- Final verification verdict: **PASS**. Every predefined Application contract check passed with no blocked or unresolved acceptance item.
