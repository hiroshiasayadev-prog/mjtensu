# PRODUCT-TASK-APPLICATION-001-01: Implement scoring session service

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production scoring-session service implementation
  - PRODUCT-TASK-APPLICATION-001-01

## Goal

Implement the page-independent ScoringSessionService state transitions, defaults, scoring orchestration, winning-tile preservation, and latest-result invalidation defined by the Application contracts.

## Work

- Implement session creation from a committed `RecognizedStructure` and explicit rule profile.
- Apply the accepted initial Tsumo/East round/East seat/Riichi None/all-situational-false condition set.
- Select the rightmost completed-hand tile as the initial winning tile only.
- Implement winning-tile selection among completed-hand instances.
- Implement structure replacement with stable-ID selection preservation and corrected-rightmost fallback when the selected instance no longer remains in completed hand.
- Implement condition/rule-profile replacement and score-relevant latest-result invalidation.
- Implement preview/calculate orchestration against a public ScoringService dependency without concrete Agari knowledge.
- Add focused state-transition tests using a deterministic fake ScoringService.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| session creation/defaults | Create one active scoring session from committed structure + rule profile with the accepted initial conditions and initial rightmost winning tile. | Exact initial state matches `application-session-api` and contains one non-null completed-hand winningTileId. | Focused creation tests. |
| winning-tile mutation | Allow user selection of any completed-hand instance and reject/outcome invalid selection according to the service contract. | Duplicate tile identities remain distinguishable by `TileInstanceId`. | Identity/selection tests. |
| structure replacement | Preserve selected winning tile when the same instance ID remains in completed hand, including identity correction; otherwise default to corrected rightmost. | Replace/delete/move cases produce exact specified selection semantics. | Table-driven replacement tests. |
| scoring orchestration | Delegate preview/calculate to public ScoringService and retain only successful latest result. | Score-relevant mutation invalidates stale latestResult; no concrete Agari types enter Application. | Fake-service orchestration tests. |
| rule/condition replacement | Apply accepted command semantics while leaving normalization ownership to the shared condition policy path. | Replaced values become session state and invalidate score as required. | State mutation tests. |

## Done condition

ScoringSessionService matches the accepted Application state/command contract and passes focused tests for defaults, selection identity, structure replacement, orchestration, and stale-result invalidation.

## Verification

- Run session creation/default tests.
- Run winning-tile identity/replacement matrix tests.
- Run score-result invalidation tests.
- Run fake ScoringService preview/calculate orchestration tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.application.scoring_session` and `spec:product.system.contracts.application_session_api` are the implementation authorities.
- `spec:product.system.contracts.testing_strategy` requires these Application transitions at the deterministic layer.
- Added canonical domain types for `TileInstanceId`, `TileInstance`, `RecognizedMeldGroup`, and `RecognizedStructure` in `product/frontend/src/domain/index.ts`.
- Added the public scoring contract surface in `product/frontend/src/scoring/index.ts`, including draft/strict inputs, rule profiles, preview/calculation result types, and `ScoringService`.
- Implemented `createScoringSessionService()` in `product/frontend/src/application/scoring-session-service.ts`.
- Session creation now installs `INITIAL_SCORING_CONDITIONS`, preserves the explicit rule profile, selects only the rightmost completed-hand tile, and leaves `latestResult` null.
- Session updates support completed-hand winning-tile selection, stable-ID preservation on structure replacement, corrected-rightmost fallback when the selected instance is gone or moved out of completed hand, condition/rule replacement, and score-result invalidation.
- Preview delegates the current `ScoringDraft` to the public `ScoringService`; calculate builds strict `ScoringInput`, delegates to `ScoringService.calculate()`, and installs only successful returned calculations as `latestResult`.
- Added focused fake-service tests in `product/frontend/test/scoring-session-service.test.ts` for defaults, identity selection, replacement matrix behavior, invalidation, preview orchestration, strict calculate orchestration, and failed-calculation non-installation.
- Adjusted the existing condition-policy normalization implementation to avoid excessive per-draft object copying so the full deterministic condition/session suite completes within the Vitest timeout.
- Focused verification PASS: `npm test -- scoring-session-service.test.ts scoring-condition-policy.test.ts` completed 2 test files / 38 tests.
- Strict typecheck PASS: `npm run typecheck`.
- Full Vitest PASS: `npm test` completed 10 test files / 82 tests.
- Architecture lint PASS: `npm run lint` reported `Architecture import boundaries: OK (16 source files checked)`.
- Production build PASS: `npm run build` completed TypeScript check and Vite production build successfully.
