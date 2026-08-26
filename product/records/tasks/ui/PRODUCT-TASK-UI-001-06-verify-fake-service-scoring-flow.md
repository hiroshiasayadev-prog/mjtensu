# PRODUCT-TASK-UI-001-06: Verify fake-service scoring flow

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-UI-001-01
  - PRODUCT-TASK-UI-001-02
  - PRODUCT-TASK-UI-001-03
  - PRODUCT-TASK-UI-001-04
  - PRODUCT-TASK-UI-001-05
- **outputs**:
  - PRODUCT-TASK-UI-001-06

## Goal

Execute one deterministic browser-level acceptance gate for the complete visible scoring flow and recovery paths using public-contract fake Camera, Recognition, Scoring, and Application dependencies.

## Work

- Run Playwright through Top -> Recognition -> Conditions -> Result using deterministic recognition/scoring results.
- Verify Recognition preparation, camera/runtime failure, retry, and stable auto-transition behavior.
- Verify Recognition -> Conditions history replacement and guarded-route behavior.
- Verify Conditions winning-tile selection, condition editing, preview/readiness, calculation, and structural correction entry.
- Verify Result condition correction, recognition correction, immediate recalculation, Conditions fallback after confirmed repair-needed correction, and explicit new Recognition.
- Verify stale pre-correction Result is not restored after a confirmed structural correction.
- Verify Help round-trip and session preservation behavior.
- Record expected/observed outcomes and one PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined fake-service browser-flow and recovery check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result | observed result |
|---|---|---|
| Top -> Recognition -> Conditions -> Result | PASS | PASS - Chromium E2E primary flow passed |
| Recognition startup/preparation state matrix | PASS | PASS - camera-first and runtime-first cases passed |
| camera/runtime retry ownership | PASS | PASS - both owner-specific retry cases passed |
| Recognition -> Conditions history replacement | PASS | PASS - browser Back returned to Top rather than reopening completed Recognition |
| winning-tile/condition interaction | PASS | PASS - corrected exact-match locator rerun completed successfully. |
| no-yaku / invalid-input / invalid-shape visible recovery | PASS | PASS - all three visible recovery states passed on rerun. |
| Result -> Conditions -> Result condition correction | PASS | PASS - changed seat wind recalculated and replaced the Result |
| Result -> recognition correction -> immediate Result | PASS | PASS - confirmed correction recalculated immediately to the new Result |
| confirmed recognition correction -> Conditions fallback | PASS | PASS - repair-needed correction routed to Conditions |
| pre-confirm correction cancel preserves old Result | PASS | PASS - cancelled local correction preserved the old Result |
| confirmed correction never restores stale old Result | PASS | PASS - Back after confirmed repair-needed correction showed no old score |
| explicit new Recognition replaces prior session | PASS | PASS - second recognition installed `recognition-2-*` session and old Result was not restored |
| no-session Conditions/Result route guards | PASS | PASS - both guarded routes redirected to Top |
| Help navigation/session preservation | PASS | PASS - Help round-trip/history returned to the existing Result |
| strict typecheck/lint/architecture gate | PASS | PASS - `npm run typecheck`, `npm run lint`, and `npm test` all pass; architecture gate reports 47 source files checked and Vitest reports 26 files / 232 tests passed. |

**Overall verification verdict: PASS.** Every predefined fake-service browser-flow/recovery check is PASS, the focused Playwright suite completed 12/12 tests successfully, and the required strict typecheck/lint/architecture/unit-test gates are green.

## Evidence

- `spec:product.system.contracts.testing_strategy` defines this L3 fake-service browser acceptance layer.
- The fake services emit only public product contract values; real ONNX/WASM compatibility remains covered by feature L2 and final integration verification.
- Added `product/frontend/src/ui/scoring-flow-services.tsx` as a public-contract dependency seam so browser tests can supply deterministic ScoringSession/Correction services without importing fake implementations into production modules.
- Added the E2E-only multi-page Vite build mode and `product/frontend/test/e2e/fake-flow.html` / `fake-flow-main.tsx`; the normal production build does not include the fake-flow HTML entry.
- Added `product/frontend/test/e2e/fake-service-scoring-flow.spec.ts` covering the complete U06 check matrix with deterministic Camera/Recognition/Scoring inputs and the real Application session service/store boundary.
- Prepared fixture identities: recognition runs use `recognition-<n>-hand-<1..14>` plus `recognition-<n>-dora-1`; deterministic scoring sentinels exercise invalid winning tile (`9s`), no-yaku repair fallback (`9p` as corrected first tile), invalid input (West round + North seat), and immediate corrected recalculation (`2m` as corrected first tile).
- Installed test package metadata reports `@playwright/test` 1.62.1; Playwright's bundled Chromium metadata is 151.0.7922.34 (revision 1234), project name `chromium`.
- Execution evidence on 2026-08-26: `npm run lint` PASS (`Architecture import boundaries: OK (47 source files checked)`); `npm test` PASS (26 files / 232 tests); production build and E2E-mode Vite build both PASS.
- Initial Playwright execution on 2026-08-26: 13 Chromium tests ran; the bootstrap smoke and 11/12 fake-flow checks passed, with one fake-flow test stopped by ambiguous `リーチ` / `ダブルリーチ` locator matching. The selector was corrected to `{ name: 'リーチ', exact: true }`; product behavior was not implicated by the failure.
- Rerun on 2026-08-26: `npx playwright test test/e2e/fake-service-scoring-flow.spec.ts` PASS, 12/12 fake-service scoring-flow tests in 11.7s.
- Final strict-gate rerun on 2026-08-26: `npm run typecheck` PASS after the previously unrelated Agari scoring-test typing errors were corrected. With the earlier `npm run lint` PASS, `npm test` PASS (26 files / 232 tests), production/E2E builds PASS, and Playwright fake-flow PASS (12/12), the U06 overall verdict is PASS.
