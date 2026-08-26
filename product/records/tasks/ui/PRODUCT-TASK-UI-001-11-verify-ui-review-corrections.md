# PRODUCT-TASK-UI-001-11: Verify UI review corrections

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: verification
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-UI-001-08
  - PRODUCT-TASK-UI-001-09
  - PRODUCT-TASK-UI-001-10
- **finding_refs**:
  - PRODUCT-TASK-UI-001-07/F-MAJ-01
  - PRODUCT-TASK-UI-001-07/F-MAJ-02
  - PRODUCT-TASK-UI-001-07/F-MAJ-03
  - PRODUCT-TASK-UI-001-07/F-MIN-01
- **outputs**:
  - objective verification evidence for U07 corrections
  - PRODUCT-TASK-UI-001-11

## Goal

Verify that U08 through U10 resolve every U07 finding without regressing the accepted production scoring-flow behavior or architecture boundary.

## Work

- Re-run focused component and navigation tests affected by the correction Tasks.
- Re-run the deterministic fake-service browser scoring flow with fakes implementing the same public feature contracts used by production.
- Add and execute the missing Result -> Conditions edit -> cancel acceptance path and prove the prior Result/session remains unchanged.
- Prove accepted Result-origin correction recalculates and replaces Result while post-confirmed Recognition correction cannot restore stale score state.
- Prove Recognition UI composes against public Camera/Recognition types without a UI-private semantic service contract.
- Prove missing scoring-flow composition cannot fabricate valid/no-yaku/correction-readiness semantics.
- Run strict typecheck, lint/architecture, unit/component tests, production build, and affected Playwright flow.
- Record expected and observed results plus one overall verdict.

## Done condition

Every U07 finding has direct objective verification evidence and the complete affected production UI gate is green.

## Verification

| check | expected result | observed result |
|---|---|---|
| Result condition edit -> cancel preserves prior Result/session | PASS | PASS (static/test coverage) - `result-page.test.tsx` keeps the exact canonical session/result object unchanged through local edits and after cancel; fake-service Playwright contains the same browser acceptance path |
| Result condition edit -> calculate replaces Result | PASS | PASS (static/test coverage) - Result-origin edit remains local until `計算する`, then the test requires a new canonical session/result with changed seat wind/winning tile and recalculated points |
| confirmed Recognition correction cannot restore stale Result | PASS | PASS (static/test coverage) - fake-service Playwright confirms repair-needed correction reaches Conditions without cancel and browser Back exposes an unscored Result with the old 6,000-point score absent |
| Result dealer/child shortcut focuses seat wind | PASS | PASS (static/test coverage) - component, Result, and Playwright checks require the `自風` group to receive focus and `data-edit-focus="true"` |
| Recognition UI/test fakes use public Camera/Recognition contracts | PASS | PASS (static) - production page and test/E2E fixtures import `CameraService`/`CameraSession` from `@/camera` and `RecognitionRuntime`/`RealtimeRecognizer`/semantic snapshot types from `@/recognition`; no page-private duplicate service/update contract remains |
| production-compatible Recognition composition typechecks | PASS | PASS - current-tree `npm run typecheck` completed successfully for application and test TypeScript projects |
| missing scoring-flow composition exposes no fabricated semantics | PASS | PASS (static/test coverage) - scoring-flow context is nullable with no fallback Scoring service; guarded pages render `点数計算サービスを利用できません。`, and shell tests require absence of `役なし`, calculate, and correction-confirm actions |
| focused UI/navigation/component tests | PASS | PASS - current-tree `npm test` passed 29/29 files and 306/306 tests, including Conditions, Result, navigation-history, shell-routing, Recognition, architecture, public-entry, correction UI, and app smoke coverage |
| fake-service Playwright scoring flow | PASS | PASS - `npm run build:e2e` produced the E2E harness and the affected Chromium fake-service flow passed 14/14 cases |
| strict typecheck/lint/architecture/unit/build gate | PASS | PASS - current-tree `npm run typecheck`, `npm run lint`, `npm test`, `npm run build`, and `npm run build:e2e` all passed; lint reports `Architecture import boundaries: OK (52 source files checked)`, Vitest reports 29 files / 306 tests, and Playwright reports 14/14 PASS |

**Overall verification verdict: PASS.** Every U07 finding has direct objective verification evidence, the current-tree strict typecheck/lint/architecture/unit/build gate is green, the E2E-mode build succeeds, and the affected fake-service Playwright flow passes 14/14 cases. F-MAJ-01, F-MAJ-02, F-MAJ-03, and F-MIN-01 are verified corrected without an observed production UI regression.

## Evidence

- U07 recorded three major and one minor unresolved findings after U06 PASS.
- U11 is the objective correction gate for U08 through U10 and feeds the subsequent independent re-review.
- F-MAJ-01 / F-MIN-01 static closure: `src/ui/navigation.ts` marks Result-origin condition correction explicitly, distinguishes confirmed Recognition repair, and carries the seat-wind focus hint. `src/ui/pages.tsx` uses the injected pure scoring-session service for Result-origin local edits, omits structural correction in that mode, exposes cancel only for Result-origin correction, and commits through Application-backed actions followed by recalculation. `test/result-page.test.tsx`, `test/navigation-history.test.ts`, and `test/e2e/fake-service-scoring-flow.spec.ts` contain direct assertions for cancel preservation, accepted replacement, seat-wind focus, and stale-result prevention.
- F-MAJ-02 static closure: `src/camera/contracts.ts` defines the public Camera contract and `src/camera/index.ts` exports it. `src/ui/recognition-page.tsx` consumes Camera/Recognition dependencies only through `@/camera` and `@/recognition`; semantic rendering reads public `bbox`, `classification`, and meld `interpretation`. `test/recognition-page.test.tsx` and `test/e2e/fake-flow-main.tsx` implement those same public contract types.
- F-MAJ-03 static closure: `src/ui/scoring-flow-services.tsx` contains only a nullable injected service-reference context and no fabricated scoring fallback. `src/app/App.tsx` installs the provider only when composition supplies services. `src/ui/pages.tsx` exposes an explicit unavailable state before Conditions or Recognition correction can invoke scoring semantics. `test/shell-routing.test.tsx` directly asserts that missing composition cannot surface `役なし`, calculation, or correction-readiness controls.
- U08 execution evidence on 2026-08-27 records focused Vitest 65/65 PASS, fake-service Playwright 14/14 PASS, `npm run typecheck` PASS, and `npm run lint` PASS.
- U09 execution evidence on 2026-08-27 records focused Vitest 24/24 PASS, fake-service Playwright 14/14 PASS, `npm run typecheck` PASS, and `npm run lint` PASS with the public Camera/Recognition contract composition.
- U10 execution evidence on 2026-08-27 records focused Vitest 61/61 PASS, `npm run typecheck` PASS, `npm run lint` PASS, and fake-service Playwright 14/14 PASS after production and E2E builds completed successfully.
- Current-tree U11 execution evidence on 2026-08-27: `npm run typecheck` PASS; `npm run lint` PASS with `Architecture import boundaries: OK (52 source files checked)`; `npm test` PASS with 29/29 files and 306/306 tests; `npm run build` PASS with the production bundle generated successfully.
- The initial U11 Playwright invocation ran immediately after the normal `npm run build` and produced 14/14 failures at the same `openHarness()` heading assertion before any flow-specific assertion. `vite.config.ts` includes `test/e2e/fake-flow.html` only when Vite runs in `mode === 'e2e'`, so this execution lacked the E2E harness and is an invalid acceptance run rather than evidence of 14 product regressions.
- Final U11 E2E rerun on 2026-08-27: `npm run build:e2e` PASS and `npx playwright test test/e2e/fake-service-scoring-flow.spec.ts` PASS with 14/14 Chromium tests in 13.4s. This includes Result-origin correction commit/cancel, seat-wind focus, stale-Result prevention after confirmed Recognition correction, explicit new Recognition session replacement, route guards, and Help/session preservation.
- Git inspection is not authorized for this repository by the available Git tool, so no current worktree diff is fabricated in this Evidence.
