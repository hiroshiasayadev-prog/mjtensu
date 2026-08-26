# PRODUCT-TASK-APPLICATION-001-09: Reverify Application contracts after review corrections

- **status**: completed
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: verification
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-07
  - PRODUCT-TASK-APPLICATION-001-08
- **finding_refs**:
  - PRODUCT-TASK-APPLICATION-001-06/F-MAJ-01
- **outputs**:
  - current-state Application acceptance evidence
  - PRODUCT-TASK-APPLICATION-001-09

## Goal

Close the A06 verification-provenance gap by executing the complete Application acceptance gate against the current source after all A06 implementation/contract corrections are complete.

## Work

- Execute the full A05 focused Application acceptance suite against the current post-A07/A08 source.
- Verify session defaults/transitions, shared condition policy, correction semantics, store ownership, stale-result invalidation, and Result-origin correction routing.
- Include the corrected store mutation boundary and reconciled correction issue vocabulary in the observed acceptance evidence.
- Run strict typecheck and architecture/lint gates.
- Record exact expected/observed results and one PASS, FAIL, or validly BLOCKED verdict.
- Do not rely on the historical A05 PASS as evidence for files changed after A05.

## Done condition

The current Application implementation has one complete objective acceptance result, and every predefined Application contract check plus the A06-corrected boundaries is PASS, FAIL, or validly BLOCKED with current-state evidence.

## Verification

| check | expected result | observed result |
|---|---|---|
| complete focused Application acceptance suite | PASS | PASS: 7 files / 101 tests passed |
| session defaults/transitions and winning-tile semantics | PASS | PASS: scoring-session and Application-store coverage passed, including semantic creation/update and stale-result invalidation |
| shared condition policy | PASS | PASS: normalization/availability consistency and exhaustive condition-policy coverage passed |
| correction command/identity/commit semantics | PASS | PASS: permissive draft editing, TileInstanceId preservation, targeted validation, validated commit, and Scoring validation mapping passed |
| corrected store mutation boundary | PASS | PASS: production mutable store surface has no whole-session replacement action; construction-only hydration remains explicitly bounded |
| reconciled correction issue vocabulary | PASS | PASS: `invalid-completed-hand-tile` is consistent across accepted contract, Application mapping, and UI coverage |
| Result-origin correction routing and result invalidation | PASS | PASS: Result-origin condition/correction flows preserve or invalidate session/result state as required and route correctly |
| strict typecheck | PASS | PASS: `tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.test.json --noEmit` |
| architecture/lint gate | PASS | PASS: `Architecture import boundaries: OK (52 source files checked)` |

The overall result is PASS only when every required check is PASS.

## Evidence

- A06 F-MAJ-01 identified that A05's recorded PASS predated later changes to Application source and therefore could not establish the reviewed current state.
- A07 and A08 must complete before this gate so the verification evidence describes the exact candidate state sent to independent re-review.
- A09 supersedes A05 only as current-state verification evidence; it does not rewrite historical A05 execution evidence.
- 2026-08-27 pre-verification inspection confirms both review corrections are present in the current source before executing the acceptance gate:
  - `ApplicationStoreState` no longer exposes `installScoringSession()`; exact-state hydration is construction-only through `ApplicationStoreHydrationState`, while production creation/update/calculation actions delegate to the configured Application session port.
  - `application-store.test.ts` explicitly asserts that hydrated construction remains available while `installScoringSession` is absent from the production mutable store surface.
  - `CorrectionIssue` includes `invalid-completed-hand-tile`, `correction-draft-service.ts` maps Scoring `completed-hand-tile` to it, and `spec:product.system.contracts.correction_editor_api` declares the same mapping.
  - Focused correction service/UI coverage includes the reconciled completed-hand tile issue and preserves delegation of whole winning-shape validation to `ScoringService.validateWinningStructure()`.
- First objective acceptance execution on 2026-08-27 from `product/frontend`:
  - focused Application acceptance suite: PASS, 7 files / 99 tests.
  - `npm run lint`: PASS, `Architecture import boundaries: OK (52 source files checked)`.
  - `npm run typecheck`: FAIL with 3 TS2345 errors in `src/ui/pages.tsx` where the already-guarded `activeScoringSession` lost its non-null narrowing inside `commitResultConditionCorrection()`.
- The typecheck failure was traced to TypeScript control-flow narrowing across the nested callback rather than a session-contract failure. `pages.tsx` now captures the guarded session in a non-null `initialScoringSession` alias before the callback and passes that alias through the store-backed scoring-session adapter; the adapter continues to route mutation/calculation through Application store actions and the semantic behavior is unchanged.
- Because `pages.tsx` changed after the first test/lint evidence, the complete acceptance command set was rerun against the corrected candidate; historical A05/A07/A08 results and the pre-fix A09 run are not substituted for the final current-state evidence.
- Final objective acceptance execution on 2026-08-27 from `product/frontend`:
  - `npm test -- scoring-session-service.test.ts scoring-condition-policy.test.ts correction-draft-service.test.ts application-store.test.ts tile-correction-ui.test.tsx result-page.test.tsx architecture-boundaries.test.ts`: PASS, 7 files / 101 tests.
  - `npm run typecheck`: PASS.
  - `npm run lint`: PASS, `Architecture import boundaries: OK (52 source files checked)`.
  - `result-page.test.tsx` emitted two non-failing React style warnings about mixing `border` shorthand with `borderColor`; the suite still passed and the warnings do not affect Application contract acceptance.
- Final verification verdict: **PASS**. The exact current post-A07/A08 candidate now has complete objective Application acceptance evidence, closing A06 F-MAJ-01 and making the state eligible for A10 independent re-review.
