# PRODUCT-TASK-APPLICATION-001-07: Correct session-store invariant bypass

- **status**: completed
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-06
- **finding_refs**:
  - PRODUCT-TASK-APPLICATION-001-06/F-MAJ-02
- **outputs**:
  - corrected Application store mutation boundary
  - PRODUCT-TASK-APPLICATION-001-07

## Goal

Correct F-MAJ-02 by removing the production UI-facing ability to replace an active `ScoringSessionState` wholesale without passing through Application session semantics.

## Work

- Remove or narrow the public `installScoringSession()` mutation seam so ordinary production flow cannot inject arbitrary `winningTileId`, conditions, rule profile, or `latestResult` state.
- Route production session creation, semantic updates, calculation installation, correction replacement, and new-recognition reset through Application-owned semantic store actions backed by `ScoringSessionService`.
- Preserve explicit no-session state and route-consumable selectors.
- Preserve the invariant that every score-relevant mutation invalidates `latestResult` unless the result was produced by the exact successful Application calculation being installed.
- If a bounded construction/test hydration seam is still required, keep it outside the ordinary UI-facing mutable store contract and make its non-production role structurally explicit.
- Update directly affected focused tests and integration bindings without changing scoring, recognition, or presentation semantics.

## Done condition

Ordinary production callers cannot replace the active scoring session in a way that bypasses `ScoringSessionService` invariants, while recognition creation, Conditions mutation, Result correction, calculation, reset, and route guards continue to work through semantic Application actions.

## Verification

- Run focused Application store/session tests.
- Run Conditions and Result correction integration/component tests affected by the store API change.
- Confirm stale-result invalidation cannot be bypassed through the exported production store surface.
- Run `npm run typecheck`.
- Run `npm run lint`.

## Evidence

- A06 F-MAJ-02 identified `installScoringSession(session)` as an unrestricted exact-state replacement path on exported `ApplicationStoreState`.
- Removed `installScoringSession()` from `ApplicationStoreState`; construction/test hydration remains available only through the explicitly named `ApplicationStoreHydrationState` argument to `createApplicationStore()`.
- `createScoringSession()` and `updateScoringSession()` now return the exact service-produced session they install, allowing production UI bindings to remain semantic without a whole-session setter.
- Recognition confirmation now calls the Application store `createScoringSession()` action instead of constructing a session in UI code and installing it wholesale.
- Conditions and Result-origin recognition correction now use a store-backed `update` / `preview` / `calculate` adapter whose mutations delegate to `ScoringSessionService` through Application store actions; production pages no longer bind `onSessionChange` to a whole-session store replacement.
- Removed `ScoringSessionService` from the UI scoring-flow context, so production UI composition no longer exposes a parallel direct session-semantic path beside the Application store.
- Updated Application-store, Recognition, route, Result, and fake-service E2E fixtures so stores that exercise semantic session actions receive an injected `ScoringSessionService`.
- Added focused store coverage asserting that hydrated construction remains possible while `installScoringSession` is absent from the exported mutable store surface.
- `spec:product.system.contracts.application_session_api` requires UI mutations and result staleness semantics to remain owned by the Application boundary.
- This correction does not move scoring or recognition solver logic into the store.
- Verification executed from `product/frontend` on 2026-08-27:
  - `npm test -- application-store.test.ts recognition-page.test.tsx conditions-page.test.tsx tile-correction-ui.test.tsx result-page.test.tsx shell-routing.test.tsx`: PASS, 6 files / 72 tests.
  - `npm run typecheck`: PASS.
  - `npm run lint`: PASS, `Architecture import boundaries: OK (51 source files checked)`.
  - Existing React inline-style shorthand/non-shorthand warning from `conditions-page.test.tsx` remained non-failing and did not affect the Application invariant verification.
- Final correction verdict: **PASS**. F-MAJ-02 is corrected: the production mutable store surface no longer permits arbitrary whole-session replacement, and production flows preserve Application-owned session semantics.
