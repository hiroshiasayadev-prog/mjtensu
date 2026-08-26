# PRODUCT-TASK-UI-001-08: Correct Result-origin condition correction transaction

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: correction
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-UI-001-07
- **finding_refs**:
  - PRODUCT-TASK-UI-001-07/F-MAJ-01
  - PRODUCT-TASK-UI-001-07/F-MIN-01
- **outputs**:
  - contract-conformant Result-origin Conditions correction flow
  - PRODUCT-TASK-UI-001-08

## Goal

Correct F-MAJ-01 and F-MIN-01 by making Result-origin condition correction an explicit cancellable edit transaction that preserves the unchanged Result until accepted, while making the Result dealer/child shortcut actually focus the seat-wind control.

## Work

- Distinguish ordinary initial Conditions flow from Conditions entered from Result for condition correction and from Conditions entered after a confirmed Recognition correction.
- Keep Result-origin condition edits local/uncommitted until successful acceptance rather than mutating canonical Application state on each control interaction.
- Add an explicit cancel path for Result-origin condition correction that returns to the unchanged prior Result with its exact prior session/result state intact.
- On accepted Result-origin correction, apply the edited condition/winning-tile state through Application-owned semantic actions and recalculate; install/show Result only from the successful recalculation state.
- Preserve the existing post-confirmed-Recognition-correction rule: that repair continuation must not restore or cancel back to the stale pre-correction Result.
- Keep recognized structure unchanged on the `条件を修正` path; structural editing remains owned by the dedicated Recognition-correction path.
- Consume the Result `focus: seatWind` navigation hint and visibly focus/emphasize the seat-wind control without introducing an independent dealer flag.
- Update focused Conditions/navigation/Result tests and the fake-service browser flow for cancel, commit, focus, and stale-result outcomes.

## Done condition

Result-origin condition correction can be edited and cancelled without changing the canonical scoring session or invalidating its current Result; accepted edits recalculate through Application semantics and replace Result atomically; post-confirmed Recognition repair remains non-cancellable to the stale Result; the dealer/child shortcut visibly targets seat wind.

## Verification

- Verify Result -> Conditions edit -> cancel returns the exact unchanged Result/session.
- Verify Result -> Conditions edit -> calculate installs only the newly calculated state and returns to Result.
- Verify structural correction is not exposed through the Result-origin `条件を修正` mode.
- Verify Conditions entered after confirmed Recognition correction cannot restore the stale prior Result.
- Verify `親` / `子` shortcut focuses/emphasizes seat wind and still derives dealer status only from seat wind.
- Run affected Conditions, Result, navigation-history, shell-routing, and fake-service Playwright tests.
- Run `npm run typecheck` and `npm run lint`.

## Evidence

- U07 F-MAJ-01 found that ordinary Conditions mutations were installed immediately into Application state on the Result-origin correction path, invalidating `latestResult` before correction acceptance and leaving no valid cancel-to-old-Result behavior.
- U07 F-MIN-01 found that navigation produced `{ focus: 'seatWind' }` but Conditions never consumed it.
- Result-origin condition correction now carries an explicit route-state marker and uses the pure `ScoringSessionService` from `ScoringFlowServices` as a page-local edit transaction. Winning-tile and condition changes therefore leave the canonical Application session and its current `latestResult` untouched until acceptance.
- Result-origin correction exposes `キャンセル`, omits the structural tile-correction editor, and replaces the correction history entry with the unchanged Result on cancellation.
- Accepted Result-origin edits first complete the local scoring calculation, then apply winning-tile and condition changes through Application-owned semantic session actions, recalculate through Application, and navigate to Result only after that recalculation succeeds.
- Conditions entered after confirmed Recognition correction continue to use the store-backed semantic session service and intentionally expose no cancel-to-stale-Result path.
- The Result dealer/child shortcut now opens the same Result-origin condition transaction with `focus: 'seatWind'`; Conditions focuses and visibly emphasizes the seat-wind fieldset while dealer status remains derived solely from `conditions.seatWind`.
- The unscored Result fallback uses a separate ordinary Conditions navigation helper so it is not accidentally treated as a cancellable old-Result correction.
- Focused Conditions, Result, navigation-history, shell-routing, and fake-service Playwright coverage was updated for local edit/cancel, accepted commit, structural-editor exclusion, post-Recognition stale-result behavior, and seat-wind focus.
- `spec:product.ui.screen_flow` and `spec:product.ui.pages.conditions` require cancellable Result-origin condition correction but forbid stale Result restoration after a confirmed Recognition correction.
- `spec:product.ui.pages.result` and `spec:product.ui.components.condition_controls` require the dealer/child shortcut to target seat wind rather than creating another dealer-state source of truth.
- User verification on 2026-08-27: fake-service Playwright flow passed 14/14; `npm run typecheck` passed; `npm run lint` passed with `Architecture import boundaries: OK (52 source files checked)`.
- The first focused Vitest run passed 64/65 tests; the only failure was an accessibility-query collision introduced by adding `aria-label` to every radio fieldset (`getByLabelText('リーチ')` matched both the radio and fieldset). The fieldset label override was removed so the native `legend` supplies the group name without colliding with label queries.
- The same run also surfaced React inline-style warnings from mixing `border` shorthand with `borderColor`; the selected-option style now replaces the complete `border` shorthand instead.
- Final focused Vitest rerun passed on 2026-08-27: 5/5 test files and 65/65 tests passed (`conditions-page`, `result-page`, `navigation-history`, `shell-routing`, and `tile-correction-ui`).
- Final verification status: focused Vitest PASS, fake-service Playwright 14/14 PASS, `npm run typecheck` PASS, and `npm run lint` PASS. PRODUCT-TASK-UI-001-08 Done condition is satisfied.
