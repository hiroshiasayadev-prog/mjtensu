# PRODUCT-TASK-UI-001-08: Correct Result-origin condition correction transaction

- **status**: in_progress
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
- `spec:product.ui.screen_flow` and `spec:product.ui.pages.conditions` require cancellable Result-origin condition correction but forbid stale Result restoration after a confirmed Recognition correction.
- `spec:product.ui.pages.result` and `spec:product.ui.components.condition_controls` require the dealer/child shortcut to target seat wind rather than creating another dealer-state source of truth.
