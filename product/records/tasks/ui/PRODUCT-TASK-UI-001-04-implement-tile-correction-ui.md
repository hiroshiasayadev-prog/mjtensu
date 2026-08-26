# PRODUCT-TASK-UI-001-04: Implement tile correction UI

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: implementation
- **estimate**: 2d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production tile-correction editor and Recognition correction page
  - PRODUCT-TASK-UI-001-04

## Goal

Implement the shared smartphone-friendly tile correction editor and the Result-origin Recognition correction page against the public correction-draft/Application contracts.

## Work

- Render clearly separated completed-hand, meld, and dora-indicator correction regions.
- Provide local add controls and tile selection/replacement/delete interactions using one smartphone-friendly selector surface.
- Support ordinary/red five selection and preserve editor-provided tile instance identity semantics.
- Support member/group movement/reordering as required by the correction contract without forcing one exact gesture implementation.
- Render explicit visible 明槓/暗槓 control for every four-tile kan group and allow user correction.
- Show malformed/incomplete groups and whole-structure validation issues at the smallest useful target with text in addition to color.
- Keep invalid intermediate drafts local and disable the page primary commit action until the correction service reports a supported complete winning structure.
- Implement Result-origin correction cancel/confirm behavior: cancel keeps the unchanged old Result; confirmed correction installs the new structure, then either immediately recalculates to Result or continues to Conditions when no-yaku/incomplete/invalid-input requires repair.
- Add focused component/interaction tests against deterministic correction/Application fakes.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| shared editor regions | Render completed hand, meld groups, and dora indicators with local add/edit/remove affordances. | Opening an insertion selector does not create a null placeholder; temporary malformed drafts remain visibly editable. | Component interaction tests. |
| tile selector | Support tile replacement/insertion, ordinary/red five choice, and delete. | Selector emits correction commands only; it does not mutate canonical Application state before valid commit. | Command-spy tests. |
| meld correction | Render member repair and explicit open/concealed control for every four-tile kan. | Valid 3-tile semantics require no redundant chi/pon selector; kan openness is always visible/correctable. | Meld UI matrix tests. |
| issue/readiness feedback | Show targeted repair messages/outlines and block primary commit while structural validation is invalid/non-winning. | Lack of yaku/conditions alone does not appear as correction invalidity. | Validation-state component tests. |
| Result-origin correction flow | Implement cancel and confirmed-correction follow-up according to screen flow. | Pre-confirm cancel returns unchanged Result; post-confirm stale Result is never restored; scored correction returns Result, repair-needed correction routes Conditions. | Fake-session/router flow tests. |

## Done condition

The shared correction editor and Result-origin correction page implement all accepted edit/readiness/navigation semantics and pass focused deterministic tests without owning correction-domain logic themselves.

## Verification

- Run add/replace/delete/red-five selector tests.
- Run meld member/kan-openness/reorder interaction tests.
- Run targeted validation/readiness tests.
- Run Result-origin cancel/confirm/recalculate/Conditions routing tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.ui.components.tile_correction_editor` and the Recognition correction page Specification define the visible interaction boundary.
- `spec:product.system.contracts.correction_editor_api` defines the semantic draft/command/validation interface consumed here.
- Execution results are recorded here when the Task is performed.
- 2026-08-26: Implemented reusable `product/frontend/src/ui/tile-correction-editor.tsx` with separated hand/meld/dora regions, selector-driven add/replace/delete including red fives, tile reorder/regroup commands, meld group add/remove, explicit kan openness control, targeted validation feedback, and commit gating through `CorrectionEditorService`.
- 2026-08-26: Implemented `product/frontend/src/ui/recognition-correction-page.tsx` and the guarded `/recognition/correction` route. Result-origin cancel leaves the existing session untouched; confirmed correction installs the corrected structure before preview, recalculates ready-with-yaku state to Result, and routes no-yaku/incomplete/invalid-input or calculation failure to Conditions without restoring the stale Result.
- 2026-08-26: Integrated the shared correction editor into the production Conditions page through the U03 correction slot.
- 2026-08-26: Added focused `product/frontend/test/tile-correction-ui.test.tsx` coverage for insertion without placeholders, red-five commands, replace/delete identity targeting, reorder/regroup, kan openness, local validation/readiness, and Result-origin cancel/recalculate/Conditions flow; updated route/history/result expectations for the dedicated correction route.
- 2026-08-26: Focused verification passed: `npm test -- tile-correction-ui.test.tsx conditions-page.test.tsx result-page.test.tsx navigation-history.test.ts shell-routing.test.tsx` -> 5 files, 57 tests passed.
- 2026-08-26: `npm run typecheck` passed.
- 2026-08-26: `npm run lint` passed: Architecture import boundaries OK (37 source files checked).
- 2026-08-26: React emitted existing style shorthand/non-shorthand warnings from `conditions-page.test.tsx`; they did not fail verification and are outside U04 correction semantics.
