# PRODUCT-TASK-APPLICATION-001-08: Align correction issue contract

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-06
- **finding_refs**:
  - PRODUCT-TASK-APPLICATION-001-06/F-MAJ-03
- **outputs**:
  - reconciled correction-editor public issue contract
  - PRODUCT-TASK-APPLICATION-001-08

## Goal

Correct F-MAJ-03 by making the public correction issue vocabulary and the current Scoring winning-structure validation mapping agree explicitly.

## Work

- Reconcile `spec:product.system.contracts.correction_editor_api` with the Application implementation for Scoring `WinningStructureIssue.kind === 'completed-hand-tile'`.
- Keep the correction issue product-semantic and presentation-independent; do not expose concrete Agari/WASM error types or strings.
- Prefer the existing `invalid-completed-hand-tile` semantic when retaining a distinct completed-hand tile validity issue, and document it in the accepted correction-editor public union and validation behavior.
- If the issue is intentionally collapsed into an existing accepted correction issue instead, remove the extra public discriminant from Application/UI and prove the mapping remains semantically useful. Do not leave contract and implementation divergent.
- Update exhaustive Application/UI consumers and focused tests for whichever single public vocabulary is accepted.
- Do not duplicate Scoring's winning-hand solver or structural validation logic in Application.

## Done condition

The accepted correction-editor contract, exported Application `CorrectionIssue` union, Scoring validation mapping, UI exhaustive handling, and focused tests describe one identical issue vocabulary with no undocumented discriminant.

## Verification

- Run correction-draft service tests including Scoring `completed-hand-tile` mapping.
- Run tile-correction UI tests covering the reconciled issue presentation.
- Confirm whole winning-shape determination still delegates to `ScoringService.validateWinningStructure()`.
- Run `npm run typecheck`.
- Run `npm run lint`.

## Implementation

- Accepted `invalid-completed-hand-tile` as the correction-editor public semantic for Scoring `WinningStructureIssue.kind === 'completed-hand-tile'`.
- Updated `spec:product.system.contracts.correction_editor_api` so its `CorrectionIssue` union exactly matches the exported Application vocabulary and documents the Scoring-to-Application mapping.
- Kept the existing Application mapping and UI presentation unchanged because they already implement the accepted product-semantic boundary without exposing Agari/WASM errors.
- Added focused Application coverage for `completed-hand-tile -> invalid-completed-hand-tile` alongside the other winning-structure issue mappings.
- Added focused tile-correction UI coverage proving the issue targets the completed-hand region, renders repair text, and blocks commit.
- Updated the tile-correction UI concept to include completed-hand tile validity feedback as a supported local-validation case.

## Verification

Executed from `product/frontend` on 2026-08-27:

```text
npm test -- correction-draft-service.test.ts tile-correction-ui.test.tsx
PASS: 2 files / 24 tests

npm run lint
PASS: Architecture import boundaries: OK (51 source files checked)

npm run typecheck
BLOCKED by 3 unrelated concurrent API-migration errors:
- test/application-store.test.ts:155 still calls removed `installScoringSession()`
- test/e2e/fake-flow-main.tsx:191 still supplies removed `RecognitionPageServices.scoringSession`
- test/recognition-page.test.tsx:476 still supplies removed `RecognitionPageServices.scoringSession`
```

The typecheck failures are outside this Task's correction-issue contract surface and arise from the concurrent Application store / RecognitionPage API migration. The A08-focused Application and UI tests compile and pass with the reconciled `CorrectionIssue` vocabulary.

Source inspection also confirms `validateCorrectionDraft()` still calls `ScoringService.validateWinningStructure()` and only maps its product-semantic result; no winning-hand solver was introduced into Application.

## Evidence

- A06 F-MAJ-03 identified that current Application exports `invalid-completed-hand-tile` while `spec:product.system.contracts.correction_editor_api` declared only `completed-hand-count | invalid-meld | not-winning-shape`.
- The Scoring API exposes the product-owned `WinningStructureIssue.kind === 'completed-hand-tile'` outcome; the corrected contract now explicitly maps it to `CorrectionIssue.kind === 'invalid-completed-hand-tile'` targeted at `completed-hand`.
- `createCorrectionEditorService()` continues to delegate whole winning-shape determination to `ScoringService.validateWinningStructure()`; no solver or concrete Agari/WASM error vocabulary was added to Application or UI.
- Focused verification passed: `correction-draft-service.test.ts` + `tile-correction-ui.test.tsx` = 24/24 tests.
- Architecture lint passed for 51 source files.
- Repository-wide typecheck was executed but currently reports only three unrelated concurrent API-migration errors in `application-store.test.ts`, `fake-flow-main.tsx`, and `recognition-page.test.tsx`; none references the A08 contract, implementation, or focused tests.
- F-MAJ-03 is resolved: contract, exported Application union, Scoring mapping, UI exhaustive presentation, and focused tests now use one documented correction issue vocabulary.
