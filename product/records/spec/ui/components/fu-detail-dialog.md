# Concept: Fu detail dialog

- **id**: `spec:product.ui.components.fu_detail_dialog`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.components`

## What this is

On-demand modal/dialog responsibility for explaining how the displayed final fu value was produced without occupying the default Result layout.

## Required content

When ordinary fu calculation applies, the dialog presents the aggregate fu categories supplied by `spec:product.scoring.result`.
The visible explanation must be able to distinguish:

- base fu;
- closed-ron fu;
- tsumo fu;
- aggregate meld fu;
- aggregate pair fu;
- aggregate wait fu;
- raw total before rounding;
- rounding to the final fu value.

The dialog does not reconstruct which individual meld generated which part of `melds` when the scoring result does not expose that detail.

For Chiitoitsu, show the fixed 25-fu rule directly rather than rendering a misleading ordinary-fu breakdown.

The dialog ends with or otherwise clearly identifies the final fu used by scoring.

## Applicability

If the scored result does not have meaningful ordinary fu detail, Result need not expose this dialog action.
The UI must not invent a fake fu breakdown for a yakuman-class or otherwise non-applicable result.

## Interaction

- Opening the dialog preserves the current Result and scoring session.
- Closing returns to the unchanged Result.
- The dialog contains no recognition or condition editing controls.
- Recalculation closes or replaces any fu detail that belongs to the prior result before presenting the new result's detail.

## Presentation boundary

Exact Japanese wording and typography are implementation-owned.
However, the presentation must make the arithmetic relationship between contributors and final rounded fu understandable.

## Non-goals

- Full scoring-rule tutorial.
- Yaku explanations.
- Editing the scoring input.
- Persisting dialog state across a new result.

## Boundary

| concern | owner |
|---|---|
| Fu-detail presentation | This concept. |
| Fu-detail data | `spec:product.scoring.result`. |
| Result trigger placement | `spec:product.ui.pages.result`. |
| Modal implementation | UI implementation. |
