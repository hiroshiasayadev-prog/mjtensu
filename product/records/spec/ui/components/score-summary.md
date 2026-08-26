# Concept: Score summary

- **id**: `spec:product.ui.components.score_summary`
- **status**: draft
- **date**: 2026-08-19
- **parent**: `spec:product.ui.components`

## What this is

Reusable result component responsibility for presenting the calculated hand value, its payment structure, and the compact controls needed to inspect or correct score-relevant context.

## Required content

The summary exposes, when applicable:

- final fu;
- total han;
- limit classification;
- final hand point result;
- ron or tsumo payment breakdown;
- dealer/non-dealer status.

The final hand point result is the strongest numeric element.
Fu/han/limit are supporting score explanation rather than stronger than the final point result.

## Payment presentation

### Ron

Show the discarder payment amount.
If this is the same number as the primary final hand point result, the UI may avoid visually duplicating an identical large number while still making the payment meaning clear.

### Tsumo

Show the payer distinction explicitly:

- non-dealer winner: payment from each non-dealer and payment from dealer;
- dealer winner: payment from each opponent.

The payment line appears below or directly adjacent to the primary result, not hidden only inside a modal.

## Dealer / child status action

The payment area includes a compact `親` or `子` status action.
Its displayed state is derived from seat wind.
Activating it opens Conditions focused on seat wind so the user can correct the role when needed.

This action must not directly create an independent dealer boolean.
Changing dealer/non-dealer status requires a semantically valid seat-wind change and subsequent recalculation.

## Fu-detail access

When fu detail applies, the summary exposes a `符の詳細` action associated with the fu value.
Activating it opens `spec:product.ui.components.fu_detail_dialog` without changing the scoring session.

## Excluded settlement

The summary does not add:

- honba;
- riichi-stick pool;
- other-player score movement;
- post-hand table scores.

## Non-goals

- Yaku list content.
- Fu-detail line items in the default collapsed result.
- House-rule editing.
- Concrete button or card component implementation.

## Boundary

| concern | owner |
|---|---|
| Visible score/payment summary | This concept. |
| Score meaning | `spec:product.scoring.result`. |
| Page placement | `spec:product.ui.pages.result`. |
| Seat-wind correction/recalculation | `spec:product.application.scoring_session`. |
| Fu-detail modal | `spec:product.ui.components.fu_detail_dialog`. |
