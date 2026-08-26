# Concept: Result page

- **id**: `spec:product.ui.pages.result`
- **status**: draft
- **date**: 2026-08-24
- **parent**: `spec:product.ui.pages`

## What this is

Primary scoring-result surface.
It presents the recognized hand as evidence, then emphasizes awarded yaku and the final point result while keeping fu detail and correction paths available without crowding the main result.

## Information hierarchy

The page follows this semantic priority:

```text
recognized hand / dora evidence
        ↓
yaku and han
        ↓
fu + total han + limit classification
        ↓
final point result
        ↓
payment breakdown
        ↓
correction / restart actions
```

The visual treatment may take inspiration from mahjong-game result screens, but this contract owns only information hierarchy and meaningful relative placement.

## Recognized-hand presentation

The upper result area displays:

- completed-hand tiles in order;
- the winning tile distinctly identified;
- melds on the same overall hand line or adjacent hand area in a smaller, compact presentation;
- separate meld groups divided by spacing or another lightweight grouping cue;
- no required `ポン`, `チー`, or equivalent text labels when the tile grouping itself is sufficient;
- supplied dora indicators.

Result melds are intentionally compact. The explicit editing/group-management presentation belongs to Recognition correction rather than Result.

## Yaku and han presentation

The page displays each awarded yaku with its awarded han where han applies.
Dora-related contribution may appear in the same list as aggregate indicator dora and aka dora. The UI does not need to distinguish visible, kan, ura, or kan-ura indicator source.

The page must make total han available without forcing the user to reconstruct it from the list.

## Fu and limit summary

When ordinary fu is meaningful, the main result displays the final fu value next to or near total han.
A `符の詳細` action opens the fu-detail dialog defined by `spec:product.ui.components.fu_detail_dialog`.

When the result is a limit/yakuman-class result where ordinary fu detail is not useful, the page may prioritize the limit classification and omit an inapplicable fu-detail action.

The limit classification, when present, is shown with the score summary rather than hidden inside the detailed explanation.

## Final score and payment

The final point result is the strongest numeric element on the page.
Below it, or immediately adjacent in the same score block, the page shows the payment detail from `spec:product.scoring.result`.

For tsumo, payment detail must expose the payer distinction rather than only one aggregate number:

```text
non-dealer winner: child payment / dealer payment
 dealer winner:    each opponent payment
```

For ron, payment detail may show the single discarder payment.

The score block also shows current dealer/non-dealer status as a compact `親` / `子` action adjacent to the payment area.
This action is a shortcut to condition correction for seat wind; it must not maintain an independent dealer boolean or silently override seat wind.

The page does not add honba or riichi-stick pool settlement to the displayed hand score.

## Dora evidence

The supplied dora indicators are displayed on Result even though the calculated yaku list already reports dora contribution.
The display does not distinguish visible, kan, ura, or kan-ura source; it shows the indicator set the user supplied for this winning hand.
This display is recognition evidence, not an additional scoring input created by the page.

## Result actions

The lower action area exposes:

| action | behavior |
|---|---|
| `認識結果を修正` | Open Recognition correction while preserving conditions. |
| `条件を修正` | Open Conditions with current conditions pre-populated. |
| `もう一度判定` | Discard the current scoring session and begin live Recognition. |

The two correction actions are secondary to the result itself.
`もう一度判定` is the clear action for beginning a new hand rather than mutating the current result.

## Recalculation replacement

After a successful recognition or condition correction, Result must replace every score-dependent value atomically from the new calculation:

- hand evidence where edited;
- winning-tile marker where changed;
- yaku list;
- han;
- fu and fu detail availability;
- limit classification;
- dealer/child status;
- payment detail;
- final point result;
- dora-indicator evidence where edited.

A previous score must not remain visible as the current score after input changes invalidate it.

## Non-goals

- Kyoku number display.
- Riichi-stick pool count.
- Honba count.
- Player avatar, rank, table score, or opponent information.
- `ポン` / `チー` textual labels for melds.
- Inline full fu calculation occupying the default result layout.
- Pixel-perfect reproduction of any existing mahjong game's result screen.

## Boundary

| concern | owner |
|---|---|
| Result-page information hierarchy and actions | This concept. |
| Semantic score result | `spec:product.scoring.result`. |
| Compact tile/meld presentation | `spec:product.ui.components.tile_presentation`. |
| Score block | `spec:product.ui.components.score_summary`. |
| Fu detail | `spec:product.ui.components.fu_detail_dialog`. |
| Correction/recalculation state | `spec:product.application.scoring_session`. |
