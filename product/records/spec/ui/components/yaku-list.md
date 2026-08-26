# Concept: Yaku list

- **id**: `spec:product.ui.components.yaku_list`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.components`

## What this is

Reusable visible responsibility for showing why the hand received its han or yakuman-class value.

## Content

The list presents every awarded scoring entry needed to explain the result.
For ordinary yaku, derive the Japanese display name from the product yaku identity and show the awarded han supplied by scoring.
Dora contribution may be shown as aggregate indicator dora plus aka dora. The list does not need to distinguish visible, kan, ura, or kan-ura indicator source.

The UI must not parse or display Agari's diagnostic/display strings as the yaku identity and must not recalculate open/closed han adjustments.

The list must not require the user to add han manually to discover the total; total han belongs to the score summary.

For yakuman entries, derive the yaku name from the product yakuman identity and do not invent an individual yakuman multiplier. The final applied yakuman unit count belongs to the score/limit summary.

## Ordering

The scoring result may provide a presentation-ready order.
If implementation chooses an order, it must remain deterministic for the same result and should keep yaku entries readable before aggregate dora contribution.

## Non-goals

- Fu calculation.
- Tile evidence.
- Yaku encyclopedia or rule tutorial.
- Exact typography and animation.

## Boundary

| concern | owner |
|---|---|
| Visible yaku-entry presentation | This concept. |
| Yaku and han meaning | `spec:product.scoring.result`. |
| Result-page placement | `spec:product.ui.pages.result`. |
