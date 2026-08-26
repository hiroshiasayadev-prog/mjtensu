# Overview: Scoring

- **id**: `spec:product.scoring`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product`

## What this is

Owner of the library-independent contract for calculating Japanese riichi mahjong yaku, han, fu, payment, and final score from recognized tiles plus non-image conditions.

## Current contract

| concern | contract |
|---|---|
| Input | Use recognized tile structure, one winning tile, round/seat conditions, win method, and situational conditions. |
| Derived conditions | Derive tile-visible and structure-visible facts where possible instead of asking the user to re-enter them. |
| Output | Return yaku, han, fu, fu detail, limit classification, point total, and ron/tsumo payment detail. |
| Dora | Treat all supplied indicator tiles as one indicator set, while keeping indicator-derived dora and red-five aka-dora contribution distinguishable in the result. |
| Independence | The product contract does not expose concrete scoring-library types. |
| Excluded settlement | Round number, honba, riichi-stick pool, other-player riichi state, and nagashi mangan settlement are outside the current score contract. |

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Scoring input | Contract | `spec:product.scoring.input` | Required tile structure and non-image conditions. |
| Scoring result | Contract | `spec:product.scoring.result` | Yaku, han, fu, limit, payment, and final-point semantics. |

## Non-goals

- Camera recognition behavior.
- Concrete TypeScript scoring-library API.
- UI layout and presentation styling.
- Table settlement including honba and riichi-stick awards.
- Full game-state tracking.

## Boundary

| concern | owner |
|---|---|
| Score input/result semantics | `spec:product.scoring` |
| Recognition output | `spec:product.recognition` |
| Session coordination and recalculation | `spec:product.application` |
| Visible conditions and result presentation | `spec:product.ui` |
| Concrete scoring adapter | `spec:product.system.contracts.agari_adapter`. |
| Production scoring-engine fork semantics | `spec:product.system.contracts.agari_fork`. |
