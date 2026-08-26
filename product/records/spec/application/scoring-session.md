# Contract: Scoring session

- **id**: `spec:product.application.scoring_session`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.application`

## What this is

Transient application state and transition contract for one recognition/scoring attempt.
The session coordinates recognition output and scoring without owning either implementation.

## Session state

One active scoring session contains:

| state | meaning |
|---|---|
| Recognized structure | Completed-hand tile instances, supplied dora indicators, and spatially reconstructed meld groups from the latest committed recognition/correction state. The structure may still require semantic correction before it is calculation-ready. |
| Winning-tile selection | Exactly one completed-hand tile instance selected for the active scoring session. |
| Conditions | Current non-image scoring conditions. |
| Latest result | Latest successful scoring result for the exact current recognized structure, winning tile, and conditions. |

A latest result becomes stale immediately when any score-relevant recognized data, winning-tile selection, or condition changes.
A stale result must not remain presented as if it described the edited session.

## Session creation

When runtime recognition commits one stable result:

1. Create a new scoring session from that recognition result.
2. Select the rightmost completed-hand tile instance as the default winning tile.
3. Initialize scoring conditions to the product-defined initial draft: win method defaults to Tsumo, round wind defaults to East, seat wind defaults to East, riichi is `none`, and all boolean situational conditions are off.
4. Do not require the recognized structure to already be a legal winning hand; Conditions owns pre-calculation correction and scoring feedback.
5. Do not calculate until the corrected structure, winning-tile selection, and required scoring conditions form a valid scoring input.
6. Enter the conditions flow.

The rightmost-tile rule is an interaction default only. It is not a claim that recognition inferred the actual winning tile.

The initial condition draft is:

```ts
{
  winMethod: 'tsumo',
  roundWind: 'east',
  seatWind: 'east',
  riichi: 'none',
  ippatsu: false,
  rinshan: false,
  chankan: false,
  haitei: false,
  houtei: false,
  tenhou: false,
  chiihou: false,
}
```

Tsumo, East round, and East seat are convenience defaults chosen to reduce ordinary input taps. These are initial selections only and remain fully user-editable on Conditions. Boolean situational conditions use their natural unselected value `false`.

## Winning-tile identity

- The winning tile must reference one tile instance, not only a tile kind.
- A meld tile cannot be selected as the winning tile.
- Selecting another completed-hand tile changes only the winning-tile selection until calculation occurs.
- If recognition correction retains the selected tile instance, preserve the selection even when that instance's tile identity is corrected.
- If correction deletes the selected instance or moves it out of the completed hand so it no longer remains a valid winning-tile candidate, select the corrected completed-hand rightmost tile as the new default.

## Pre-calculation recognition correction

Conditions may edit the committed recognized structure before the first calculation.
These edits may repair tile identities, missing or extra recognized tiles, dora indicators, meld membership, or unresolved meld semantics.
The current-yaku preview is recalculated from the draft session as edits occur.
No scoring result exists until the corrected session becomes calculation-ready and the user requests calculation.

## Initial calculation

When the user requests calculation from a calculation-ready session:

1. Build `spec:product.scoring.input` from the current session.
2. Calculate through the scoring boundary.
3. On success, install the returned `spec:product.scoring.result` as the latest result.
4. Enter result presentation.

A scoring error must not overwrite a previously valid state with a fabricated result.

## Recognition correction

Recognition correction edits semantic recognized content rather than detector boxes.
The correction flow may change:

- completed-hand tiles;
- dora indicators;
- meld tile identities;
- meld group membership and score-relevant meld semantics.

On correction confirmation:

1. Replace the recognized structure with the corrected structure.
2. Preserve all non-image conditions; any condition that is no longer compatible with the corrected structure is surfaced by scoring preview rather than silently rewritten from image-derived structure.
3. Preserve the winning-tile instance when it still exists; otherwise apply the rightmost completed-hand default.
4. Invalidate the previous result.
5. Preview the corrected session.
6. When preview is `ready`, recalculate immediately and return to Result with the replacement calculation.
7. When preview is `no-yaku`, route to Conditions with the corrected structure and current conditions installed; Conditions shows `役なし` and lets the user add/change condition-derived yaku such as riichi.
8. When preview is `incomplete` or `invalid-input`, route to Conditions with the corrected session so the missing or contradictory scoring input can be repaired.
9. A stale pre-correction Result is never presented as current after the corrected structure has been committed.

Recognition correction itself requires a supported complete winning shape before confirmation, so `invalid-winning-shape` is not an expected post-confirmation preview state. If that state occurs across this boundary, the implementation must not fabricate or restore the stale result.

## Condition correction

On condition correction confirmation:

1. Preserve recognized structure and winning-tile selection.
2. Replace the edited condition values.
3. Validate the condition combination.
4. Invalidate the previous result.
5. Recalculate immediately when valid.
6. Return to Result after successful recalculation.

The result page may provide shortcuts into a specific condition control, but the application must retain one source of truth for that condition.
Dealer status is derived from seat wind and must not become an independent hidden boolean.

## New recognition

Selecting `もう一度判定` from Result:

- discards recognized structure;
- discards winning-tile selection;
- discards conditions;
- discards the latest result;
- starts a fresh live-recognition flow.

No prior session data is implicitly applied to the new recognition.

## Help navigation

Opening Help from Top has no scoring session to preserve.
If Help becomes reachable from an active flow, opening and closing it must preserve the active session unless the user explicitly starts a new recognition or leaves the product flow.

## Persistence boundary

The first product contract does not require a scoring session to survive:

- browser reload;
- PWA process termination;
- tab close and reopen;
- cross-device use.

Durable score history and resume are outside this contract.

## Boundary

| concern | owner |
|---|---|
| Scoring-session state and recalculation | This contract. |
| Committed recognition structure | `spec:product.recognition.runtime_recognition`. |
| Scoring input/result | `spec:product.scoring`. |
| Pages and visible correction interactions | `spec:product.ui`. |
| Concrete frontend state container and route implementation | Implementation / internal design. |
