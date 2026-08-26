# Concept: Conditions page

- **id**: `spec:product.ui.pages.conditions`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.pages`

## What this is

Post-recognition surface for selecting the winning tile and entering only the scoring conditions that cannot be determined from recognized tiles alone.
The page also gives the user a lightweight visual check and correction surface for the recognized hand before calculation.

## Required composition

The page is organized in this semantic order:

1. recognized tile overview;
2. winning-tile selection within the completed hand;
3. ordinary scoring conditions;
4. less-common situational conditions;
5. calculation action.

### Recognized tile overview

The completed hand must be displayed prominently as tile faces in recognition order.
The selected winning tile must be visually distinguishable from the other completed-hand tiles.
Melds and recognized dora indicators may be shown more compactly as supporting recognition evidence.

The recognized structure is editable before calculation. The page uses the shared `spec:product.ui.components.tile_correction_editor` interaction for mistaken or missing tile identities, dora edits, and meld repair without requiring the user to return to the live camera page.

While correction is in progress, temporarily incomplete or malformed tile structure is allowed. Affected hand/meld regions are marked with local repair feedback and the page cannot proceed to scoring-condition completion/calculation until the tile structure is again a supported complete winning shape. Lack of yaku by itself does not invalidate tile correction because condition-derived yaku such as riichi may be supplied on this page.

Meld tiles are never selectable as the winning tile.

### Winning-tile selection

- The initial selection is the rightmost completed-hand tile provided by Application.
- Tapping another completed-hand tile selects that tile instance as the winning tile.
- Duplicate tile kinds must remain independently selectable by instance.
- The UI label should describe this as the winning tile rather than assuming only tsumo; the same selection is required for ron scoring.

Placing the actual winning tile at the right edge of the completed-hand capture is therefore a convenience convention, not a hard recognition requirement.

## Ordinary conditions

The page exposes the commonly needed controls directly:

| control | values |
|---|---|
| Win method | Ron / Tsumo. |
| Round wind | East / South / West / North. |
| Seat wind | East / South / West / North. |
| Riichi | None / Riichi / Double riichi. |
| Ippatsu | On / off when semantically available. |

Dealer status is derived from seat wind East. The page must not maintain an independent dealer flag that can contradict seat wind.

For a newly created scoring session, Win method defaults to Tsumo, Round wind defaults to East, and Seat wind defaults to East. Riichi begins at `None`, and Ippatsu plus all additional situational conditions begin off. These select-style defaults reduce ordinary input taps but remain fully user-editable before calculation.

## Additional situational conditions

Less-common conditions may be grouped behind an `その他の条件` disclosure or equivalent secondary surface:

- Rinshan kaihou;
- Chankan;
- Haitei;
- Houtei;
- Tenhou;
- Chiihou.

The page keeps contradictory dependent selections out of ordinary interaction by using `spec:product.system.contracts.scoring_condition_policy`.
When an ordinary condition change makes a selected dependent condition impossible, that dependent value is cleared immediately and the corresponding unavailable control is shown off and disabled/unselectable (or hidden when clearer for a secondary condition).
The scoring boundary still rejects contradictory input defensively if one reaches it from outside the ordinary UI flow.

Nagashi mangan is outside the current product flow and is not offered as a scoring condition.

## Derived information not requested from the user

The page must not ask the user to re-enter:

- whether the hand is closed when this follows from recognized meld structure;
- menzen tsumo;
- ordinary tile-composition yaku;
- red-five count;
- dora count from the supplied indicators;
- dealer status separately from seat wind.

## Current-yaku feedback

The page provides a compact current-yaku preview using the recognized/corrected structure, selected winning tile, and scoring conditions currently available.
The preview exists primarily as semantic feedback: if a yaku the user expects is absent, the user can inspect the recognized tiles, meld grouping, winning-tile selection, or conditions before calculating.

- The preview updates when recognized tile content, meld structure, winning-tile selection, or scoring conditions change.
- When enough input exists to identify awarded yaku, list the currently awarded yaku compactly below the main input area.
- When the current tiles do not form a supported winning shape, indicate that the current hand is not a valid winning shape rather than treating recognition as failed.
- When the hand is a winning shape but has no yaku, indicate `役なし` or equivalent.
- The preview is not recognition acceptance and does not send the user back to the camera page automatically.

## Calculation action

The calculation action becomes available only when:

- the recognized/corrected structure can form a supported scoring input;
- a valid winning tile is selected;
- all required ordinary conditions exist;
- the selected condition combination is internally consistent;
- the hand has at least one scoring yaku under the active rule profile.

On initial calculation, success transitions to Result.
When this page was opened from Result for condition correction, successful recalculation returns to Result while preserving recognized structure.

## Correction-mode behavior

When opened directly from Result for condition correction:

- existing values are pre-populated;
- cancelling returns to the unchanged Result because no correction has yet been committed;
- confirming valid edits recalculates rather than starting a new scoring session.

When Conditions is entered after a **confirmed recognition correction** because the corrected session is `no-yaku`, `incomplete`, or `invalid-input`, the corrected structure has already replaced the old structure and the prior Result is stale. In that path the page is a repair continuation, not a cancellable view of the old Result; it must not navigate back to or restore the stale pre-correction score as current.

A Result shortcut targeting dealer/child status may open this page with the seat-wind control emphasized. Dealer status itself remains derived from seat wind.

## Non-goals

- Pixel-level detector-box correction.
- Full game-state entry.
- Kyoku number, honba, riichi-stick pool, or other-player riichi input.
- Nagashi mangan settlement.
- House-rule configuration.

## Boundary

| concern | owner |
|---|---|
| Visible condition controls, pre-calculation recognition correction, winning-tile selection, and current-yaku feedback | This concept. |
| Semantic score input | `spec:product.scoring.input`. |
| Default/preserved values and recalculation | `spec:product.application.scoring_session`. |
| Reusable controls | `spec:product.ui.components.condition_controls` and `spec:product.ui.components.tile_correction_editor`. |
