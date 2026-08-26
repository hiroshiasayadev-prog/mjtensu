# Concept: Recognition correction page

- **id**: `spec:product.ui.pages.recognition_correction`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.pages`

## What this is

Semantic correction surface for a committed recognition result after score calculation.
It edits the recognized tile structure, not detector boxes or model annotations.

## Required composition

The page presents the recognized structure in editable semantic groups:

1. completed hand;
2. dora indicators;
3. meld groups.

The completed hand remains an ordered horizontal tile sequence.
Dora indicators remain one ordered row; visible, kan, ura, and kan-ura source is not edited separately.
Melds are shown as clearly separated group columns/containers rather than as one compact result-line presentation.

For melds, each meld receives its own visible group column/container. The page must make group boundaries unmistakable so the user can tell which tiles belong to each meld.
No `ポン` / `チー` text label is required when the corrected tile composition and grouping communicate the meld sufficiently, but score-relevant kan/open/closed semantics must remain editable or otherwise unambiguous when they cannot be derived from tile composition alone.

## Editing capabilities

The correction surface must support semantic operations sufficient to repair ordinary recognition errors:

- replace one tile instance with another tile identity;
- distinguish ordinary five from red five;
- add a missing tile where the recognized structure permits repair;
- remove an extra/false recognized tile;
- reorder tiles when recognition order is wrong;
- add or remove dora indicators;
- move or regroup meld tiles between meld groups;
- add or remove a meld group;
- correct score-relevant meld semantics where required.

The concrete interaction follows `spec:product.ui.components.tile_correction_editor`: each semantic region has a local add action, tapping an existing tile opens the tile selector with replace/delete actions, and incomplete editing state is permitted until commit.

## Winning-tile interaction

Recognition correction is not the primary winning-tile selector.
Application preserves the current winning-tile instance when it remains present after edits.
If the selected instance is removed or moved out of the completed hand, Application applies the rightmost completed-hand default when possible. Replacing only that instance's tile identity preserves the same `TileInstanceId` and therefore preserves the winning-tile selection.
The corrected winning-tile state must be visible before or after recalculation so the user is not left with an invisible changed assumption.

## Confirm and cancel

| action | behavior |
|---|---|
| Confirm correction | Enabled only when the correction draft is a supported complete winning structure. Apply the corrected structure and preserve current conditions. If scoring preview is ready, recalculate and return to Result; if the corrected session has no yaku or otherwise needs condition repair, continue to Conditions instead. |
| Cancel | Discard correction draft and return to the unchanged Result. |

The page must not show a stale Result as if correction had already been applied.

While editing, temporarily malformed tile counts, incomplete/invalid meld groups, unresolved meld semantics, and non-winning whole-hand shapes are permitted. The affected semantic region/group is visibly marked with repair-oriented feedback and Confirm remains disabled.

A complete supported winning shape is required before correction can be committed. Having no yaku under the currently supplied conditions does not make the tile correction invalid; after confirmation, a `no-yaku` corrected session continues to Conditions with `役なし` feedback so condition-derived yaku may be supplied or changed.

Once a correction is confirmed and installed into the scoring session, the pre-correction Result is stale. A route to Conditions for further repair must not offer that stale Result as a cancellable/current score.

## Non-goals

- Bounding-box correction.
- Confidence-threshold tuning.
- Model retraining feedback.
- Camera recapture inside this page.
- Condition editing other than recognition-derived structure.

## Boundary

| concern | owner |
|---|---|
| Visible semantic correction interaction | This concept. |
| Correction state application/recalculation | `spec:product.application.scoring_session`. |
| Tile/meld editing presentation | `spec:product.ui.components.tile_presentation` and `spec:product.ui.components.tile_correction_editor`. |
| Recognition semantic shape | `spec:product.recognition.runtime_recognition`. |
