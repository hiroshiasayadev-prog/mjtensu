# Concept: Tile correction editor

- **id**: `spec:product.ui.components.tile_correction_editor`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.components`

## What this is

Reusable smartphone-oriented editor for correcting one committed recognized tile structure before it is applied back to the scoring session.

The editor operates on a UI-local correction draft. The draft may be temporarily incomplete or structurally invalid while the user is editing. Invalid intermediate state must not be committed as the canonical scoring-session structure.

## Editable regions

The editor presents three clearly separated semantic regions:

1. completed hand;
2. melds;
3. dora indicators.

Each region has an add action at its right edge or equivalent visually local position.

- completed-hand `+` adds one tile instance to the completed-hand draft;
- dora `+` adds one dora indicator;
- meld-region `+` adds a new meld group;
- a meld group may expose its own member-add action when another member is needed while repairing that group.

The UI must make meld-group boundaries unmistakable.

## Tile editing interaction

Tapping an existing tile opens a tile-selection surface suitable for smartphone use, such as a bottom sheet or drawer.

The selection surface must permit:

- replacing the selected tile identity;
- choosing ordinary versus red five where applicable;
- deleting the selected tile instance.

The add action opens the same tile selector in insertion mode. A new tile is added only after the user selects an identity; the canonical draft need not create a placeholder `null` tile merely because the selector was opened.

Exact tile artwork, selector layout, and whether the selector uses tabs/rows/grids are implementation-owned.

## Meld semantic editing

When the current meld member composition uniquely determines the meld class, the editor may derive that class automatically for presentation and validation:

- a valid three-tile same-suit sequence is `chi`;
- three equal base tile kinds are `pon`;
- four equal base tile kinds are a kan candidate.

Kan openness is never treated as trustworthy merely because recognition inferred it previously or because four equal tile identities are present. Every four-tile kan group exposes an explicit visible `明槓` / `暗槓` control. The control always reflects the currently stored semantic state and tapping it flips the group between open-kan and concealed-kan.

A recognition-derived kan state may initialize that control, but the user can always correct it directly. The editor must not hide the distinction behind an automatic inference that cannot be verified from tile identities alone.

For chi and pon, no redundant `チー` / `ポン` selector is required when the current member composition uniquely determines the legal meld type. A malformed member composition remains an invalid meld group until repaired.

## Permissive editing state

The correction draft may temporarily contain states that cannot be committed, including:

- the wrong number of completed-hand tiles for the current meld count;
- an incomplete meld group;
- a meld group whose current member identities do not form a supported chi, pon, or kan;
- an unresolved meld semantic state;
- a total tile structure that is not yet a complete supported winning hand.

The user is allowed to pass through such states while repairing recognition output.

## Local validation feedback

Validation feedback is shown at the smallest useful semantic region.

Examples:

- a malformed meld group receives an error outline around that meld group;
- an inconsistent completed-hand count receives an error outline around the completed-hand region;
- a completed-hand tile rejected by scoring structure validation receives repair feedback on the completed-hand region;
- a whole-structure winning-shape failure is shown against the hand/meld structure rather than against unrelated dora or condition controls.

The UI accompanies the visual error state with a short repair-oriented message.
A color change alone must not be the only indication of invalid state.

## Commit eligibility

The correction action (`次へ`, `修正を確定`, or the context-equivalent primary action) is disabled while the correction draft cannot be converted into a supported complete winning structure.

Commit gating consumes `ScoringService.validateWinningStructure()` from `spec:product.system.contracts.scoring_api`. UI may render the returned product-semantic issue locations, but it must not implement its own mahjong winning-hand solver.

Commit eligibility requires at least:

- every meld group has valid score-relevant semantics and valid member composition;
- completed-hand tile count is consistent with the number of logical meld groups;
- no unresolved meld group remains;
- the complete hand/meld tile structure forms a supported winning shape.

For an ordinary four-meld hand, the completed-hand count follows the logical meld count:

```text
0 melds -> 14 completed-hand tiles
1 meld  -> 11 completed-hand tiles
2 melds ->  8 completed-hand tiles
3 melds ->  5 completed-hand tiles
4 melds ->  2 completed-hand tiles
```

A kan still consumes one logical meld slot, so it does not change this completed-hand count rule.
Special supported closed winning shapes such as seven pairs or thirteen orphans remain zero-meld 14-tile structures and are determined by the scoring/shape-validation boundary rather than by UI-specific hand solvers.

## What does not block correction commit

Correction is a tile-structure editor, not the scoring-condition form.
The following do **not** make an otherwise valid correction draft uncommittable:

- no currently established yaku;
- riichi not yet selected;
- Ron/Tsumo not yet selected;
- round wind not yet selected;
- seat wind not yet selected;
- other Conditions-page situational inputs not yet supplied.

A structurally complete winning hand may therefore proceed to Conditions even when its eventual scoring yaku depends on a later condition such as riichi.

## Reordering

The editor must support correcting completed-hand order and other order-sensitive presentation where required.
The exact interaction (drag, move-left/right controls, or another mobile interaction) is implementation-owned.

## Boundary

| concern | owner |
|---|---|
| Visible add/replace/delete/reorder interaction and local invalid-state feedback | This concept. |
| Meld semantic interpretation/validation | System/application/scoring correction validation boundary. |
| Winning-shape determination | `ScoringService.validateWinningStructure()` in `spec:product.system.contracts.scoring_api`; UI must not reimplement a mahjong hand solver. |
| Canonical session replacement | `spec:product.system.contracts.application_session_api`. |
| Winning-tile selection and scoring conditions | Conditions page / scoring session. |
| Detector boxes/model annotations | Outside this editor. |
