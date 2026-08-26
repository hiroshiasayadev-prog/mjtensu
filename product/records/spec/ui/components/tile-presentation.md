# Concept: Tile presentation

- **id**: `spec:product.ui.components.tile_presentation`
- **status**: draft
- **date**: 2026-08-24
- **parent**: `spec:product.ui.components`

## What this is

Reusable visible responsibility for presenting recognized/corrected tile structure without exposing recognition-model details.
The same semantic tile data is presented differently depending on page purpose.

## Shared tile semantics

The presentation must preserve:

- canonical tile identity;
- ordinary-five versus red-five distinction;
- completed-hand order;
- winning-tile instance identity when selected;
- dora-indicator order;
- meld group boundaries;
- score-relevant meld semantics when the page needs them.

Tile instances with the same tile kind must remain distinguishable when interaction targets one instance.

## Presentation modes

### Conditions selection and correction mode

- Completed hand is prominent and horizontally ordered.
- Exactly one valid completed-hand tile can be selected as the winning tile.
- Selected winning tile is visibly distinct.
- Meld and dora evidence may be smaller/secondary.
- Individual tile instances and meld grouping/semantics can be targeted for pre-calculation semantic correction.
- Meld tiles cannot enter winning-tile selection.
- Editing changes semantic tile structure only; detector boxes are not edited.

### Result compact mode

- Completed hand and winning tile remain the primary tile evidence.
- Meld groups are displayed beside or adjacent to the hand in a smaller compact form.
- Separate meld groups retain visual spacing/grouping.
- No `ポン` / `チー` text label is required.
- Supplied dora indicators remain visible as supporting evidence.

### Recognition-correction editing mode

- Completed hand, dora indicators, and melds are clearly separated editable regions.
- Meld groups are rendered as independent group columns/containers rather than compressed into the Result layout.
- The user can target individual tile instances and group boundaries for semantic correction.
- Editing must not require detector-box manipulation.

## Winning-tile marker

The marker must identify one tile instance, not merely one tile kind.
The concrete marker may use offset, outline, badge, spacing, or another visual cue as long as the selected instance remains unambiguous.

## Meld grouping

Meld grouping is semantic, not decorative.
A page must never render multiple meld groups as one visually indistinguishable continuous group when that would make the recognized/corrected structure ambiguous.

Result may use small spacing because the purpose is verification.
Recognition correction must use stronger separation because the purpose is editing.

## Non-goals

- Exact tile artwork source.
- CSS dimensions and colors.
- Recognition bounding boxes and confidence.
- Concrete tile-picker implementation.

## Boundary

| concern | owner |
|---|---|
| Reusable tile presentation semantics | This concept. |
| Page-specific placement | `spec:product.ui.pages`. |
| Recognition structure | `spec:product.recognition.runtime_recognition`. |
| Winning-tile/session state | `spec:product.application.scoring_session`. |
