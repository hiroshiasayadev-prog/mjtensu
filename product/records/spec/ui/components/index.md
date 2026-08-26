# Overview: UI components

- **id**: `spec:product.ui.components`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui`

## What this is

Catalog of reusable visible responsibilities in the mjtensu scoring UI.
These are product-level UI concepts, not a requirement for one framework component or source file per concept.

## Current component concepts

| concept | ref | visible responsibility |
|---|---|---|
| Tile presentation | `spec:product.ui.components.tile_presentation` | Render ordered hand tiles, winning-tile state, dora rows, and meld grouping in page-appropriate modes. |
| Condition controls | `spec:product.ui.components.condition_controls` | Present winning-context controls and prevent contradictory combinations. |
| Tile correction editor | `spec:product.ui.components.tile_correction_editor` | Edit hand/meld/dora tile structure with local add/replace/delete controls, repair feedback, and commit gating. |
| Yaku list | `spec:product.ui.components.yaku_list` | Present awarded yaku and han/dora contributions. |
| Score summary | `spec:product.ui.components.score_summary` | Present fu/han/limit, final points, payer breakdown, and dealer/child shortcut. |
| Fu detail dialog | `spec:product.ui.components.fu_detail_dialog` | Present itemized fu calculation on demand. |

## Composition guidance

```text
Conditions page
  +-- Tile presentation (selection mode)
  +-- Tile correction editor
  +-- Condition controls

Result page
  +-- Tile presentation (compact result mode)
  +-- Yaku list
  +-- Score summary
  +-- Fu detail dialog

Recognition correction page
  +-- Tile presentation (editing mode)
  +-- Tile correction editor
```

The implementation may merge or split concrete frontend components differently.
The visible responsibilities and page-level semantics remain authoritative.

## Non-goals

- React/Vue/Svelte component names.
- Props, event signatures, hooks, state stores, or source files.
- Exact DOM, CSS, design tokens, or animations.

## Boundary

| concern | owner |
|---|---|
| Reusable visible responsibility | `spec:product.ui.components` |
| Page assembly and semantic placement | `spec:product.ui.pages` |
| Session state | `spec:product.application.scoring_session` |
| Scoring meaning | `spec:product.scoring` |
