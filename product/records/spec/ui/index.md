# Overview: UI

- **id**: `spec:product.ui`
- **status**: draft
- **date**: 2026-08-19
- **parent**: `spec:product`

## What this is

Owner of the PWA's visible screen flow, page composition, semantic layout, and reusable visible responsibilities for one scoring session.

## Current contract

| concern | contract |
|---|---|
| Primary flow | Top -> Recognition -> Conditions -> Result. |
| Recognition correction | Result can open recognition correction and return through recalculation. |
| Condition correction | Result can reopen conditions and return through recalculation. |
| Restart | Result can discard the session and begin recognition again. |
| Help | Top exposes a lower-prominence help entry. |
| Recognition interaction | Live camera recognition is shutterless and auto-confirms after recognition stability succeeds. |
| Layout responsibility | Page specs own required visible composition and meaningful relative placement, not pixel-perfect CSS. |
| Component responsibility | Component specs own reusable visible meaning and actions, not React/Vue component files or props. |

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Screen flow | Contract | `spec:product.ui.screen_flow` | Main navigation and correction/restart transitions. |
| Pages | Overview | `spec:product.ui.pages` | Page-level composition and semantic layout. |
| Components | Overview | `spec:product.ui.components` | Reusable visible responsibilities. |

## Placement rules

- Put page-specific required content and meaningful relative placement under `pages/`.
- Put reusable visible concepts used by multiple pages under `components/`.
- Put state transition and navigation rules under `screen-flow.md` unless they are application session semantics.
- Do not freeze concrete framework component trees, props, hooks, route files, CSS values, or DOM structure in product UI specs.

## Non-goals

- Detector/classifier implementation.
- Scoring formulas and concrete scoring-library types.
- Pixel-perfect styling and brand system.
- Frontend framework, state library, router, file names, and props.

## Boundary

| concern | owner |
|---|---|
| PWA-visible behavior and semantic layout | `spec:product.ui` |
| Session state and recalculation | `spec:product.application.scoring_session` |
| Recognition semantics | `spec:product.recognition` |
| Scoring semantics | `spec:product.scoring` |
| Concrete UI implementation | Implementation / internal design. |
