# Concept: Top page

- **id**: `spec:product.ui.pages.top`
- **status**: draft
- **date**: 2026-08-29
- **parent**: `spec:product.ui.pages`

## What this is

Minimal entry page for starting a new score recognition and accessing usage help, with a deliberately low-prominence entry to internal recognition diagnostics.

## Required composition

The page contains:

- product identity/title;
- one dominant `判定する` action with camera affordance;
- one lower-prominence `使い方` action;
- one very small `debug` link fixed at the bottom-right for internal diagnostics access.

The scoring action must be visually primary over help.
The page does not require history, account, saved-score, settings, or mode-selection surfaces in the current product contract.

## Actions

| action | result |
|---|---|
| `判定する` | Enter Recognition and request camera access as needed. |
| `使い方` | Enter Help. |
| `debug` | Enter the internal Debug recognition-diagnostics surface. |

Camera permission should be requested as part of entering the recognition interaction rather than unexpectedly on initial Top-page display.

## Non-goals

- Score history.
- Account/login.
- House-rule settings.
- Diagnostic content on Top itself beyond the low-prominence Debug entry link.
- Exact branding and visual styling.

## Boundary

| concern | owner |
|---|---|
| Top-page composition | This concept. |
| Transition destinations | `spec:product.ui.screen_flow`. |
| Camera recognition | `spec:product.ui.pages.recognition`. |
| Help content | `spec:product.ui.pages.help`. |
