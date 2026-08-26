# Overview: UI pages

- **id**: `spec:product.ui.pages`
- **status**: draft
- **date**: 2026-08-19
- **parent**: `spec:product.ui`

## What this is

Page-level visible contract for the first mjtensu PWA.
Pages own required composition, page-specific actions, and meaningful relative placement.
They do not prescribe one concrete frontend component tree.

## Pages

| page | ref | responsibility |
|---|---|---|
| Top | `spec:product.ui.pages.top` | Entry to scoring and help. |
| Recognition | `spec:product.ui.pages.recognition` | Landscape live camera, fixed semantic regions, and live recognition feedback. |
| Conditions | `spec:product.ui.pages.conditions` | Winning-tile selection and non-image scoring conditions. |
| Result | `spec:product.ui.pages.result` | Recognized hand evidence, yaku/han/fu, payment, point result, and correction actions. |
| Recognition correction | `spec:product.ui.pages.recognition_correction` | Semantic tile and meld correction after a calculated result. |
| Help | `spec:product.ui.pages.help` | Capture and operation instructions. |

## Page composition boundary

Page specs may require statements such as:

- one region appears beside another;
- one element is visually primary or secondary;
- a meld is compact on Result but explicitly grouped on Recognition correction;
- score actions appear below the calculation result;
- a modal is opened from a page action.

Page specs do not own exact CSS dimensions, breakpoints, colors, font sizes, DOM nesting, or implementation-component names unless a value is itself part of a recognition/scoring semantic contract.

## Boundary

| concern | owner |
|---|---|
| Page composition and semantic layout | `spec:product.ui.pages` |
| Reusable visible responsibilities | `spec:product.ui.components` |
| Cross-page transitions | `spec:product.ui.screen_flow` |
| Session state | `spec:product.application.scoring_session` |
