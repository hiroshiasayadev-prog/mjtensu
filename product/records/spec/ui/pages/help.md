# Concept: Help page

- **id**: `spec:product.ui.pages.help`
- **status**: draft
- **date**: 2026-08-24
- **parent**: `spec:product.ui.pages`

## What this is

Short usage guide for the first scoring flow.
It explains how to place tiles for recognition and what information the user will still need to enter after recognition.

## Required content

Help must explain at least:

- use the PWA in landscape during recognition;
- place every dora indicator that should count for the winning hand in the upper-left dora region; this includes kan-dora and, for a riichi hand, applicable ura-dora / kan-ura indicators; indicators that should not count are not placed there;
- place the completed hand in the lower-left hand region;
- place meld groups in the right square region, stacked as separate horizontal groups;
- keep tiles fully inside their visible capture region;
- recognition runs automatically without a shutter;
- the app moves on when the same recognized tile structure is observed stably;
- placing the actual winning tile at the right edge of the completed-hand row makes the default winning-tile selection correct more often, but the user can change that selection on Conditions;
- non-image conditions such as ron/tsumo, round wind, seat wind, and riichi state are entered after recognition;
- recognition mistakes can be corrected on Conditions before calculation;
- recognition and condition mistakes can also be corrected from Result after calculation.

## Presentation

Help should remain short enough to scan before using the camera.
Detailed ML explanations, model confidence terminology, dataset information, and scoring-rule theory are not required for normal operation.

## Actions

- Return to Top.
- Help does not start camera permission or recognition by itself.

## Non-goals

- Full riichi rules tutorial.
- Yaku encyclopedia.
- Scoring-rule configuration.
- Model troubleshooting console.

## Boundary

| concern | owner |
|---|---|
| User-facing operation guidance | This concept. |
| Exact recognition geometry | `spec:product.recognition.runtime_recognition`. |
| Conditions | `spec:product.ui.pages.conditions`. |
| Screen navigation | `spec:product.ui.screen_flow`. |
