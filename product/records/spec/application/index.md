# Overview: Application

- **id**: `spec:product.application`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product`

## What this is

Owner of one transient scoring session from a committed recognition result through condition entry, calculation, correction, recalculation, and explicit restart.

## Current contract

| concern | contract |
|---|---|
| Session start | A newly committed recognition result starts one scoring session. |
| Recognized state | Preserve the committed completed hand, one supplied dora-indicator row, and meld groups until correction or restart. |
| Winning tile | Initialize the winning tile to the rightmost completed-hand tile. Preserve the selected tile instance while it remains valid. |
| Conditions | Preserve non-image scoring conditions across result viewing and recognition correction. |
| Calculation | Convert the current recognized structure plus conditions into the scoring contract and retain the latest successful result. |
| Recognition correction | Replace the corrected recognition structure, preserve current conditions, preserve or restore a valid winning-tile selection, and either recalculate or continue to Conditions when scoring input/yaku needs repair. |
| Condition correction | Preserve recognition state, update conditions, and recalculate. |
| New recognition | Discard the current scoring session and return to live recognition. |
| Help | Help navigation does not implicitly destroy an active scoring session. |

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Scoring session | Contract | `spec:product.application.scoring_session` | State ownership, defaults, correction, recalculation, and disposal. |

## Non-goals

- ML inference and camera rendering.
- Scoring formulas and yaku semantics.
- Exact page layout or component implementation.
- Persistent history, accounts, cloud synchronization, or cross-device resume.

## Boundary

| concern | owner |
|---|---|
| One transient scoring session | `spec:product.application` |
| Recognition semantics | `spec:product.recognition` |
| Score input/result semantics | `spec:product.scoring` |
| Screen navigation and visible controls | `spec:product.ui` |
| Concrete frontend store/hooks/routes | Implementation / internal design. |
