# Concept: Debug page

- **id**: `spec:product.ui.pages.debug`
- **status**: draft
- **date**: 2026-08-29
- **parent**: `spec:product.ui.pages`

## What this is

Internal recognition-diagnostics surface separated from the production Recognition UI.
It reuses the production camera and recognition runtime so performance observations represent the same inference path without exposing diagnostic controls in the ordinary scoring flow.

## Entry and presentation

- Top exposes a small, low-prominence `debug` link at the bottom-right.
- Debug uses only the portrait-locked capture presentation. It does not switch to the ordinary landscape viewport presentation when the browser viewport becomes landscape.
- The camera preview and fixed semantic recognition regions remain the same recognition input geometry used by production Recognition.
- The Debug route does not create or replace a scoring session.

## Runtime behavior

Camera and recognition-runtime preparation follow the same parallel startup and owner-specific recovery behavior as production Recognition.
Realtime recognition continues for diagnostics instead of transitioning to Conditions after a stable structure is confirmed. After a confirmation, stabilization is reset and evaluation continues.

The latest completed evaluation timing is shown continuously using these rows:

```text
detector preprocessing: xxx ms
detector inference:      xxx ms
postprocess:             xxx ms
crop extraction:         xxx ms
base preprocessing:      xxx ms
base inference:          xxx ms
red5 preprocessing:      xxx ms
red5 inference:          xxx ms
total:                   xxx ms
```

Before the first completed evaluation, timing values may be shown as unavailable placeholders.
Values are diagnostics only and must not alter recognition behavior.

## Actions

- `終了` returns to Top and tears down the page-owned camera/realtime run in the same way as production Recognition abandonment.
- The timing block is placed below `終了` in the portrait Debug controls.
- Diagnostic JSON capture/save is available on Debug when supported by the recognition runtime.
- Diagnostic JSON capture/save and continuous timing telemetry are not exposed on the production Recognition surface.

## Non-goals

- Scoring-session creation.
- Automatic transition to Conditions.
- User-facing performance promises or pass/fail thresholds.
- Separate debug-only model or inference implementation.

## Boundary

| concern | owner |
|---|---|
| Debug-page composition and diagnostic presentation | This concept. |
| Recognition timing collection | `spec:product.recognition.pipeline`. |
| Camera/runtime contracts | `spec:product.system.contracts.camera_api`, `spec:product.system.contracts.model_runtime`. |
| Production Recognition composition | `spec:product.ui.pages.recognition`. |
