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
- Debug uses the same presentation as production Recognition while the smartphone viewport is portrait-locked: a `9:16` camera surface with the existing logical-landscape `16:9` recognition UI quarter-turned over it.
- Debug does not introduce a separate portrait-native recognition UI and does not provide a second landscape-layout variant.
- The camera preview and fixed semantic recognition regions remain the same recognition input geometry used by production Recognition.
- The Debug route does not create or replace a scoring session.

## Runtime behavior

Camera and recognition-runtime preparation follow the same parallel startup and owner-specific recovery behavior as production Recognition.
Realtime recognition continues for diagnostics instead of transitioning to Conditions after a stable structure is confirmed. After a confirmation, stabilization is reset and evaluation continues.

The latest completed evaluation timing is shown continuously for these nine values: detector preprocessing, detector inference, postprocess, crop extraction, base preprocessing, base inference, red5 preprocessing, red5 inference, and total. On the Debug surface they are arranged compactly across the upper unused strip, normally as a three-column by three-row grid rather than a single vertical list.

Before the first completed evaluation, timing values may be shown as unavailable placeholders.
Values are diagnostics only and must not alter recognition behavior.

## Actions

- `終了` returns to Top and tears down the page-owned camera/realtime run in the same way as production Recognition abandonment.
- The timing telemetry uses the long unused strip above the recognition regions on the logical-landscape surface. The nine values may wrap into two or three horizontal rows so they remain compact and do not overlap the recognition regions.
- `終了` and the diagnostic JSON action are placed together in the unused space directly above the meld region rather than being forced into the timing strip.
- The timing telemetry and both actions live on the same logical-landscape surface as the existing portrait-locked production UI, so they rotate together with the recognition overlay.
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
