# Concept: Recognition page

- **id**: `spec:product.ui.pages.recognition`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.pages`

## What this is

Landscape live-camera surface that guides the user to place score-relevant tiles inside the fixed semantic regions and leaves automatically when recognition becomes stable.

## Required composition

The camera preview is the primary surface.
The page overlays the fixed recognition regions from `spec:product.recognition.runtime_recognition`:

```text
+--------------------------------------+
|  +-----------------+  +-----------+ |
|  |      ドラ       |  |           | |
|  +-----------------+  |           | |
|                       |   副露    | |
|  +-----------------+  |           | |
|  |      手牌       |  |           | |
|  +-----------------+  +-----------+ |
+--------------------------------------+
```

The semantic arrangement is fixed:

- dora region at upper-left;
- completed-hand region at lower-left;
- square meld region at the right;
- dora and completed-hand regions preserve `17:4` aspect ratio;
- meld region preserves `1:1` aspect ratio.

The visible frames must correspond to the actual recognition input boundary. A tile shown fully inside a frame must not be silently cut by an additional hidden crop.
Camera content outside the three recognition regions must be visually dimmed or masked so that the user can see which parts of the preview are ignored by recognition. The recognition regions themselves remain visually clear.

## Startup and runtime behavior

On Recognition entry, camera startup and recognition-runtime initialization begin in parallel:

```text
Recognition entry
  +-- CameraService.open()
  +-- RecognitionRuntime.initialize()
          |
          +-- await/deduplicate any app-lifetime model-asset prefetch

camera ready + runtime ready
          |
          v
start realtime recognition
```

The camera preview becomes available as soon as the camera session is usable even when model acquisition/session initialization is still in progress. While the runtime is not yet ready, the page shows a compact preparation state such as `認識モデルを準備しています` over the camera surface and does not start realtime inference.

The user may therefore position tiles while the recognition models finish preparing. Entering Recognition must not wait on model readiness before opening the camera.

Startup presentation follows the independently progressing camera/runtime work:

| camera | recognition runtime | presentation |
|---|---|---|
| preparing | preparing | Show the Recognition surface in preparation state. |
| ready | preparing | Show the live camera preview and `認識モデルを準備しています` or equivalent. Realtime inference has not started yet. |
| preparing | ready | Show `カメラを起動しています` or equivalent until a usable camera session exists. |
| ready | ready | Start realtime recognition. |

A numeric model-download percentage is not required. The initial product may expose only coarse preparation state rather than byte-level download progress.

### Startup and runtime failure recovery

Camera and model/runtime failures are presented according to the failing owner rather than collapsing the whole page into one generic error.

- If camera startup fails, replace the unavailable camera surface with a camera-specific recovery message and `再試行` / `トップへ` actions.
- Camera retry calls the camera-open path again. An already-ready recognition runtime is retained and must not be recreated merely because camera startup failed.
- If model/runtime preparation fails after the camera is already ready, keep the live camera preview visible and overlay a model-preparation failure message with `再試行` / `トップへ` actions.
- Model/runtime retry retries recognition-runtime initialization/resolution without closing and reopening an otherwise healthy camera session.
- If both sides fail independently, each owned resource may be retried without requiring the other healthy side to be torn down first.
- A fatal inference failure after recognition has started uses the same recognition-runtime recovery surface: stop the current realtime run, preserve a healthy camera session where possible, and allow runtime retry or return to Top.

The UI branches on the normalized camera/recognition error categories from `spec:product.system.contracts.runtime_errors`; browser exception names, HTTP status codes, model filenames, and execution-provider details are not user-facing recovery inputs.

- The page requires landscape orientation for active recognition.
- Recognition runs continuously without a shutter button.
- The page overlays current tile detections and region-local feedback from `spec:product.recognition.pipeline`.
- Each retained detector candidate is shown with its bounding box; when a tile identity is available, a small recognized-tile icon is shown over or adjacent to that box.
- Live detector boxes are feedback, not a user-editable annotation surface.
- The page remains active while the current recognition structure has not yet stabilized.
- When recognition commits the same recognized structure for the required three consecutive evaluations, the page transitions automatically to Conditions.
- Winning-shape legality and yaku existence do not keep the user on the camera page; those are handled after recognition.
- The user is not required to press `OK` after stabilization.

## Feedback

The primary realtime feedback is visual recognition overlay on the camera preview.

- For each current retained detector candidate, the page associates the detector box with the tile identity currently produced by recognition, normally by rendering a small tile icon over or adjacent to the box.
- A candidate that has not produced a supported tile identity may remain as a neutral box or unresolved marker; the UI does not need to guess whether the underlying cause is darkness, blur, glare, angle, or another capture condition.
- Duplicate-suppressed candidates do not need a user-facing diagnostic entry.
- Meld grouping is shown separately from individual detector boxes. Member bounding-box centers are connected using a visual treatment distinguishable from the detector-box treatment, and a compact reconstructed-meld preview is shown above or adjacent to the group connector.
- For a concealed kan inferred from two matching face-up observations, the meld preview shows the two recognized tiles with face-down tiles at both ends so the user can see that the two detector boxes were reconstructed as one concealed kan.
- Exact overlay colors are not fixed, but detector boxes and meld-group connectors must be visually distinguishable.
- The page must not present raw recognized-tile count as progress toward a fixed `14`-tile target because valid physical layouts, including concealed kan presentation, can produce a different number of visible recognized tile faces.
- Device orientation may be called out explicitly when active recognition requires landscape.

Detailed recognition-cause messages are optional rather than the primary recovery mechanism. The overlay should let the user see missing or misrecognized tiles directly while adjusting the physical layout.
Model names, raw classifier labels, confidence thresholds, and inference-provider details are not required user-facing concepts in the production UI.

## Region availability

All three semantic regions remain active throughout recognition.
The page does not provide a tap or toggle interaction for disabling the dora or meld region.
An unused dora or meld region is simply left empty and remains a valid recognition state.

## Actions

- The page provides a way to abandon recognition and return to Top, including while model/runtime preparation is still in progress.
- Leaving Recognition stops/releases the page-owned camera session and realtime run when present. App-owned model asset acquisition and app-lifetime recognition-runtime initialization are not cancelled merely because the Recognition route is left.
- The primary successful transition is automatic stabilization, not a capture action.

## Non-goals

- Manual tile correction.
- Score conditions.
- Score calculation.
- Training/debug telemetry in the production surface.
- Pixel-perfect overlay styling.

## Boundary

| concern | owner |
|---|---|
| Recognition-page visible composition | This concept. |
| Region geometry and stability semantics | `spec:product.recognition.runtime_recognition`. |
| Per-frame boxes, tile identities, and meld-group overlay data | `spec:product.recognition.pipeline`. |
| Auto-transition after commit | `spec:product.ui.screen_flow`. |
| Recognition-correction editing | `spec:product.ui.pages.recognition_correction`. |
