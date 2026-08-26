# PRODUCT-INV-RECOGNITION-003: Collect deployment-layout tile captures

- **status**: investigating
- **date**: 2026-08-05
- **trigger**: PRODUCT-ADR-RECOGNITION-002 defines the deployment composite input, but representative real iPhone captures do not yet exist for detector validation, detector adaptation, tile-crop extraction, or capture-UX evaluation.
- **scope**: Collect original camera frames, semantic-region crops, deployment composites, task slot order, environment conditions, and current NanoDet detections from the fixed completed-hand, dora-indicator, and meld capture layout.
- **non_scope**: Manual bounding-box correction on iPhone, tile classification, final hand interpretation, score calculation, full yaku coverage, and automatic acceptance based on current detector output are excluded.
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-ADR-RECOGNITION-002
  - PRODUCT-INV-RECOGNITION-001
  - PRODUCT-INV-RECOGNITION-002
  - tools/recognition/capture_layout.v1.json
  - tools/recognition/pwa_capture_dataset
  - tools/recognition/capture_dataset_api
  - tools/recognition/annotation_tool
- **follow_up_candidates**:
  - Re-run an adapted detector over every original capture.
  - Match ordered detections to ordered task slots and export a tile-classification dataset.
  - Compare grayscale, binary, and four-level tile-face representations.
  - Evaluate capture-frame dimensions and camera angle from observed user friction.
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-ADR-RECOGNITION-002

## Investigation scope

Collect real iPhone captures using the exact deployment composition contract from PRODUCT-ADR-RECOGNITION-002:

```text
landscape rear camera
  -> visible completed-hand, dora-indicator, and meld frames
  -> exact source crops inside those frames
  -> fixed 320 x 320 black-padded composite
  -> composite-augmented NanoDet inference
  -> original frame, crops, composite, task order, detections, and telemetry
```

The collection application is not an annotation editor.
It records a known physical tile arrangement and preserves enough source data to run improved detectors and preprocessing methods later on a PC.

## Ordered task labels

A task defines semantic order rather than exact pixel positions.
The user places physical tiles in the instructed order anywhere inside the corresponding visible frame.

The ordering contract is:

- Completed hand: left to right.
- Dora indicators: top row left to right.
- Ura-dora indicators: bottom row left to right.
- Meld groups: top to bottom.
- Tiles within one meld group: left to right.

Every task slot records:

- Tile code.
- Front or back face.
- Rotation in multiples of 90 degrees.
- Ordinal within its row or group.

No user action aligns a tile to a predetermined pixel coordinate.
Pixel rectangles are derived only from the camera preview, capture frames, and detector output.

After a detector is re-run on the PC, detections can be sorted by the same semantic order.
A sample whose detected structure matches its task structure can then receive tile-class labels by ordinal correspondence without manually labeling each crop.

## Initial campaign

Use thirty representative physical layouts rather than attempting to cover every yaku.

Each layout is captured under four environment conditions:

| brightness | shadow |
|---|---|
| bright | none |
| bright | partial |
| dark | none |
| dark | partial |

Capture every condition once. Repeated near-identical frames are not part of the initial campaign; additional captures are added only when they introduce a deliberate geometric or lighting variation or target a failed case.

```text
30 layouts x 4 environments = 120 capture tasks
```

The layout generator must validate physical tile inventory for every layout:

- Four copies of ordinary tiles.
- Three ordinary fives plus one red five in each suit.
- Back-facing tiles still consume their underlying physical tile.

The campaign must include, in at least three distinct layouts each:

- All thirty-four ordinary tile classes.
- Red five man, pin, and sou.
- Back-facing tiles as the separate visible `back` class.

Representative geometry must include:

- Closed hands.
- One through four meld groups.
- Chi, pon, open kan, and closed kan.
- Sideways called tiles.
- One-row dora and two-row dora plus ura-dora arrangements.

## Capture behavior

The PWA shows the current task as tile cards before entering the camera.
The camera uses the fixed landscape overlay and the composite-augmented NanoDet model.

Realtime output displays:

- Current bounding boxes.
- Confidence.
- Assigned semantic region.
- Expected and detected count per region.
- Runtime stage timings.

The detector output is informational only.
It must not disable the shutter or force the user to correct missed, duplicate, partial, or merged detections.
No manual bounding-box editor, missed-tile tap, or problem tag is included.

At shutter time, the application must freeze one original camera frame, rebuild its composite from that same frame, and run detection again against the frozen composite.
This ensures that saved detections correspond to the saved images rather than to an earlier live-preview frame.

The review screen shows:

- An annotated copy of the original frame.
- An annotated copy of the `320 x 320` composite while preserving the raw composite separately.
- Enabled semantic-region crops.
- Expected and detected region counts.

The original image itself remains unannotated.

## Persistence contract

Use a Windows local API, SQLite metadata, and image files.

```text
.local/recognition/capture_dataset/
├─ dataset.sqlite
├─ originals/
├─ composites/
└─ regions/
   ├─ hand/
   ├─ dora/
   └─ meld/
```

Each capture stores:

- Linked task definition and ordered tile slots.
- Original camera frame.
- Lossless PNG crops for every enabled original-resolution semantic region.
- Fixed lossless deployment composite.
- Camera-frame region rectangles in source-pixel, normalized, and displayed-preview coordinates.
- Preview viewport and `object-fit: cover` geometry needed to reproduce the shown overlay.
- NMS-processed current-model detections in composite, original-source, and displayed-preview coordinates.
- Camera settings.
- Model name and SHA-256.
- Confidence and NMS thresholds.
- Execution provider and runtime telemetry.
- Capture and storage timestamps.

The PWA writes a draft to IndexedDB before upload.
A failed upload remains pending and can be retried.
The client-generated upload UUID provides idempotency when the server saved a capture but its response was lost.
A capture task accepts only one distinct capture UUID.

## Re-detection and downstream use

Current detector errors do not invalidate a capture.
Original images and task order remain the source material.

The intended PC workflow is:

1. Manually annotate only a small representative subset if detector adaptation is required.
2. Fine-tune the region detector.
3. Re-run detection over all original captures.
4. Sort hand and dora detections horizontally.
5. Cluster meld detections vertically by group and sort each group horizontally.
6. Compare detected structure with the task structure.
7. Assign tile classes by ordinal correspondence when the structure matches.
8. Retain unmatched captures for further detector improvement rather than silently assigning labels.

This permits the same capture corpus to support:

- Deployment-format detector validation.
- Detector fine-tuning.
- Tile-classification crop generation.
- Binary and four-level preprocessing experiments.
- Rotation and perspective correction experiments.
- Capture-layout and lighting UX analysis.

## Implemented artifacts

| artifact | responsibility |
|---|---|
| `tools/recognition/capture_layout.v1.json` | Shared ADR-002 composite dimensions, source aspect ratios, padding value, and layout identity. |
| `tools/recognition/composite_capture_dataset_tool/layout.py` | Loads the shared layout rather than maintaining an independent constant copy. |
| `tools/recognition/pwa_capture_dataset` | Task instruction, fixed camera overlay, composite inference, capture review, IndexedDB pending storage, and API upload. |
| `tools/recognition/capture_dataset_api/campaign.py` | Deterministic thirty-layout campaign generation, physical-inventory validation, and visible-class coverage validation. |
| `tools/recognition/capture_dataset_api/database.py` | SQLite schema, campaign seed, progress, ordered task slots, captures, and detections. |
| `tools/recognition/capture_dataset_api/server.py` | Local HTTP API, multipart validation, idempotent capture storage, filesystem persistence, and annotation endpoints. |
| `tools/recognition/annotation_tool` | PC canvas editor for detector-based initialization, arbitrary-angle rectangle correction, expected-tile labels, validation, draft persistence, and sequential completion. |

## Annotation correction workflow

The saved detector rectangles are only initialization candidates. The annotation source of truth is an arbitrary-angle rectangle in semantic-region crop coordinates:

```text
center x / center y / width / height / continuous angle in degrees
```

The editor supports move, resize, add, delete, free rotation by dragging, optional five-degree snapping while Shift is held, and screen-horizontal or screen-vertical splitting of a merged rectangle.

Expected tile labels are assigned from geometry rather than by manual slot selection:

- Completed hand: left to right.
- Dora: row clustering from top to bottom, then left to right within each row.
- Melds: group clustering from top to bottom, then left to right within each group.

Only front-facing task slots require rectangles. A closed kan therefore expects the two visible center tiles; its two back-facing outer tiles remain in the task definition but do not require annotation.

The editor may save incomplete work as a `draft`. It enables `complete` save and progression to the next capture only when:

- Each row or meld group has the expected number of front-facing tile rectangles.
- Every arbitrary-angle rectangle is fully inside its semantic-region crop.
- No expected region remains structurally incomplete.

The API independently validates the same completion contract before accepting a `complete` annotation.

## Current state

The initial iPhone capture collection has been completed. Observed failures are concentrated in meld regions and become more pronounced in dark captures; adjacent tiles may merge into one detector rectangle, while straight isolated tiles are detected more reliably.

The annotation API and PC correction UI are authored. Python tests, TypeScript build, browser interaction, persisted annotation inspection, and downstream export remain to be executed.
