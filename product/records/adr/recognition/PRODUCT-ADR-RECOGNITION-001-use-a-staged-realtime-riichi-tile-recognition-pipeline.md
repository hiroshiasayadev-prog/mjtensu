# PRODUCT-ADR-RECOGNITION-001: Use a staged realtime riichi tile recognition pipeline

- **status**: superseded
- **date**: 2026-08-02
- **depends_on**:
- **supersedes**:
- **migrated_to_spec**:

## Context

The product must recognize a complete riichi mahjong hand from a live camera stream.

The target interaction does not require a shutter operation.
The product must report the first actionable capture failure while the user adjusts the camera.
The product must confirm the result immediately after recognition becomes stable.

A direct multi-class object detector can locate and classify every tile in one model.
That approach combines region detection, tile classification, and capture-condition tolerance in one learned component.
The combined responsibility makes failure diagnosis difficult and increases dependence on broad training data.

Mahjong tile designs vary by manufacturer.
Lighting color, glare, camera angle, and arm shadows also change the observed tile image.
The pipeline must normalize these variations before tile classification where practical.

Chinese mahjong datasets contain useful tile-region examples.
The same datasets also contain designs and bonus tiles outside Japanese riichi mahjong.
The pipeline needs separate training-data boundaries for region detection and tile classification.

## Decision

Adopt a staged realtime recognition pipeline.

| stage | responsibility |
|---|---|
| Frame scheduling | Evaluate one recognition frame every 100 milliseconds. |
| Tile region detection | Use NanoDet as a single-class mahjong-tile detector. |
| Detection quality gate | Validate tile count, clipping, overlap, minimum size, and rough orientation. |
| Tile corner extraction | Find tile-face corners inside each detected region with deterministic image processing. |
| Perspective normalization | Warp each tile face into one fixed-size rectangular image. |
| Image quality diagnostics | Detect blur, glare, overexposure, occlusion, excessive shadow, and insufficient color separation. |
| Shared illumination normalization | Estimate a shared white reference from all detected tiles and correct local illumination variation where needed. |
| Shared palette inference | Infer the active color palette from the combined normalized tile images. Do not assume chromatic colors are present. |
| Multi-level color quantization | Apply one shared palette to every tile image in the frame. |
| Tile classification | Classify normalized and quantized tile images with one tiny CNN batch. |
| Mahjong consistency validation | Order tiles by physical position and reject impossible tile multiplicities or unsupported classes. |
| Temporal stabilization | Confirm only after the same complete tile sequence passes every gate three consecutive times. |
| Score calculation | Calculate yaku, fu, han, and points in a separate deterministic component. |

The camera interaction must not require a shutter action.
The application must show the earliest actionable failure from the current pipeline stage.
The application must identify a specific tile when the failure can be localized.
The application must confirm the hand automatically after temporal stabilization succeeds.

Use the following training-data boundary.

| consumer | allowed training data |
|---|---|
| Tile region detector | Japanese and Chinese mahjong tile images, including flower and season tiles, as one generic tile class. |
| Tile classifier | Japanese riichi mahjong tile images only. Chinese mahjong datasets are not classifier training data. |

Use one shared palette per recognition frame.
Do not infer an independent palette for each tile.
The shared palette must support frames that contain only white and dark markings.
Missing color categories must not create artificial palette entries.

## Rationale

The staged pipeline separates generic object location from manufacturer-sensitive tile classification.
The separation allows each model to use a smaller and more focused training corpus.

Perspective normalization removes camera geometry before classification.
Shared illumination normalization uses the large white tile-face area as an in-frame reference.
Shared palette inference gives every tile one consistent color encoding.

The tiny CNN receives fixed-size tile images with reduced background, geometry, and lighting variation.
The classifier can therefore remain smaller than a full-frame multi-class detector.

Stage-specific gates provide actionable realtime guidance.
A direct detector usually exposes only missing or low-confidence detections.
The staged pipeline can distinguish missing tiles, bad geometry, blur, glare, shadow, and classification uncertainty.

A 100-millisecond evaluation interval limits continuous compute cost.
Three matching results provide an approximate 300-millisecond minimum confirmation window.
The confirmation window is short enough for a shutterless interaction.

## Rejected alternatives

### Direct multi-class object detection

A single YOLO-style model would simplify the first implementation.
The model would also own region detection, tile design variation, lighting variation, and tile classification.
The combined responsibility would require more varied labeled images and provide weaker failure diagnostics.

### Independent palette inference for each tile

Per-tile palette inference would repeat clustering work for every tile.
Cluster identities could also differ between tiles in the same frame.
The inconsistent representation would increase classifier complexity.

### Fixed RGB thresholds

Fixed RGB thresholds would be small and fast.
The thresholds would be brittle under warm lighting, shadows, glare, and manufacturer color variation.

### Shutter-based recognition

A shutter-based flow would reduce temporal coordination work.
The flow would reveal capture failures only after the user takes a picture.
The delayed feedback would remove the primary interaction advantage of the realtime pipeline.

## Consequences

The implementation must define stable contracts between each pipeline stage.
The implementation must retain frame coordinates through perspective normalization for localized guidance.

The architecture adds deterministic image-processing code beside learned models.
The additional components increase implementation and validation effort.

The following risks remain unverified:

- Detection of fourteen closely aligned tiles with one lightweight detector.
- Reliable tile-face corner extraction from a detector bounding box.
- White-reference estimation under partial arm shadows.
- Shared palette inference when only small chromatic regions exist.
- Classification accuracy across unseen Japanese tile manufacturers.
- End-to-end execution within the 100-millisecond evaluation budget.

Follow-up investigations must measure:

- Full-hand exact-match rate.
- Per-tile classification accuracy.
- Unknown-manufacturer accuracy.
- Accuracy under shadow, glare, warm lighting, and perspective distortion.
- Model size and initial load time.
- End-to-end inference latency on target mobile devices.
- Training-data volume required for each learned component.
- Comparison with a direct multi-class detector baseline.

A failure of the staged responsibility split requires a superseding ADR.
A change to one implementation detail may amend this ADR when the core pipeline remains valid.

## Evidence

The repository contains a COCO-formatted mahjong tile dataset under `data/coco_mahjong`.
The dataset can support the initial single-class region-detector investigation.

Existing camera-based mahjong scorers demonstrate that direct multi-class detection is viable on mobile or browser runtimes.
The direct approach provides a practical baseline for latency, model size, and recognition accuracy.

The selected pipeline uses these observed domain properties:

- A scored hand presents a small bounded number of tile faces.
- Tile faces contain large white regions that can provide an illumination reference.
- Tile faces can be normalized to one rectangular coordinate system.
- All tiles in one frame share a related capture environment.
- Score calculation is deterministic after tile recognition succeeds.
