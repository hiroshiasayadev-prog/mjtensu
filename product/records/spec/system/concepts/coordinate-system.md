# Concept: Recognition coordinate system

- **id**: `spec:product.system.concepts.coordinate_system`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Canonical geometry convention shared by camera input, recognition output, and Recognition-page overlay rendering.
The purpose is to keep visible capture regions and recognition geometry in one coordinate system and prevent detector-composite coordinates from leaking into UI/application contracts.

## Source-frame normalized geometry

Public recognition geometry uses the original camera source frame as its reference space.
Rectangles are normalized to that source frame:

```ts
export interface Size {
  readonly width: number;
  readonly height: number;
}

export interface NormalizedRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}
```

For a valid rectangle:

- `x`, `y`, `width`, and `height` are finite;
- values describe fractions of the source-frame width/height;
- the represented rectangle lies within the source frame;
- `(0, 0)` is the source-frame top-left;
- increasing `x` moves right and increasing `y` moves down.

## Capture-region identity

The exact normalized rectangles used to render the three visible Recognition-page regions must also be the rectangles supplied to recognition for that frame.

There must not be a second hidden inner crop or separately maintained recognition rectangle that can drift from the visible UI boundary.

```text
visible region geometry
        │
        ├─ UI frame/mask rendering
        └─ RecognitionFrame.regions
```

## Observation geometry

`TileObservation.bbox` uses the same source-frame normalized coordinate system.
The Recognition page may therefore map one observation directly into its displayed source-frame transform without understanding detector-composite layout.

## Internal detector geometry

The fixed `320 x 320` detector-composite coordinates established by the recognition product specification are private recognition-pipeline geometry.

Conversion between:

```text
source-frame normalized geometry
        ↕
source-frame pixel geometry
        ↕
320 x 320 detector composite
```

belongs to recognition implementation.
Composite coordinates must not appear in public UI, Application, or scoring contracts.

## Display transforms

The browser may scale/crop the source frame to fit the visible camera preview.
That display transform is a UI/camera concern and does not alter the canonical source-frame coordinates returned by recognition.

Overlay code must apply the same source-to-preview transform to both recognition-region rectangles and observation bounding boxes.
