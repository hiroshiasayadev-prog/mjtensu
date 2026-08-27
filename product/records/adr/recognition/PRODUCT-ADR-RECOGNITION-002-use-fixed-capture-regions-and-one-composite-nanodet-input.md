# PRODUCT-ADR-RECOGNITION-002: Use fixed capture regions and one composite NanoDet input

- **status**: accepted
- **date**: 2026-08-04
- **depends_on**: PRODUCT-ADR-RECOGNITION-001
- **supersedes**:
- **migrated_to_spec**:

## Context

PRODUCT-ADR-RECOGNITION-001 adopts a staged realtime recognition pipeline whose first learned stage is a single-class NanoDet tile-region detector.

The detector does not need every visible tile on a mahjong table.
It needs only the tiles that contribute to score calculation:

- The completed hand.
- Dora and ura-dora indicators when present.
- Open or concealed melds when present.

Running the detector against an unrestricted camera frame can include unrelated tile walls, other players' tiles, and distant tiles with weak or inconsistent annotations.
Those objects increase false-positive risk and make the detector solve a broader problem than the application requires.

The capture interaction must show the user exactly which pixels will enter recognition.
A visible frame that is later cropped again would make a tile appear accepted by the UI while removing part of it from the detector input.
The application therefore needs one mapping that preserves the complete contents of every enabled capture frame.

The three semantic regions have different expected arrangements:

- The completed hand is primarily one horizontal row.
- Dora and ura-dora indicators may use two horizontal rows.
- Melds are expected as horizontal groups stacked vertically.

The detector still benefits from one fixed square input and one inference call.

## Decision

Adopt a landscape-phone capture overlay with three fixed semantic regions.

```text
+--------------------------------------+
|        Dora indicators    +--------+ |
|                           |        | |
|        Completed hand     | Melds  | |
|                           |        | |
|                           +--------+ |
+--------------------------------------+
```

Use the following capture-region aspect ratios:

| region | aspect ratio | intended capacity |
|---|---:|---|
| Completed hand | 17:4 | One horizontal hand row with capture margin. |
| Dora indicators | 17:4 | Dora and ura-dora indicator rows within one region. |
| Melds | 1:1 | Up to four horizontal meld groups stacked vertically. |

The overlay arrangement may move within the available landscape preview and safe area, but each region must retain its specified aspect ratio.
The complete visible contents of an enabled capture region must be preserved.
No additional crop may be applied inside a displayed frame.

Each region can be enabled or disabled from the capture UI.
A disabled region contributes no camera pixels and is represented by the fixed padding value in the detector input.

For the initial NanoDet-Plus-m 320 condition, compose all enabled regions into one `320 x 320` detector input with the following exact layout:

| region | x | y | width | height | aspect ratio |
|---|---:|---:|---:|---:|---:|
| Completed hand | 7 | 0 | 306 | 72 | 17:4 |
| Dora indicators | 7 | 74 | 306 | 72 | 17:4 |
| Melds | 74 | 148 | 172 | 172 | 1:1 |

This layout leaves two horizontal separator rows between the first and second regions and between the second and third regions.
The remaining pixels are padding.

Use black as the initial padding and disabled-region value.
The exact RGB value must be fixed and shared by training-data generation, validation, and application inference.

For every enabled region:

1. Crop exactly the camera pixels inside the displayed frame.
2. Resize the crop uniformly to the corresponding composite-input rectangle.
3. Do not stretch, trim, or apply an additional inner crop.
4. Transform every retained annotation with the same scale and destination offset.

Run NanoDet once against the composite image.
Assign each detected tile to the completed-hand, dora-indicator, or meld domain from the center point of its predicted bounding box and the fixed destination rectangles.
Detections whose centers fall only in padding or separator pixels are invalid.

The meld region retains two-dimensional detector coordinates.
Downstream meld interpretation may use tile-center alignment to recover the horizontally arranged groups stacked inside that region.
The exact grouping and ambiguity-resolution rules are a separate implementation decision.

Training, validation, and deployment inference must use the same composite-image contract.
The detector dataset must include representative enabled and disabled region combinations rather than relying only on unrestricted full-table images.

## Rationale

Fixed capture regions make the displayed interaction boundary equal to the recognition boundary.
A user who places a tile fully inside a frame can expect every visible pixel in that frame to reach the detector.

The regions remove unrelated table content before inference.
This reduces exposure to distant tile walls, other players' tiles, and poorly annotated small objects that do not contribute to scoring.

The dora region encourages a sufficiently top-down capture angle when dora indicators are required.
When dora or meld recognition is unnecessary, disabling those regions removes their camera content entirely.
This also improves hand-only captures taken from a lower angle because background tiles outside the completed-hand frame do not enter the model.

Preserving three semantic regions in one fixed square image keeps one NanoDet invocation while retaining region identity through geometry.
The detector remains a generic one-class tile detector; the application supplies the semantic meaning of each tile through the destination rectangle.

The selected region sizes preserve the agreed capture-frame aspect ratios without hidden cropping or geometric distortion.
The layout consumes the complete 320-pixel image height while providing deterministic black separation between semantic regions.

## Rejected alternatives

### Unrestricted full-camera inference

Full-frame inference would require the detector to return every visible table tile, including irrelevant distant tiles.
It would increase false-positive exposure and retain the annotation-quality problems already observed in small background tiles.

### Separate inference for every region

Independent inference would give each crop the full model resolution.
It would also require multiple inference invocations and separate scheduling and result assembly.
The initial design instead tests whether one composite 320-pixel input provides sufficient accuracy.

### Re-cropping inside the displayed capture frames

An additional crop could enlarge the tiles within each detector rectangle.
It would also create a mismatch between what the UI claims to capture and what the detector receives.
The hidden loss of visible frame content is rejected.

### Stretching each crop to an arbitrary destination rectangle

Non-uniform scaling would use every destination pixel.
It would distort tile geometry and make the capture-frame aspect ratio meaningless.
The destination rectangles therefore preserve the source frame ratios.

### Dynamic composite layout based on enabled regions

A dynamic layout could enlarge the remaining regions when another region is disabled.
It would make tile scale and region coordinates depend on UI state and would require broader training coverage.
The initial design keeps region coordinates and scale fixed and fills disabled regions with black.

## Consequences

The camera overlay, composite-image builder, annotation transformer, detector post-processing, and generated datasets must share one versioned layout definition.
Duplicating the rectangle constants independently across these components is not acceptable.

Existing full-table NanoDet metrics do not establish accuracy for this deployment input contract.
A representative composite-format validation set must be generated and evaluated.
The detector may require fine-tuning or retraining with composite-format examples.

The fixed layout reduces the pixels available to each region compared with separate full-resolution inference.
The investigation must verify tile pixel size, missed-tile rate, merged detections, and localization quality for:

- Fourteen-tile completed-hand rows.
- Two-row dora and ura-dora arrangements.
- Up to four stacked meld groups.
- Every supported region enable and disable combination.

Black padding becomes a deliberate part of the learned input distribution.
Changing its value or changing destination rectangles requires coordinated dataset regeneration and model validation.

The following values are provisional and may be superseded after deployment-format evaluation:

- NanoDet input size `320 x 320`.
- Composite rectangles `306 x 72`, `306 x 72`, and `172 x 172`.
- Two-pixel separators.
- Black padding value.
- Meld-group reconstruction rules.

A change that preserves fixed semantic capture regions and one composite detector input may amend this ADR.
A change to unrestricted capture, hidden inner cropping, dynamic region scaling, or separate inference responsibilities requires a superseding ADR.

## Evidence

The current NanoDet-Plus-m 320 investigation reports strong overall validation performance on the existing generated full-table dataset, but substantially weaker small-object performance and observed annotation-quality problems in distant background tiles.
Those observations support narrowing deployment input to the scoring-relevant regions rather than requiring full-table detection.

The agreed capture capacities are based on the physical and semantic arrangement of Japanese riichi mahjong tiles:

- A completed hand requires one horizontal row with room for fourteen tiles and capture margin.
- Dora and ura-dora indicators can be arranged within a shared horizontal region.
- Melds form horizontal groups and can be stacked within a square region.

The fixed `320 x 320` layout preserves the capture ratios exactly:

- `306 / 72 = 17 / 4` for the completed hand.
- `306 / 72 = 17 / 4` for dora indicators.
- `172 / 172 = 1` for melds.

### Production detector artifact pin: 2026-08-27

The production implementation of this ADR is pinned to the composite-format detector already used by the deployment capture tooling:

```text
.local/recognition/nanodet_runs/
  E1_plus_m_320_composite_augmented_amp40_seed42/
    model_best/nanodet-plus-m-320-composite-augmented.onnx
```

Its recorded artifact identity is SHA-256 `4768daa5cb44e7bee37fbb69c36063800164d9e9e8c852e5b3c77bc88ce9ac76`, 5,597,449 bytes, with input shape `[1, 3, 320, 320]` and output shape `[1, 2125, 33]`. The production runtime contract remains `nanodet-plus-m-320-v1`.

A later real-capture fine-tune remained evaluation evidence during the original R06 acceptance gate: it reduced false positives in the recorded comparisons but did not uniformly improve aggregate AP, and no accepted decision had yet superseded the composite-augmented model. Production therefore did not silently switch detector identity during that gate.

### Production detector artifact amendment: 2026-08-28

Target-device acceptance subsequently exposed a domain-specific meld-localization failure in the pinned composite-augmented detector. An exact iPhone debug capture preserved the production `320 x 320` composite, detector input tensor, raw detector output, post-processed boxes, and runtime provider identity. Desktop CPU inference on the same tensor matched the captured iPhone WASM raw output within `1.0251998901367188e-05` maximum absolute error, so the observed failure is not attributable to an iPhone execution-provider divergence.

After correcting the separate duplicate-suppression defect, the composite-augmented detector still retained only four meld candidates on the recorded target frame and one retained localization remained an oversized `108.98 x 50.13` merged box. The real-capture fine-tune produced seven tile-scale meld candidates on the same exact detector input.

The detector selection was therefore rerun at the deployed runtime operating point: confidence threshold `0.35`, IoU NMS `0.60`, fixed semantic-region assignment, merged-bridge rejection, and greedy pairwise duplicate suppression.

- Held-out real captures: composite-augmented baseline `TP=124 / FP=3 / FN=4 / F1=0.9725`, meld F1 `0.7273`; real-capture fine-tune `TP=128 / FP=3 / FN=0 / F1=0.9884`, meld F1 `0.9600`.
- Held-out composite validation: composite-augmented baseline `TP=1044 / FP=28 / FN=8 / F1=0.9831`, meld F1 `0.9968`; real-capture fine-tune `TP=1045 / FP=23 / FN=7 / F1=0.9858`, meld F1 `0.9968`.

The real-capture fine-tune is therefore accepted as the production detector candidate. It materially improves target-domain meld recall, removes all held-out real-capture false negatives, preserves held-out composite meld performance, and slightly improves the composite-set runtime F1. The earlier aggregate-AP ambiguity no longer governs deployment selection because it does not reflect the exact deployed operating point and post-processing policy.

The selected run is:

```text
.local/recognition/nanodet_runs/
  E1_plus_m_320_real_capture_ft10_l10_seed42/
    model_best/nanodet-plus-m-320-real-capture-ft10-l10.onnx
```

PRODUCT-TASK-SYSTEM-002-13 owns the mechanical artifact promotion, production build verification, and iPhone re-verification. The promoted production detector is `nanodet-plus-m-320-real-capture-ft10-l10.onnx`, SHA-256 `9587a02dd1bbccfc14a925dc69c66b3c4a34ab628552b840ec113f7899dbf883`, `5,597,449` bytes, pinned by model set `recognition-v2-2026-08-28`. The fixed composite geometry and `nanodet-plus-m-320-v1` runtime contract remain unchanged by this artifact amendment.
