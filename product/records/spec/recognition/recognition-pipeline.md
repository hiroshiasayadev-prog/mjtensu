# Contract: Recognition pipeline

- **id**: `spec:product.recognition.pipeline`
- **status**: draft
- **date**: 2026-08-28
- **parent**: `spec:product.recognition`

## What this is

Current per-frame recognition pipeline and output contract used by the live Recognition page.
The pipeline turns one camera frame into both live visual observations and an ordered recognition structure suitable for stabilization and later correction/scoring.

## Pipeline stages

One frame is evaluated through the following stages:

```text
camera frame
  -> fixed semantic regions / 320 x 320 composite
  -> NanoDet tile-region detection
  -> region assignment
  -> detector-duplicate suppression
  -> candidate crop extraction
  -> grayscale C8 35-class classification
       -> 34 base tile identities
       -> invalid/background
  -> RGB red-five classification for base 5m / 5p / 5s only
  -> per-tile observations
  -> completed-hand and dora ordering
  -> meld-row grouping and reconstruction
  -> frame recognition snapshot
  -> temporal stabilization
```

There is no separate invalid/background classifier stage. The selected grayscale C8 base classifier has 35 outcomes: the 34 ordinary riichi tile identities plus one invalid/background outcome.
Candidates classified as invalid/background do not become recognized tiles.

The red-five specialist runs only after the base classifier identifies `5m`, `5p`, or `5s`. It refines that base identity to ordinary-five or red-five identity.

## Detector duplicate resolution

Duplicate resolution operates independently inside each semantic region after ordinary detector NMS and before crop classification.
It must preserve separate tile candidates when a larger detector failure overlaps several otherwise distinct detections.

For duplicate-resolution geometry, define the pair overlap as:

```text
overlap(A, B) = intersection_area(A, B) / min(area(A), area(B))
```

The concrete overlap threshold is implementation-owned. The current production value is `0.80`.

Resolution has two ordered steps:

1. **Reject merged bridge boxes before confidence selection.** A candidate `M` is a merged bridge when `M` has greater area than at least two smaller candidates `A` and `B`, both `overlap(M, A)` and `overlap(M, B)` reach the duplicate-overlap threshold, and `overlap(A, B)` does not reach that threshold. `M` is removed regardless of detector confidence. The distinct smaller candidates remain eligible for later duplicate resolution.
2. **Resolve remaining pairwise duplicates greedily.** Process remaining candidates from highest detector confidence to lowest, using detector order and then frame-local detector identity as deterministic tie-breakers. Keep a candidate unless it reaches the duplicate-overlap threshold with a candidate already kept. Do not collapse a transitive connected component merely because an intermediate candidate overlaps two candidates that do not overlap each other at the threshold.

Therefore, for distinct candidates `A` and `B` that barely overlap one another but are both substantially covered by one larger merged candidate `M`, the required result is to remove `M` and preserve `A` and `B` for classification. Detector confidence chooses among actual duplicate alternatives; it does not allow a high-confidence merged box to erase several spatially distinct tile candidates.

This policy addresses detector localization failures only. A retained crop may still be classified as `invalid/background` by the 35-class base classifier.

## Per-frame observation output

The pipeline must expose enough current-frame information for the Recognition page to show what the recognizer is seeing without exposing model tensors or requiring the UI to rerun recognition logic.

For each detector candidate retained after duplicate suppression, the observation output includes at least:

- the semantic region containing the candidate;
- a bounding box that can be mapped onto the visible camera preview;
- the recognized tile identity when the base/red-five classification produces one;
- an unresolved state when the classifier outcome is invalid/background or no supported tile identity is available.

Observation geometry is live feedback data. It is not part of scoring input and does not need to remain stable across frames.

## Meld-group observation output

The pipeline groups current meld-region observations by their spatial row arrangement.
The grouping must provide enough information for the UI to show the grouping independently of the individual detector boxes.

For each current meld group, the output provides at least:

- the ordered member observations;
- geometry sufficient to connect the member bounding-box centers in the visible preview;
- an inferred meld interpretation when the observed identities make one unambiguous.

For a non-empty stable meld partition, the frame output also exposes the selected common meld-row angle as live guidance metadata. It is not part of the recognized semantic structure, stabilization equality, commit eligibility, or scoring input.

A two-member group containing the same base tile identity is interpreted as the visible evidence for a concealed kan. The recognition structure represents it as one logical concealed-kan meld even though only the two face-up tiles produced detector boxes.

For a three- or four-member group, chi/pon/open-kan interpretation may be attached when it follows unambiguously from the observed identities. A geometrically reconstructed group is not rejected merely because its current tile identities do not form a legal scoring meld; correction and scoring validity belong downstream.

Meld grouping uses the Docstrum-inspired spatial procedure defined by `spec:product.recognition.runtime_recognition`. The external inspiration is Lawrence O'Gorman's 1993 Docstrum work on bottom-up page-layout analysis: infer line orientation and within-line relationships from detected component geometry rather than assuming a fixed horizontal baseline. The production implementation adapts that idea to the bounded mahjong case rather than copying Docstrum literally.

For each frame, the pipeline must therefore:

- derive bounded common row-direction candidates from meld bbox-center pair geometry, with a horizontal fallback and `±45°` support limit;
- project observations into row-aligned parallel/perpendicular coordinates;
- enumerate admissible `2..4`-member row candidates using median-tile-size-scaled perpendicular spread and adjacent-gap constraints;
- construct complete partitions that cover every meld observation using at most four rows, rejecting partitions whose rows are not spatially distinguishable;
- rank complete partitions by geometry-only residuals and spacing regularity, then select the unique best partition;
- treat an individual short-row fitted angle as a scoring residual rather than a separate hard rejection threshold.

The exact numeric tolerances and score weights are implementation-owned. The algorithmic structure above is canonical because stabilization depends on deterministic, jitter-tolerant grouping rather than the previous `v`-sorted greedy row cut.

## Frame recognition snapshot

One evaluated frame produces two logically separate views of the same recognition work:

| view | purpose |
|---|---|
| Live observations | Bounding boxes, tile identities/unresolved states, and meld-group geometry for camera overlay. |
| Recognized structure | Ordered completed-hand tiles, ordered dora indicators, and reconstructed meld groups for stabilization and Application. |

The recognized structure may be incomplete or invalid as a winning mahjong hand. Recognition does not reject a stable frame because the current tile combination has no yaku, is not a legal winning shape, or would represent a chombo.
Those questions belong to Conditions and the scoring boundary.

A frame is eligible to participate in temporal stabilization only when it passes the runtime-recognition capture-completeness gate: at least `10` valid visible non-dora tile observations in total and at least `2` valid observations in the completed-hand region. These counts use actual observations rather than logical concealed-kan expansion.

## Stabilization boundary

Temporal stabilization compares the recognition structure, not detector-box coordinates.
Bounding-box jitter alone must not reset stabilization when tile identities, region/order, and meld grouping/reconstruction remain the same.

A stable recognition structure is committed according to the three-consecutive-evaluation rule in `spec:product.recognition.runtime_recognition`.
Scoring validity is not a stabilization criterion.

## Non-goals

- Exact detector confidence thresholds, ordinary IoU NMS parameters, duplicate-overlap threshold tuning beyond the currently recorded production value, or crop-padding constants.
- Exact ONNX/runtime provider configuration.
- Training procedure or dataset composition.
- Bounding-box visual styling.
- Mahjong winning-shape validation, yaku validation, fu calculation, or point calculation.
- Concrete TypeScript interface names.

## Boundary

| concern | owner |
|---|---|
| Per-frame recognition stages and observation/structure outputs | This contract. |
| Capture geometry, meld tilt range, and temporal commit semantics | `spec:product.recognition.runtime_recognition`. |
| Live overlay presentation | `spec:product.ui.pages.recognition`. |
| Recognition correction and current-yaku feedback | `spec:product.ui.pages.conditions`. |
| Scoring validity and point calculation | `spec:product.scoring`. |
| Model-selection rationale and training evidence | Recognition ADRs and investigations. |

## Related records

| ref | relation |
|---|---|
| PRODUCT-ADR-RECOGNITION-002 | Establishes the fixed semantic regions and `320 x 320` composite. |
| PRODUCT-ADR-RECOGNITION-004 | Establishes the current learned detector/crop-classifier runtime pipeline, integrated invalid/background outcome, and downstream scoring-validity boundary. |
| O'Gorman, Lawrence, 1993, *The Document Spectrum for Page Layout Analysis*, IEEE TPAMI 15(11):1162-1173 | External reference for the Docstrum-inspired bottom-up row-orientation and within-line grouping concept adapted by meld grouping. |
