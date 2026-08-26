# PRODUCT-ADR-RECOGNITION-003: Use learned detector-crop classifiers for runtime tile recognition

- **status**: superseded
- **date**: 2026-08-20
- **depends_on**: PRODUCT-ADR-RECOGNITION-002
- **supersedes**: PRODUCT-ADR-RECOGNITION-001
- **migrated_to_spec**:

## Context

PRODUCT-ADR-RECOGNITION-001 established the product interaction and the initial staged recognition architecture.
The interaction remains valid: recognition is shutterless, runs repeatedly while the camera is active, reports actionable failures, and commits only after the same complete semantic result succeeds three consecutive times.

The original ADR also made several image-normalization stages mandatory before tile classification:

- deterministic tile-face corner extraction;
- per-tile perspective normalization;
- shared illumination normalization;
- shared palette inference;
- multi-level color quantization.

Subsequent investigations did not support keeping those stages as mandatory classifier inputs.
PRODUCT-INV-RECOGNITION-004 found that deterministic color quantization can discard weak class evidence under dark and shadowed crops.
PRODUCT-INV-RECOGNITION-005 then showed that detector-derived crops can be classified directly with a small grayscale C8 rotation-equivariant CNN at very high observed accuracy.
PRODUCT-INV-RECOGNITION-006 showed that red-five discrimination is better handled by a separate RGB specialist than by a hand-designed Cr or Y+Cr projection.

The detector side has also converged on the fixed semantic capture layout from PRODUCT-ADR-RECOGNITION-002 and on explicit duplicate suppression for overlapping detections.
The application therefore no longer needs the original deterministic palette pipeline as the mandatory bridge between NanoDet and tile identity.

The accepted C8 classifiers were trained with `escnn` on Linux.
Deployment validation confirmed that the accepted shape classifier can be exported to ordinary PyTorch operations and then ONNX while preserving predictions, so browser deployment does not require `escnn` at runtime.

## Decision

Keep the staged detector-then-classifier architecture, but replace the original mandatory deterministic normalization and quantization stages with learned classification directly from detector crops.

The runtime recognition pipeline is:

```text
live camera frame
  -> fixed semantic capture regions
  -> 320 x 320 composite input
  -> single-class NanoDet tile detection
  -> semantic-region assignment
  -> duplicate suppression
  -> detector crop extraction
  -> invalid/non-tile rejection
  -> 34-class grayscale C8 base-tile classification
  -> RGB C8 red-five classification for base fives only
  -> semantic ordering and meld reconstruction
  -> mahjong-level consistency validation
  -> three-result temporal stabilization
  -> committed recognition result
```

### Frame scheduling and capture layout

Recognition remains shutterless.
While the recognition page is active, the application targets one recognition evaluation every 100 milliseconds.

Use the fixed completed-hand, dora-indicator, and meld capture regions and the fixed `320 x 320` NanoDet composite contract from PRODUCT-ADR-RECOGNITION-002.
Disabled regions remain padding and contribute no semantic tiles.

### Tile detection and region assignment

Use NanoDet as a single-class generic mahjong-tile detector.
The detector is responsible for localization, not tile identity.

Assign each accepted detection to the completed-hand, dora-indicator, or meld region using the fixed composite geometry.
Candidates in padding, separators, or disabled regions are invalid.

### Duplicate suppression

Apply explicit duplicate resolution after detector output and before tile classification.

Two detections in the same semantic region are duplicate-connected when their intersection area divided by the smaller bounding-box area is at least `0.80`.
Treat transitively connected detections as one duplicate cluster.
Retain the highest-confidence detection in each cluster and exclude the remaining cluster members from classification.

The duplicate policy is independent of the detector's ordinary IoU NMS because deployment failures have included near-contained duplicate boxes that are better described by overlap relative to the smaller box.

### Detector crop preprocessing

Map each retained detector box back to the corresponding source-region pixels and classify that crop directly.

For the accepted classifiers, preprocess each crop by:

1. preserving crop aspect ratio;
2. resizing to fit within `64 x 64`;
3. centering the resized crop on a `64 x 64` canvas;
4. filling letterbox pixels with the median color of the crop border in the representation required by that classifier;
5. applying the normalization parameters stored with the deployed model.

Do not require tile-face corner extraction, perspective warping, shared palette inference, or deterministic multi-level color quantization before classification.

### Invalid and non-tile rejection

A retained detector crop must be rejectable before it becomes a semantic tile.
The runtime owns an explicit invalid/non-tile gate between crop extraction and committed tile identity.

The concrete rejection model may evolve independently as additional reviewed detector-crop negatives are collected.
Background, unusable crops, multi-tile crops, or otherwise unsupported crops must not be exposed to scoring as recognized tiles.

The absence of the final rejection model must not block implementation of the rest of the recognition pipeline; the gate remains a stable integration boundary.

### Base tile identity

Use the accepted grayscale C8 classifier from PRODUCT-INV-RECOGNITION-005 for the 34 base Japanese riichi tile identities.

Red fives are mapped to their ordinary base-five shape for this stage:

```text
red5m -> 5m
red5p -> 5p
red5s -> 5s
```

The base classifier receives one normalized grayscale `64 x 64` crop per candidate.
The accepted model uses C8 rotation-equivariant features plus residual `+/-22.5` degree training augmentation.

### Red-five specialization

Only crops whose base identity is `5m`, `5p`, or `5s` proceed to the red-five specialist.

Use the accepted RGB C8 binary classifier from PRODUCT-INV-RECOGNITION-006 to distinguish ordinary five from red five.
Do not require an RGB-to-Cr or RGB-to-Y+Cr projection before this classifier.

The red-five specialist receives the corresponding aspect-preserved `64 x 64` RGB crop and its model-specific normalization.

### Semantic assembly and validation

After tile identities are available:

- order completed-hand tiles from left to right;
- order dora and ura-dora rows from left to right;
- recover meld groups from the two-dimensional meld-region geometry and preserve left-to-right order within each group;
- preserve score-relevant meld openness and kan semantics when they can be inferred or otherwise supplied by recognition/correction;
- reject results that cannot form a supported complete semantic recognition structure.

Exact meld-group ambiguity resolution may evolve without changing the classifier architecture.

### Temporal stabilization

A semantic result is committed only after the same complete scoring-relevant result succeeds for three consecutive recognition evaluations.
A differing, incomplete, or invalid result resets the current stabilization run.
Detector-box jitter alone does not make two results different when their semantic tile structure is unchanged.

### Deployment representation

Training may continue to use `escnn` for C8-equivariant models.
Browser/mobile inference must use deployable artifacts that do not require `escnn` at runtime.

The accepted deployment path is:

```text
escnn checkpoint
  -> exported ordinary PyTorch modules
  -> ONNX
  -> onnxruntime-web or another validated ONNX runtime
```

Export is accepted only when prediction parity is verified against the original `escnn` model on representative inputs.

## Rationale

The revised pipeline keeps the useful responsibility split from PRODUCT-ADR-RECOGNITION-001 while removing preprocessing stages that were not supported by later evidence.
NanoDet remains responsible for generic localization and the small C8 classifiers remain responsible for manufacturer-sensitive tile identity.

Direct detector crops preserve weak visual evidence that deterministic thresholding or quantization can destroy.
The C8 base classifier provides rotation robustness without requiring explicit tile-face rectification.
Residual-angle augmentation covers the orientations between the eight discrete C8 group rotations.

Separating base shape from red-five identity matches the information required by each task.
Base identity is predominantly shape-driven and works well in grayscale.
Red-five identity is color-sensitive and benefits from retaining the full RGB signal.
The specialist runs only for three base classes, so color-sensitive inference is not required for every tile.

Explicit duplicate suppression prevents a detector failure mode from being converted into additional classified tiles.
Using overlap relative to the smaller box specifically handles near-contained duplicate detections that may survive ordinary IoU NMS.

Keeping invalid-crop rejection as an explicit gate allows the currently reviewed negative-crop work to be integrated without redesigning detector, classifier, scoring, or UI contracts.

The ONNX parity result removes a deployment concern created by the Linux-only `escnn` training dependency.
The learned C8 architecture can remain a training choice without forcing the browser application to run `escnn`.

## Rejected alternatives

### Keep the original deterministic normalization and quantization pipeline

The original corner extraction, shared illumination normalization, palette inference, and multi-level quantization stages were intended to simplify classification.
Later experiments showed that irreversible quantization can remove useful weak evidence, especially under dark and shadowed crops.
The grayscale and RGB learned classifiers reached stronger observed results without requiring that pipeline.
Those stages are therefore no longer mandatory.

Image-quality diagnostics or targeted deterministic preprocessing may still be added later when evidence shows a specific deployment failure, but they must not be assumed as prerequisites for every crop.

### Use one hand-designed red-sensitive channel for red-five classification

Cr-only input reduces the nominal input channel count but did not produce a meaningful model-size or single-sample latency advantage.
It also collapsed under the initial warm-light holdout before augmentation and produced more audited physical-tile errors than the selected RGB model after the final comparison.
Y+Cr likewise did not outperform RGB after augmentation.

### Use one unified classifier for base identity and red-five identity

A unified color classifier could emit every ordinary and red-five class in one pass.
The current evidence instead validates two smaller responsibilities separately: grayscale base shape and RGB red-five specialization.
Changing to a unified classifier would require new end-to-end evidence and provides no current product advantage.

### Require the final invalid-crop classifier before pipeline implementation

Invalid-crop rejection is necessary before semantic commit, but its concrete model is still being refined from reviewed detector negatives.
Blocking all integration on that model would delay browser deployment, semantic assembly, stabilization, and end-to-end measurement without reducing architectural uncertainty.
The gate is therefore fixed now while its implementation remains replaceable.

### Return to direct multi-class object detection

The later classifier investigations do not invalidate the original reason for separating localization from tile identity.
A direct multi-class detector would again combine background localization, tile class identity, lighting variation, and manufacturer variation in one learned component.
The current detector-crop classifiers already provide strong evidence for the staged responsibility split.

## Consequences

The runtime implementation now requires three learned inference responsibilities:

- one generic NanoDet tile detector;
- one grayscale base-tile classifier;
- one RGB red-five specialist invoked only for base fives.

The application must reproduce the accepted crop resizing, letterboxing, border-fill, and normalization behavior consistently with the training artifacts.
Classifier model metadata must therefore travel with the deployed ONNX files rather than being duplicated as unversioned constants.

The application must port the accepted duplicate-overlap policy to the browser runtime and apply it before classifier inference.

The deterministic corner, perspective, palette, and quantization stages from PRODUCT-ADR-RECOGNITION-001 are removed from the mandatory runtime path.
This reduces implementation complexity but also means classifier robustness is responsible for the geometric and photometric variation represented in detector crops.

The invalid/non-tile gate remains an active implementation risk.
Its concrete model must be validated so that rejecting detector failures does not materially reduce valid-tile recall.

End-to-end deployment still needs measurement for:

- detector-to-classifier exact tile-sequence accuracy;
- full-hand exact-match rate;
- meld-group reconstruction accuracy;
- invalid-crop false accept and false reject rates;
- complete browser inference latency on the target phone;
- whether the 100-millisecond evaluation target remains practical when detector and classifier stages run together;
- memory and initial model-load cost for the complete browser model set.

A future change of detector architecture, classifier architecture, or invalid-gate model does not require a new ADR when the responsibility split remains the same and evidence supports the replacement.
A change back to direct multi-class detection, removal of the staged detector/classifier boundary, or a materially different recognition interaction requires a superseding ADR.

## Evidence

PRODUCT-INV-RECOGNITION-004 rejected deterministic three-color quantization as the mandatory sole classifier representation after dark and shadowed crops showed loss of weak class evidence.

PRODUCT-INV-RECOGNITION-005 validated the selected `64 x 64` grayscale C8 base classifier.
After human resolution of remaining annotation errors, the common-unseen detector-crop evaluation contained 47 classifier errors in 1,276,926 crops, approximately `99.9963%` observed accuracy.
The investigation also showed that C8 equivariance must be paired with residual `+/-22.5` degree augmentation for arbitrary-angle robustness.

PRODUCT-INV-RECOGNITION-006 validated the RGB red-five specialist.
The selected warm-augmented RGB model classified all 24 untouched real warm-light holdout crops correctly and produced four audited physical-tile errors across the 116,083-crop red-five corpus.
The same investigation found negligible parameter-count and single-inference timing benefit from Cr or Y+Cr relative to RGB.

The repository implementation `tools/recognition/detector_duplicate_groups.py` encodes the reviewed duplicate policy using intersection area divided by the smaller candidate area with a default threshold of `0.80`, transitive clustering, and highest-confidence winner selection.

The repository implementation `tools/recognition/export_c8_classifiers_onnx.py` validates the deployment conversion path.
The accepted shape-classifier export produced zero prediction mismatches between the original `escnn` model and exported PyTorch model and zero prediction mismatches between the original model and ONNX Runtime.
Observed ONNX logit difference was approximately `7.63e-06` maximum absolute error and `1.64e-06` mean absolute error for the validation batch used during export.

PRODUCT-ADR-RECOGNITION-002 remains the accepted capture-region and composite-input decision and is not superseded by this ADR.
