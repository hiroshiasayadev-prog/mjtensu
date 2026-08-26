# PRODUCT-ADR-RECOGNITION-004: Use a 35-class base classifier and keep scoring validity downstream

- **status**: accepted
- **date**: 2026-08-26
- **depends_on**: PRODUCT-ADR-RECOGNITION-002
- **supersedes**: PRODUCT-ADR-RECOGNITION-003
- **migrated_to_spec**: `spec:product.recognition.pipeline`, `spec:product.recognition.runtime_recognition`

## Context

PRODUCT-ADR-RECOGNITION-003 established the learned detector-crop-classifier runtime direction and replaced the earlier deterministic color-normalization pipeline.
That responsibility split remains valid, but two parts of the accepted ADR no longer match the current pipeline.

First, PRODUCT-ADR-RECOGNITION-003 preserved an explicit invalid/non-tile gate before a 34-class grayscale base classifier.
Subsequent detector-crop hard-negative work showed that background, clipped, multi-tile, and otherwise unusable detector crops are most naturally handled in the same grayscale crop-classification responsibility rather than by maintaining a second independent crop classifier/gate.
The selected runtime base classifier therefore has 35 outcomes: the 34 base riichi tile identities plus one `invalid/background` outcome.

Second, PRODUCT-ADR-RECOGNITION-003 described mahjong-level consistency validation as part of recognition acceptance before temporal stabilization.
The current product flow intentionally separates visual recognition from scoring legality.
Recognition must be able to commit a stable observed structure that is complete enough for the user to inspect and correct even when the current recognized identities do not form a legal winning hand or a legal meld composition.
Conditions/correction owns repair, and the scoring boundary owns winning-shape, yaku, fu, and point validity.
Requiring those rules in the camera loop would prevent the user from leaving Recognition precisely when recognition made a semantic mistake that the correction UI is intended to repair.

PRODUCT-ADR-RECOGNITION-002 remains the accepted fixed-region and `320 x 320` composite decision.

## Decision

Keep the staged detector -> crop classifier -> red-five specialist architecture, but revise the crop rejection and recognition-acceptance boundaries.

The current runtime responsibility split is:

```text
camera frame
  -> fixed semantic regions / 320 x 320 composite
  -> NanoDet tile-region detection
  -> semantic-region assignment
  -> detector-duplicate suppression
  -> candidate crop extraction
  -> grayscale C8 35-class classification
       -> 34 base tile identities
       -> invalid/background
  -> RGB red-five classification for base 5m / 5p / 5s only
  -> per-tile observations
  -> ordering and meld-row grouping/reconstruction
  -> capture-completeness eligibility
  -> semantic temporal stabilization
  -> committed recognition structure
  -> Conditions/correction
  -> scoring validity and calculation
```

### Invalid/background is a base-classifier outcome

Use one grayscale C8 base crop classifier whose output vocabulary contains:

- the 34 ordinary Japanese riichi base tile identities; and
- one `invalid/background` outcome.

There is no separate mandatory invalid-crop classifier or independent invalid-gate model between crop extraction and base identity.

A crop classified as `invalid/background`:

- does not become a semantic tile instance;
- does not count toward recognition capture-completeness minima;
- may remain visible as an unresolved/current detector observation for live camera feedback.

Detector-side filtering, region exclusion, and duplicate suppression remain separate responsibilities because they operate on detector geometry/post-processing rather than on crop semantic classification.

### Red-five specialization remains separate

The 35-class grayscale classifier still emits base `5m`, `5p`, and `5s` identities rather than separate red-five classes.
Only those base-five crops proceed to the RGB red-five specialist.

This ADR does not merge red-five discrimination into the base classifier.

### Recognition acceptance is not scoring validation

Recognition acceptance is based on whether the camera result is sufficiently complete and spatially stable to create an editable recognition structure.
It is not based on whether the current tile identities form a legal winning hand or contain a yaku.

In particular, recognition does not reject or keep scanning solely because:

- the completed hand plus melds are not currently a supported winning shape;
- a reconstructed three/four-member meld currently contains tile identities that do not form a legal chi/pon/kan;
- the current hand has no yaku;
- the same base tile appears in a rule-invalid multiplicity caused by recognition error.

Those states are repairable after recognition and therefore belong downstream.

The capture-completeness gate and exact minima are owned by `spec:product.recognition.runtime_recognition` rather than duplicated in this ADR.
Meld-region observations must still be assignable to stable spatial groups before the frame is eligible for stabilization; spatial grouping failure remains a recognition failure because no coherent editable meld structure can be produced.

### Temporal stabilization compares recognition semantics only

The stabilization key contains scoring-relevant recognized semantics such as tile identities, region/order, and meld grouping/reconstruction, but excludes live detector-box jitter.

Winning-shape validity, yaku existence, and scoring-library output are not part of the stabilization key or recognition acceptance.
The same eligible semantic recognition structure must still satisfy the product's consecutive-result stabilization contract before commitment.

### Downstream correction and scoring own legality

Conditions and recognition-correction surfaces may receive a committed `RecognizedStructure` that still requires repair.
They provide semantic tile/meld correction without returning to detector-box editing.

The scoring boundary separately owns:

- conversion from permissive recognition meld drafts to strict scoring melds;
- winning-shape determination;
- yaku existence;
- fu, han, limits, points, and payments;
- defensive rejection of contradictory scoring input.

The recognition implementation must not reproduce those scoring rules as a camera acceptance gate.

## Rationale

The invalid/background decision and base tile identity are both judgments about one detector crop.
Putting them in one 35-outcome classifier avoids an additional model/session, an extra threshold boundary, and inconsistent cases where one classifier accepts a crop while another classifier assigns a strong tile identity.
It also lets reviewed detector-crop negatives directly train the decision boundary that decides whether a crop becomes a tile.

Keeping red-five discrimination separate still matches the evidence from PRODUCT-INV-RECOGNITION-005 and PRODUCT-INV-RECOGNITION-006: base identity is well served by grayscale shape information, while ordinary-versus-red-five identity is color-sensitive and only relevant for three base classes.

Separating recognition completeness from mahjong legality is necessary for the correction UX.
If a stable visual recognition error creates an illegal hand, rejecting the frame indefinitely gives the user no post-recognition place to repair it.
Committing the stable visual semantics and letting Conditions identify the invalid winning shape or malformed meld makes the failure actionable without weakening scoring correctness.

This separation also prevents the camera pipeline from becoming coupled to the concrete scoring library merely to decide whether a visually stable frame may leave Recognition.

## Rejected alternatives

### Keep a separate invalid/non-tile classifier before the 34-class base classifier

A separate model would preserve the boundary described by PRODUCT-ADR-RECOGNITION-003, but it duplicates crop-level learned inference and introduces a second model lifecycle, deployment artifact, threshold/calibration problem, and disagreement surface.

The reviewed detector-crop negative work can instead be represented directly as a 35th base-classifier outcome.
No current evidence justifies retaining another learned gate.

### Accept every retained detector crop as some tile identity

Removing invalid/background handling entirely would force detector false positives, clipped crops, and multi-tile crops into one of the 34 tile identities.
That converts localization/crop failures into confident semantic tiles and makes temporal stabilization easier to satisfy with a wrong structure.
An explicit invalid/background outcome remains necessary even though it is integrated into the base classifier.

### Require a legal winning hand before recognition can commit

A winning-shape gate would prevent obvious impossible recognition results from reaching Conditions, but it would also trap the user on the camera page when the recognizer made a stable tile-identity or meld-composition error.
The product already has a semantic correction surface specifically for that recovery.

Winning-hand legality is therefore a post-recognition concern.

### Require every reconstructed meld to already be a legal chi/pon/kan

Spatial grouping and scoring legality answer different questions.
A row of three detections can be spatially coherent even when one member was misclassified.
Rejecting the row at Recognition would discard useful grouping evidence and force a camera retry instead of allowing one-tile correction.

The pipeline preserves the group and allows unresolved/invalid semantic composition to be repaired downstream.

### Merge red fives into the 35-class grayscale head

The additional 35th outcome addresses invalid/background crops, not red-five color identity.
PRODUCT-INV-RECOGNITION-006 specifically supports an RGB specialist for the red-five task.
There is no current evidence that replacing that specialist with grayscale red-five classes improves accuracy, runtime, or deployment simplicity.

## Consequences

The production recognition model set continues to contain three learned model roles:

- generic NanoDet detector;
- grayscale base classifier with 35 outputs;
- RGB red-five specialist.

The runtime contract for the base classifier must preserve the 35-class output vocabulary and must not assume that every classifier result is a tile.
The current model-runtime spec identifies that deployment contract as `c8-tile-35-v1`.

Recognition/UI code must distinguish an unresolved live detector observation from a committed semantic tile.
An invalid/background classifier result may still have a detector box for feedback but contributes no tile to the committed structure.

Recognition stabilization can now succeed for a stable structure that is not a legal winning hand.
Conditions/correction must therefore remain capable of presenting and repairing such structures, and scoring must remain defensive.

The accepted capture-completeness gate is intentionally a recognition heuristic rather than a hidden hand solver.
Changing the exact observation minima or geometric grouping algorithm can be handled in the current recognition specification when the responsibility split remains unchanged.

PRODUCT-ADR-RECOGNITION-003 is superseded by this ADR.
Its learned detector/crop-classifier direction, grayscale base classification, RGB red-five specialization, duplicate suppression, and ONNX deployment rationale remain historical evidence, but its separate invalid-gate and mahjong-level recognition-validation decisions are no longer current.

PRODUCT-ADR-RECOGNITION-002 remains accepted and is not superseded.

## Evidence

PRODUCT-INV-RECOGNITION-005 validated the grayscale C8 shape-classification approach on detector-derived Japanese riichi crops and established that base-shape classification is no longer the dominant error source in the evaluated corpus.
It explicitly identified invalid/unusable crop handling as a separate follow-up risk after the 34-class experiment.

Subsequent reviewed detector-crop negative work produced persistent 35-class classifier datasets, including:

```text
.local/recognition/tile_classifier_datasets/gray35_jp500_seed42.sqlite
.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v2.sqlite
.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite
```

The v3 dataset extends v2 with 189 reviewed Japanese detector crops from the training partition, including 180 `invalid` examples and 9 valid tile examples. The selected production 35-class run artifact is therefore:

```text
.local/recognition/tile_classifier_runs/
  gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/best.pt
```

The selected checkpoint is epoch 45. Its production preprocessing normalization is `mean = 0.6815832403977466` and `std = 0.2725553681973969`; `c8-tile-35-v1` is bound to those values in the runtime-spec implementation. The exported deployment artifact is `tile-c8-gray35-v3-jp189.onnx` with SHA-256 `b8a8fa3ff6c6d1e944a7593fa0afc947e0cd2513fb79ca46e5f8fcd6e19c97d0`.

`tools/recognition/tile_shape_classifier.py` supports checkpoint-selected class counts without changing the C8 backbone responsibility, and `tools/recognition/export_c8_classifiers_onnx.py` reconstructs the tile-shape output count/class labels from checkpoint metadata for deployment export.
The selected v3 export parity check produced zero prediction mismatches between the source C8 model and ONNX Runtime; ONNX Runtime parity was `allclose=true` with maximum absolute error `1.1920928955078125e-05`.

PRODUCT-INV-RECOGNITION-006 continues to support RGB as the red-five specialist representation and found no meaningful deployment-size or single-sample forward advantage from replacing RGB with the tested Cr/Y+Cr projections.

The migrated current behavior is specified in:

- `spec:product.recognition.pipeline`;
- `spec:product.recognition.runtime_recognition`.

Those specifications define the current 35-class pipeline, invalid/background exclusion, permissive recognition structure, capture-completeness gate, and separation of scoring validity from temporal stabilization.
