# PRODUCT-INV-RECOGNITION-010: Improve NanoDet localization with balanced replay and geometric augmentation

- **status**: in_progress
- **date**: 2026-08-31
- **trigger**: iPhone 13 debug output with the Plain e150 classifier exposed badly localized detector boxes in the completed-hand, dora, and meld regions. Several boxes include large amounts of background or only part of a tile, yet the classifier still often returns the correct class. The production pipeline crops the classifier input directly from `detection.sourceBox`, so detector localization error propagates unchanged into the base classifier. INV-009 found a real but much smaller local-angle weakness in Plain, making detector crop quality the higher-priority failure hypothesis.
- **scope**: Characterize production NanoDet localization as classifier-crop quality, inventory the available annotated real-capture diversity, rebuild the real/composite/base replay mix with explicit source ratios, strengthen deployment-relevant geometric/photometric augmentation, compare fine-tuning recipes under the existing NanoDet-Plus-m 320 architecture, and conditionally compare the best recipe against a joint mixed-data retrain from the official pretrained NanoDet checkpoint.
- **non_scope**: Retraining the Plain/C8 tile classifiers, changing NanoDet architecture or input size, changing duplicate suppression or semantic grouping merely to hide detector errors, red-five classification, score calculation, UI redesign, or production promotion before iPhone verification.
- **source_refs**:
  - `tools/recognition/build_nanodet_capture_finetune_dataset.py`
  - `tools/recognition/build_nanodet_composite_augmented_dataset.py`
  - `tools/recognition/build_nanodet_region_rotation_augmented_dataset.py`
  - `tools/recognition/nanodet/configs/e1_nanodet_plus_m_320_real_capture_ft10_l10.yml`
  - `tools/recognition/nanodet/configs/e1_nanodet_plus_m_320_composite_augmented_amp40.yml`
  - `.local/recognition/nanodet_capture_finetune_dataset/provenance.json`
  - `.local/recognition/nanodet_composite_augmented_dataset/provenance.json`
  - `.local/recognition/capture_dataset/campaign-initial-120.json`
  - `.local/recognition/capture_dataset/campaign-initial-240.json`
  - `product/frontend/src/recognition/production-pipeline.ts`
  - `PRODUCT-INV-RECOGNITION-001`
  - `PRODUCT-INV-RECOGNITION-003`
  - `PRODUCT-INV-RECOGNITION-009`
- **planned_outputs**:
  - `.local/recognition/nanodet_localization_retrain/`
  - detector crop-quality evaluation artifacts and worst-case contact sheets
  - source-mix / augmentation provenance for every candidate
  - this investigation record

## Question

Can the existing NanoDet-Plus-m 320 detector produce materially tighter and more complete classifier crops on real iPhone layouts when it is trained with:

1. more unique annotated real-capture layouts rather than repeatedly exposing a very small source set;
2. an explicit, measured balance between real capture, deployment-composite replay, and base replay;
3. geometric augmentation that covers the deployment failure modes visible in hand and meld crops;
4. and, if needed, joint mixed-data training instead of a short second-stage fine-tune?

The goal is not merely a higher COCO mAP. The product-relevant output of the detector is the **classifier crop**.

## Why the current detector can pass detector metrics and still fail the product

`product/frontend/src/recognition/production-pipeline.ts` performs the following operation for every accepted detection:

```text
detection.sourceBox
  -> extractCrop(frame.source, detection.sourceBox)
  -> base classifier
```

There is no crop expansion, box rectification, or classifier-side recovery step. A box that is shifted, too narrow, too short, or background-heavy becomes exactly that classifier input.

The existing real-capture acceptance result therefore needs a stronger localization view than TP/FP/FN alone. A detection can satisfy the matching threshold while still yielding a poor classifier crop.

The new iPhone debug evidence is consistent with that failure mode: completed-hand tiles including the visually inspected `2s` and `3m`, as well as several other candidates, are classified from visibly incomplete or badly shifted boxes. The fact that many of those crops are still classified correctly is evidence of classifier robustness, not evidence that the detector crop is acceptable.

## Current training baseline: exact source state

The current production detector is the 15-epoch real-capture fine-tune defined by `e1_nanodet_plus_m_320_real_capture_ft10_l10.yml`, initialized from the accepted composite-augmented detector.

The generated fine-tune dataset provenance is:

| source | train image entries | unique/source basis | annotations | image-entry share | annotation-target share |
|---|---:|---:|---:|---:|---:|
| `real_capture` | 640 | 32 unique real captures repeated 20x | 10,480 | 44.5% | 17.9% |
| `composite_replay` | 286 | 286 deployment-composite images | 4,230 | 19.9% | 7.2% |
| `base_replay` | 512 | 512 sampled base images | 43,762 | 35.6% | 74.8% |
| total | 1,438 | - | 58,472 | 100% | 100% |

The real split uses only ten annotated layouts from `initial-120`: eight training layouts / 32 captures and two validation layouts / 8 captures. The high real-capture image share is therefore obtained largely by repeating a small set rather than by increasing geometric diversity.

This also means that saying the current fine-tune is simply "mostly real" is misleading. It is 44.5% real by image entries, but only 17.9% of the positive annotation targets are real; sampled base replay contributes 74.8% of the annotation targets. Both image exposure and positive-target exposure must be reported for future mixtures.

### Current augmentation is present, but the geometric coverage is weak

The production fine-tune is **not augmentation-free**. Its current NanoDet training pipeline is:

```text
perspective: 0.0
scale:       [0.8, 1.2]
stretch:     [[0.9, 1.1], [0.9, 1.1]]
rotation:    0
shear:       0
translate:   0.08
flip:        0.5
brightness:  0.1
contrast:    [0.8, 1.2]
saturation:  [0.8, 1.1]
```

However, `rotation`, `perspective`, and `shear` are all disabled, and scale/translation/photometric ranges were weakened relative to the earlier composite-augmented training (`scale [0.6,1.4]`, `stretch [0.8,1.2]`, `translate 0.2`, `brightness 0.2`, `contrast [0.6,1.4]`, `saturation [0.5,1.2]`). The earlier composite run also had rotation/perspective/shear disabled.

Thus the current evidence supports a more precise hypothesis than "no augmentation": **the detector has augmentation, but it has not been trained with enough deployment-relevant geometric variation, and the final real-capture fine-tune narrows several augmentation ranges while repeatedly sampling only 32 unique real images.**

## Phase 0: inventory unique real annotations before training

Do not start by increasing `real_repeat` again.

First inventory the capture database by campaign/layout and report:

- captured layouts and captures;
- complete annotations by layout;
- incomplete/unannotated captures;
- region/tile counts for completed hand, dora, and melds;
- lighting/shadow coverage;
- sideways-tile / dense-meld coverage where it is available from task metadata.

`campaign-initial-240.json` exists and records the larger capture campaign, but its complete annotation coverage must be measured from `dataset.sqlite`; do not assume every captured frame is currently usable as supervised detector data.

If additional captures are available but unannotated, prioritize **more unique layouts and known failure geometry** over more repeats of the current 32 training captures. Keep an entire set of layouts as a never-trained deployment holdout.

## Evaluation contract: measure crop quality, not only detection count

Freeze the detector postprocessing threshold/NMS contract while comparing training candidates.

For every matched prediction/ground-truth pair, compute at minimum:

- IoU;
- GT coverage = `intersection(pred, gt) / area(gt)` — how much of the true tile reaches the classifier crop;
- crop purity = `intersection(pred, gt) / area(pred)` — how much of the classifier crop is actually the target tile;
- normalized center error in X/Y;
- predicted/GT width and height ratios (or their log error);
- region label: completed hand / dora / melds.

Also report:

- TP / FP / FN and precision/recall by semantic region;
- duplicate/merged detection counts where identifiable;
- p10 / median / p90 for IoU, GT coverage, crop purity, and center/scale error;
- worst-case contact sheets showing GT, prediction, and the **actual classifier crop** side by side.

The existing 8-image / 2-layout real validation split is too small to be the only deployment acceptance set. Build an additional layout-disjoint iPhone holdout before candidate training. Captures used to define or tune a candidate must not also serve as its final holdout.

### Frozen-classifier downstream check

Use the currently selected Plain random360 e150 classifier unchanged as a secondary crop-quality probe. For matched detector crops, report tile-class accuracy and prediction stability against the task/annotation label where the label mapping is valid.

Optionally report production C8 as a reference, but do not retrain either classifier in INV-010. If a detector candidate improves both geometric crop metrics and frozen-classifier accuracy, that is much stronger evidence than detector F1 alone.

## Dataset-mixture design

The current builder controls the mixture indirectly through `real_repeat` and `base_replay_images`. Replace/extend this with an explicit source-aware mixture contract.

Every generated training dataset must record, separately:

- unique images per source;
- repeated/exposed image entries per source;
- annotations per source;
- image-entry fraction per source;
- annotation-target fraction per source;
- unique-layout count for real capture.

Do **not** freeze a nominal ratio such as `40/30/30` merely by image count: one sampled base image contains far more tile annotations on average than one real/composite image, so equal image fractions do not imply equal positive-target exposure.

### Initial ratio candidates

Use the existing production mixture as `R0` and construct at least one balanced alternative `R1` after the Phase-0 inventory:

- `R0 current`: 640 real-repeat / 286 composite / 512 base image entries, from only 32 unique real train captures.
- `R1 balanced`: increase unique real/deployment-composite representation and reduce the extreme base positive-target dominance while preserving enough base replay to prevent catastrophic regression.

`R1` must be selected from the measured source inventory and documented by both image-entry and annotation-target shares. Prefer adding unique annotated real captures over raising repetition counts.

If one balanced ratio is clearly promising but source balance remains ambiguous, add a small `R2` ratio sweep rather than changing augmentation at the same time.

## Augmentation design

Separate source balance from augmentation so their effects are identifiable.

### A0: current fine-tune augmentation

Use the exact production fine-tune pipeline as the control.

### A1: strengthened deployment-geometry augmentation

Candidate A1 should cover the observed crop/localization failure axes:

- wider scale and stretch, initially using the earlier composite-training ranges as a known-safe reference;
- stronger translation;
- restored stronger photometric ranges;
- modest perspective / camera-plane distortion;
- row/region tilt and rotated-tile geometry representative of real hand and meld placement.

Do **not** blindly enable a large global `rotation` value in the standard NanoDet pipeline. Production composites have a fixed three-region layout; globally rotating the whole 320x320 composite also rotates the layout/padding structure and is not necessarily representative of a tilted tile row inside a fixed capture region.

Before training A1, generate a visual preflight sheet of transformed images and boxes. A1 must preserve the fixed three-region composite layout. Geometry augmentation is therefore applied **inside each semantic-region image before the fixed 320x320 composite is rebuilt**, not by rotating the completed composite.

### A1 fixed-layout rotation contract

For completed hand, dora, and meld semantic regions independently:

1. keep the semantic-region destination rectangle and the surrounding 320x320 composite/padding unchanged;
2. sample a modest in-plane rotation angle for the image content inside that region;
3. derive an isotropic shrink from the absolute angle so stronger tilts receive slightly more margin. The initial contract is `scale(theta) = 1 - 0.10 * (abs(theta) / max_rotation)^2`, giving scale `1.000` at 0°, `0.975` at half of the configured maximum angle, and `0.900` at the maximum angle;
4. apply the same scale + rotation affine transform to the photographic region content and every GT tile rectangle belonging to that region, around the GT-union center;
5. transform all four corners of each original GT rectangle;
6. represent the transformed tile with an ordinary axis-aligned detector box equal to the enclosing rectangle of those four transformed corners: `xmin=min(x')`, `ymin=min(y')`, `xmax=max(x')`, `ymax=max(y')`;
7. compute the union extent of all transformed GT corners in the region. If that union would cross a region boundary, translate the transformed region content and every transformed GT corner together by a feasible `(dx, dy)` so all tile boxes remain fully inside the semantic region;
8. if no translation can fit the transformed GT union inside the region, reject/resample that augmentation rather than clipping a box or allowing it to spill into another semantic region;
9. rebuild the normal fixed-layout 320x320 composite from the transformed semantic regions.

The detector therefore still learns the production contract:

```text
fixed completed-hand region
fixed dora region
fixed meld region
fixed composite padding
```

while the photographic tile geometry inside each region can vary in tilt and position. Rotation must never move a GT box into another semantic region, and training labels must not be clipped merely to make an invalid transform fit.

This first A1 rotation is a **region-level transform**: the photographed region content and all boxes in that region move together. It is not an independent random rotation of each tile. Independent tile-level synthesis, if ever required, is a separate experiment because it changes physical-layout semantics.

Before training A1, generate a visual preflight sheet of transformed region images and boxes, including near-boundary cases that exercised the translation-to-fit path. No A1 candidate is accepted for an overnight run until transformed labels visibly remain aligned with tile faces and all fixed-layout invariants pass automatically.

The initial implementation is `tools/recognition/build_nanodet_region_rotation_augmented_dataset.py`. It accepts any fixed-layout COCO partition whose annotations either carry an explicit semantic `region` or can be unambiguously inferred as fully contained in one fixed destination. It writes augmented PNGs, transformed COCO labels, a per-image transform JSONL, provenance, and an original/augmented preflight contact sheet. Pixels whose inverse affine sample falls outside the original semantic-region crop are filled black; reflected/mirrored padding is explicitly not used because it can fabricate duplicate tile-like structures at the region boundary. The default angle-dependent shrink is 10% at the configured maximum rotation and follows the quadratic curve above; the exact per-region scale is recorded in transform provenance.

The end-to-end experiment runner is `tools/recognition/run_nanodet_localization_inv010.py`. It is intended to run on the NanoDet training host and performs Phase 0 inventory, R1 construction, A1 generation, D1-D3 config generation/training, ONNX export, crop-quality evaluation, worst-crop contact sheets, and D0-relative comparison in one reproducible run. It also supports the conditional D4 joint retrain via `--run-d4`. D1-D3 failures are isolated so one failed condition does not discard the other overnight results, and successful training writes a seed/config-hash completion marker; rerunning the same command reuses only a condition carrying that valid completion marker rather than treating an early `model_best` checkpoint from an interrupted run as complete.

For the first executable R1 recipe, the runner does not hard-code an image-count ratio. The historical D0 real-train and real-validation COCO partitions are treated as frozen artifacts and are not reconstructed from the mutable capture DB; a previously completed layout may later be reopened for correction without invalidating the frozen D0 partition. The DB is consulted only to discover newly completed `initial-120` layouts. New layouts are added to R1 training while the existing D0 validation partition remains untouched, and at least one newly completed layout is reserved as a never-trained final holdout when possible. It then chooses the real-capture repeat count so real positive-target exposure is approximately the composite positive-target count, and samples base-replay images to an annotation target near the mean of those two source target counts. The resulting image-entry and annotation-target shares are written to condition provenance before training.

For D2/D3, A1 does not increase nominal real-capture exposure. Repeated views of each real source are replaced by a deterministic cycle of the original plus multiple independently transformed A1 variants, preserving the exact per-source exposure count while increasing geometric diversity. Composite replay keeps one entry per source image and uses one fixed-layout A1 transform per source. The ordinary NanoDet training pipeline is held identical to D0 for D1-D3, so D2 isolates the precomputed fixed-layout geometric augmentation rather than simultaneously changing global 320x320 transforms.

## Staged training matrix

Hold NanoDet architecture, 320x320 input, postprocessing, seed, starting checkpoint, optimizer family, and 15-epoch fine-tune budget fixed for the first matrix. The current fine-tune uses AdamW `lr=1e-4`, weight decay `0.05`, cosine `T_max=15`, and starts from the composite-augmented checkpoint.

| condition | source mix | augmentation | purpose |
|---|---|---|---|
| D0 | R0 current | A0 current | accepted production baseline; no retrain required |
| D1 | R1 balanced | A0 current | isolate source-mixture effect |
| D2 | R0 current | A1 geometry | isolate augmentation effect |
| D3 | R1 balanced | A1 geometry | test combined fix |

Evaluate D0-D3 with exactly the same crop-quality contract and held-out layouts.

### Conditional Phase 2: test whether fine-tuning itself is the ceiling

The user-visible failure may also reflect a two-stage fine-tune/forgetting limitation. Do not conflate that question with R1/A1 in the first matrix.

If D3 materially improves crop geometry but remains inadequate, train one `D4 joint-mixed` candidate using the best R/A recipe from the official NanoDet pretrained checkpoint under a full training schedule comparable to the earlier 40-epoch composite rebuild. This tests:

```text
composite model -> short real fine-tune
vs
one joint mixed-data training run
```

without changing dataset mixture and augmentation at the same time.

If D4 is run, its schedule and checkpoint selection must be recorded explicitly and it must pass the same base/composite regression validation.

## Acceptance / continuation gates

A candidate is interesting only if it improves the deployment crop, not merely mAP.

Prefer candidates that simultaneously:

1. improve completed-hand and meld **GT-coverage p10** and **crop-purity p10** on layout-disjoint real/iPhone holdout;
2. reduce normalized center/scale error and visibly eliminate truncated/background-heavy crops;
3. improve or preserve frozen Plain classifier accuracy on matched detector crops;
4. preserve real-capture recall/precision and composite/base regression behavior;
5. do not introduce a new FP/duplicate failure mode.

If R1 alone fixes localization, do not add stronger augmentation unnecessarily. If A1 alone fixes it, retain the simpler source mixture. If D3 is the only strong candidate, keep both. If all fine-tune candidates remain poor despite better held-out crop metrics during training, execute the conditional D4 joint retrain rather than mechanically increasing repeat counts or epochs.

## Expected decision

End INV-010 with one of these concrete conclusions:

1. **source mix / unique real diversity was the main problem** — promote the balanced replay recipe;
2. **geometric augmentation was the main problem** — promote the validated deployment-geometry augmentation;
3. **both were required** — retain R1+A1 and document the source/augmentation contract;
4. **short fine-tuning is the remaining problem** — prefer a joint mixed-data retrain if D4 wins;
5. **detector training is not sufficient** — only then consider detector architecture/input-resolution changes or an explicit crop-recovery stage.

Production promotion remains blocked until the selected detector is verified on iPhone 13 debug captures and its classifier crops are visibly and quantitatively better than the current `recognition-v3-2026-08-31` detector path.
