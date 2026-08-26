# PRODUCT-INV-RECOGNITION-005: Validate grayscale C8 tile classification

- **status**: completed
- **date**: 2026-08-12
- **trigger**: PRODUCT-INV-RECOGNITION-004 rejected deterministic three-color quantization as the mandatory sole classifier input because it could discard weak class evidence under dark and shadowed detector crops. A learned classifier therefore needed to be validated directly on detector-derived Japanese riichi tile crops, including arbitrary rotation and the manually captured deployment domain.
- **scope**: Validate whether a small 64 x 64 grayscale C8 rotation-equivariant CNN can classify the 34 base Japanese riichi tile types from detector-derived crops; determine the required residual-rotation augmentation and practical batch size; audit classifier disagreements for annotation and crop-quality defects; and measure whether human-reviewed hard-example mining improves classification on a mutually unseen crop corpus.
- **non_scope**: Red-five discrimination, generic unknown or invalid-crop classification, NanoDet region-detection accuracy, end-to-end detector-to-classifier recognition accuracy, mobile-runtime export, temporal stabilization, hand interpretation, or score calculation.
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-INV-RECOGNITION-003
  - PRODUCT-INV-RECOGNITION-004
  - tools/recognition/build_tile_crop_dataset.py
  - tools/recognition/build_tile_classifier_dataset.py
  - tools/recognition/tile_shape_classifier.py
  - tools/recognition/train_tile_shape_classifier.py
  - tools/recognition/run_tile_shape_classifier_sweep.py
  - tools/recognition/inspect_tile_shape_classifier_errors.py
  - tools/recognition/find_manual_slot_siblings.py
  - tools/recognition/audit_tile_crop_dataset_labels.py
  - tools/recognition/review_tile_crop_label_audit.py
  - tools/recognition/capture_dataset_api/correct_classifier_audit_labels.py
  - .local/recognition/tile_crop_dataset/dataset.sqlite
  - .local/recognition/tile_crop_dataset/quality_audit.sqlite
  - .local/recognition/tile_classifier_datasets/gray34_jp500_seed42.sqlite
  - /srv/data/mjtensu/.local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_cleanlabels_seed42
  - /srv/data/mjtensu/.local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_qualityaudit_seed42
  - /srv/data/mjtensu/.local/recognition/tile_crop_label_audit/common_unseen_old/summary.json
  - /srv/data/mjtensu/.local/recognition/tile_crop_label_audit/common_unseen_new_all_errors/summary.json
- **follow_up_candidates**:
  - Validate a separate RGB specialist for `5m/red5m`, `5p/red5p`, and `5s/red5s` discrimination.
  - Define and validate rejection behavior for background, unusable, multi-tile, and otherwise invalid classifier crops without weakening base-tile classification.
  - Measure end-to-end NanoDet-to-classifier accuracy on deployment-layout captures, where detector localization and crop quality are expected to dominate the remaining error budget.
  - Export the accepted classifier architecture to the intended mobile runtime and measure latency and memory on the target device.
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-001

## Investigation scope

This investigation evaluates the learned tile-type classification stage of the staged realtime riichi recognition pipeline.

The primary question is whether a small learned shape classifier can reliably identify the 34 base tile types from detector-derived crops while preserving enough robustness to arbitrary tile rotation and to the manually captured deployment domain.

The evaluated 34 classes are:

```text
1m..9m
1p..9p
1s..9s
east, south, west, north, white, green, red
```

`red` in this class list is the red dragon (`中`). Red fives are not additional base-shape classes in this investigation:

```text
red5m -> 5m
red5p -> 5p
red5s -> 5s
```

The original red-five label is retained as source metadata when a compact classifier dataset is built, but the grayscale shape classifier is not responsible for distinguishing red and non-red fives. That responsibility remains a separate follow-up classifier problem.

The investigation also uses classifier disagreements as a dataset-quality signal. This does not make the classifier an automatic relabeling authority. Disagreements are reviewed by a human before any label correction, crop exclusion, or hard-example selection is accepted.

## Persistent crop dataset

The persistent classifier-source database contains detector-derived crop records from the Japanese source dataset and from manually captured deployment layouts.

| source | crops |
|---|---:|
| Japanese source (`jp`) | 1,315,951 |
| manual deployment captures (`manual`) | 1,968 |
| total | 1,317,919 |

The persistent database is:

```text
.local/recognition/tile_crop_dataset/dataset.sqlite
```

Manual crops retain capture provenance including capture ID, layout, region, brightness, shadow condition, annotation angle, and expected rotation. The manual corpus covers bright and dark conditions with and without partial shadow.

The first compact classifier dataset used:

- 500 JP training crops per base class.
- 200 JP validation crops per base class.
- All eligible manual crops, split approximately 80/20 by capture rather than by individual crop.
- Seed 42 for deterministic sampling and capture-level split assignment.

Red-five records are merged into the corresponding base-five class after sampling metadata is retained.

## Input representation

The classifier input is intentionally simpler than the deterministic color pipeline evaluated in PRODUCT-INV-RECOGNITION-004.

Each crop is converted to grayscale and transformed to a 64 x 64 tensor by:

1. Preserving the original crop aspect ratio.
2. Resizing with Lanczos resampling.
3. Centering the resized crop in a 64 x 64 square.
4. Filling the remaining letterbox area with the median grayscale value of the crop border.

The classifier does not receive irreversible black/white/red thresholding, face segmentation, fixed color thresholds, artificial blur, artificial JPEG corruption, or synthetic shadow corruption in the baseline evaluated here.

The source crops themselves may already contain detector localization error, background margin, lighting variation, blur, or perspective effects. These natural defects are preserved rather than replaced by a separate synthetic-corruption policy during this investigation.

## Classifier architecture

The accepted model is a small C8 rotation-equivariant CNN implemented with `escnn`.

The principal architecture is:

```text
64 x 64 grayscale
  -> C8 equivariant convolution blocks
     regular fields: 8 / 16 / 32 / 64
  -> GroupPooling
  -> spatial global pooling
  -> 34-class linear head
```

Each regular C8 representation carries eight orientation channels internally. Group pooling removes the orientation coordinate only after the equivariant feature extractor.

The resulting model contains 142,058 parameters.

A conventional small CNN was retained as a reference implementation, but the investigation focused on the C8 model because arbitrary tile orientation is part of the deployment contract.

## Training execution

Training was performed on the Linux server with an RTX 3090 using the existing PyTorch 1.13.1+cu117 environment.

The selected training condition uses:

| setting | value |
|---|---:|
| input | 64 x 64 grayscale |
| architecture | C8 equivariant CNN |
| batch size | 512 |
| evaluation batch size | 4,096 |
| optimizer learning rate | 0.001 |
| weight decay | 0.0001 |
| random rotation augmentation | +/-22.5 degrees |
| seed | 42 |
| AMP | enabled |
| TF32 | enabled |

The compact dataset is loaded into contiguous memory before training. SQLite and PNG decoding are not part of the hot training loop. The compact image cache can be uploaded to VRAM when capacity allows.

Observed training throughput for the selected condition was approximately 7,600 samples per second with approximately 6.17 GiB peak VRAM allocation.

## C8 alone does not provide arbitrary-angle robustness

The first comparison tested the C8 architecture with and without residual rotation augmentation.

A representative no-augmentation result was:

| angle | manual accuracy |
|---|---:|
| 0 degrees | 0.9924 |
| 15 degrees | 0.8906 |
| 30 degrees | 0.8168 |
| 45 degrees | 0.9186 |
| mean | 0.9046 |

A representative `+/-22.5` degree augmentation result was:

| angle | manual accuracy |
|---|---:|
| 0 degrees | 0.9924 |
| 15 degrees | 0.9949 |
| 30 degrees | 0.9873 |
| 45 degrees | 0.9847 |
| mean | 0.9898 |

The result establishes that C8 equivariance does not by itself remove the need for residual-angle training coverage.
C8 gives exact structure at 45-degree group rotations, while random augmentation within `+/-22.5` degrees fills the residual interval between those orientations.

The accepted rotation policy is therefore C8 equivariance plus `+/-22.5` degree training augmentation.

## Batch-size sweep

The selected rotation-augmented and non-augmented conditions were each evaluated with batch sizes 512, 1,024, 2,048, and 4,096.

The best manual-validation results across the original sweep were ordered approximately as follows:

| condition | best manual 0-degree accuracy |
|---|---:|
| rotation augmented, batch 512 | 0.9949 |
| no augmentation, batch 512 | 0.9924 |
| rotation augmented, batch 1,024 | 0.9924 |
| no augmentation, batch 1,024 | 0.9873 |
| rotation augmented, batch 2,048 | 0.9873 |
| no augmentation, batch 2,048 | 0.9822 |
| rotation augmented, batch 4,096 | 0.9796 |
| no augmentation, batch 4,096 | 0.9746 |

Increasing the batch size did not materially reduce elapsed training time for these runs, while validation accuracy degraded as the batch increased.

Batch size 512 was therefore retained.

## Validation disagreements exposed source annotation errors

Manual-validation inspection initially showed a small number of persistent classifier errors.
Visual review of those crops showed that several apparent model failures were actually wrong source labels.

Confirmed examples included:

- Crops labeled `3s` that physically contained `3m`.
- Crops labeled red dragon (`red`) that physically contained white dragon (`white`).
- Crops labeled `1s` that physically contained `1p`.

The manual dataset is generated from capture tasks that repeat the same logical tile slot across lighting and shadow conditions. A wrong logical task expectation can therefore create the same wrong label in several crops.

`find_manual_slot_siblings.py` was used to resolve the affected logical slots across capture conditions. Human review confirmed the sibling crops before correction.

The source capture-task database was then corrected rather than patching only the derived classifier records. The corrections propagated through a manual-source rebuild of the persistent crop dataset.

The resulting manual crop-count changes were exactly consistent with the confirmed corrections:

- `1s -> 1p`: 4 crops.
- `3s -> 3m`: 8 visible crops.
- `red -> white`: 16 crops.

A total of 28 derived manual crop labels changed.

This finding changed the role of classifier error inspection: high-confidence classifier disagreement became useful not only for model debugging but also as a targeted annotation-quality audit.

## Clean-label classifier result

After the confirmed manual source corrections, the selected C8 condition was retrained.

The clean-label model reached perfect 0-degree manual validation early in training. By epoch 100 its full rotation evaluation was:

| angle | manual accuracy |
|---|---:|
| 0 degrees | 1.0000 |
| 15 degrees | 1.0000 |
| 30 degrees | 0.9975 |
| 45 degrees | 0.9924 |
| mean | 0.997455 |

The training run also exposed a checkpoint-selection defect in the initial trainer implementation.
`best.pt` originally used only manual 0-degree accuracy. Because the first 1.0000 score occurred at epoch 23 and updates required a strict improvement, `best.pt` remained the epoch-23 checkpoint even when later epochs had stronger arbitrary-angle robustness.

The trainer was changed so that:

- Only epochs with the complete configured angle sweep are eligible for `best.pt`.
- The checkpoint score is the mean manual accuracy across 0, 15, 30, and 45 degrees.

The default training length was also reduced from 100 to 50 epochs because the classifier reached its useful plateau much earlier; later epochs produced only marginal changes.

## Full-corpus disagreement audit

The clean-label model was then used as a dataset-quality auditor over the persistent crop corpus.

For the first audit, 18,575 compact-dataset training crops were excluded. The model scanned:

```text
1,299,344 crops
```

At zero-degree inference:

| result | count |
|---|---:|
| classifier/label disagreement | 801 |
| disagreement below 0.50 prediction confidence | 111 |
| retained review candidates | 690 |

The 690 retained candidates received a multi-angle consensus pass at 0, 15, 30, and 45 degrees and were then reviewed manually.

The review UI uses the following decision terminology:

| review decision | meaning | dataset policy |
|---|---|---|
| `label_error` | The crop is usable, but the source label is wrong. | Use the human-corrected label. |
| `false_detection` | The crop and source label are valid, but the classifier predicted the wrong base tile. The name is an audit-UI term and does not mean a NanoDet false positive. | Keep the crop and original label; eligible as a hard example. |
| `unusable_crop` | A tile is present but is too clipped, mixed with another tile, or otherwise unsuitable as classifier training data. | Exclude. |
| `background` | The crop is background or otherwise not a valid tile crop. | Exclude. |

Human review produced:

| decision | count |
|---|---:|
| `label_error` | 49 |
| `false_detection` | 624 |
| `unusable_crop` | 14 |
| `background` | 3 |
| total | 690 |

This result is important for interpretation of classifier metrics. A model/label disagreement is not equivalent to a model error once classifier accuracy approaches the quality of the source annotations. In this review set, 49 high-confidence disagreements were resolved in favor of the classifier rather than the source label.

The review process does not automatically rewrite source labels from model predictions. Every accepted label correction or exclusion is a human decision.

## Quality-audit sidecar

Human review decisions are stored separately from the approximately 10 GiB persistent crop database in:

```text
.local/recognition/tile_crop_dataset/quality_audit.sqlite
```

This keeps the source crop artifact immutable during review and permits decisions to be corrected or withdrawn.

The compact classifier builder applies the sidecar with the following policy:

```text
label_error
  -> use corrected_label

false_detection
  -> retain the original source label
  -> when from JP train, force it into training as a reviewed hard example

unusable_crop
background
  -> exclude from classifier datasets
```

For the reviewed corpus, the rebuilt compact classifier dataset selected:

| reviewed class | selected into compact dataset |
|---|---:|
| `label_error` | 45 |
| `false_detection` | 580 |
| `unusable_crop` | 0 |
| `background` | 0 |

The rebuilt compact database contains 26,388 samples in total:

| split | samples |
|---|---:|
| train | 19,195 |
| JP validation | 6,800 |
| manual validation | 393 |

Training contains 17,620 JP samples and 1,575 manual samples.

The compact schema retains both the source label and the effective reviewed label so that a corrected red-five or other source label does not lose provenance merely because the 34-class head later maps it to a base shape class.

## Hard-example retraining

The quality-audited compact dataset was used to retrain the same selected classifier condition.
The robust-checkpoint rule selected epoch 30 with mean manual rotated accuracy `0.991730`.

Its epoch-30 manual results were:

| angle | manual accuracy |
|---|---:|
| 0 degrees | 0.9975 |
| 15 degrees | 0.9975 |
| 30 degrees | 0.9873 |
| 45 degrees | 0.9847 |
| mean | 0.991730 |

This is lower than the clean-label model's best artificial-rotation mean of approximately 0.99746.
The hard-example retraining therefore did not improve every validation metric.

The relevant question became whether the reviewed hard examples improved generalization to real, mutually unseen detector-derived crops rather than merely improving the samples that had been fed back into training.

## Common-unseen evaluation

A fair comparison was created by excluding every crop appearing in either the pre-quality compact dataset or the quality-audited compact dataset.
This prevents the new model from receiving credit for memorizing reviewed hard examples that remain in its evaluation corpus.

The resulting common-unseen evaluation contained:

```text
1,276,926 evaluated crops
40,976 compact-dataset crops excluded
17 quality-audit unusable/background crops excluded
```

Both the old clean-label model and the new hard-example model were evaluated on exactly this same crop set with the same human-reviewed label corrections applied.

### Old clean-label model

| metric | result |
|---|---:|
| zero-degree disagreements | 151 |
| disagreement rate | 0.01183% |
| confidence < 0.50 disagreements | 107 |
| retained candidates | 44 |
| Tier 1 candidates | 10 |
| Tier 2 candidates | 5 |

### Quality-audited hard-example model

| metric | result |
|---|---:|
| zero-degree disagreements | 51 |
| disagreement rate | 0.00399% |
| confidence < 0.50 disagreements | 16 |
| retained candidates at the original 0.50 threshold | 35 |
| Tier 1 candidates | 1 |
| Tier 2 candidates | 0 |

The common-unseen disagreement count therefore decreased:

```text
151 -> 51
```

This is a 66.2% reduction in disagreements on crops unseen by both compared models.
The result demonstrates that reviewed hard-example feedback improved generalization to related unseen cases rather than only memorizing the reviewed crops.

The new model also reduced strong, rotation-consistent wrong predictions: Tier 1 fell from ten candidates to one, and Tier 2 from five to zero.

## Final review of every remaining disagreement

The new model was re-audited with candidate confidence set to zero so that all 51 common-unseen disagreements were retained for human review rather than dropping the 16 low-confidence cases.

Human review found:

| resolution | count |
|---|---:|
| additional annotation errors | 4 |
| classifier errors | 47 |
| total disagreements | 51 |

After resolving those four annotation errors in favor of the observed tile, the measured classifier error rate over the common-unseen corpus is:

```text
47 / 1,276,926 = 0.0036807%
```

The corresponding observed accuracy is approximately:

```text
99.99632%
```

This is not a controlled benchmark of the classifier against human tile-recognition ability, and the investigation does not claim that the model is generally more accurate than a human.
It does establish that, in this crop domain, classifier disagreement is sufficiently reliable to expose human-created annotation defects and to support a model-assisted review workflow.

## Findings

### A grayscale shape classifier is sufficient for base-tile identity in the current crop domain

The 34 base tile types can be classified at very high accuracy without the deterministic three-color representation evaluated in PRODUCT-INV-RECOGNITION-004.
Continuous RGB color information is not required by this base-shape classifier once red fives are intentionally merged into their non-red base classes.

This does not make color irrelevant to the full recognition pipeline. Red-five discrimination remains explicitly outside the 34-class grayscale task.

### C8 equivariance requires residual-angle augmentation

C8 alone left a severe accuracy trough between its discrete group orientations.
Random `+/-22.5` degree augmentation removed most of that trough while preserving zero-degree performance.

The accepted rotation treatment is therefore the combination rather than either mechanism alone.

### Very large training batches are counterproductive for this small classifier

Batch sizes above 512 did not materially improve elapsed training time and consistently produced weaker validation results in the initial sweep.
Batch 512 is the accepted training baseline.

### Classifier error analysis is also annotation-quality analysis

Manual validation and full-corpus audit both exposed source-label defects.
At this accuracy level, treating the annotation file as unquestionable ground truth would overstate classifier error and hide correctable dataset problems.

Human confirmation remains mandatory because a high-confidence classifier can still be wrong.

### Reviewed hard-example mining materially improves real-crop generalization

Forcing reviewed valid misclassifications into the training corpus reduced common-unseen disagreement by 66.2%, from 151 to 51 crops.
This improvement remained after every crop used by either compared compact training dataset was excluded from evaluation.

The hard-example model's artificial manual rotation score was slightly weaker than the clean-label model's best rotation score, so the improvement is not universal across metrics.
The much larger common-unseen real-crop corpus is more representative of the intended classifier input distribution and is the principal evidence for selecting the hard-example model.

### The remaining base-shape classifier error is no longer the dominant pipeline risk

After final human resolution, 47 classifier errors remained in 1,276,926 common-unseen crops, corresponding to approximately 99.9963% observed accuracy.

Further optimization of the same 34-class grayscale shape model is unlikely to be the highest-value next step.
Detector localization error, unusable crops, red-five discrimination, invalid-crop rejection, and end-to-end integration now represent more consequential unresolved risks.

## Judgment

Adopt the following as the current base-tile shape-classification baseline:

```text
64 x 64 grayscale input
C8 rotation-equivariant CNN
8 / 16 / 32 / 64 regular fields
GroupPooling
34 base tile classes
red fives merged into base five classes
+/-22.5 degree random rotation augmentation
batch size 512
human-reviewed quality sidecar and hard-example feedback
```

Use the quality-audited hard-example checkpoint rather than the earlier clean-label checkpoint for subsequent pipeline work.

Use mean manual accuracy across the complete configured rotation sweep for best-checkpoint selection rather than zero-degree accuracy alone.

Continue to preserve source label provenance separately from effective reviewed labels.
Do not automatically relabel crops from classifier output; classifier disagreements may prioritize human review, but accepted corrections remain human decisions.

Do not spend additional classifier-development effort on marginal improvements to the same 34-class shape problem unless later end-to-end evidence shows that this component has again become a material error source.

The next recognition investigations should move to red-five discrimination, invalid or unusable crop handling, and end-to-end detector-to-classifier evaluation.

## Impact on PRODUCT-ADR-RECOGNITION-001

PRODUCT-INV-RECOGNITION-004 already provided evidence against mandatory irreversible three-color quantization before classification.
This investigation supplies the missing downstream classifier evidence: a compact learned grayscale classifier reaches approximately 99.9963% observed common-unseen accuracy on the current base-tile crop task without that quantization pipeline.

The staged detector-then-classifier architecture remains supported.
The specific preprocessing path in PRODUCT-ADR-RECOGNITION-001 should no longer be interpreted as requiring deterministic three-color quantization as the sole classifier input.

A later ADR amendment or superseding ADR should record the accepted classifier input and the separate responsibility for red-five discrimination once that follow-up investigation is complete.
