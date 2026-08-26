# PRODUCT-INV-RECOGNITION-001: Validate lightweight mahjong tile region detection

- **status**: investigating
- **date**: 2026-08-03
- **trigger**: The tile-region detection stage adopted by PRODUCT-ADR-RECOGNITION-001 remains unverified under actual riichi hand capture conditions.
- **scope**: Validate whether a NanoDet-family single-class detector can return one usable bounding box for each visible mahjong tile under realtime capture conditions.
- **non_scope**: Tile classification, image normalization, temporal stabilization, scoring, and mobile application behavior are excluded except where detector output contracts require observation.
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-001
  - data/coco_mahjong/annotations/instances_train2017.json
  - data/coco_mahjong/annotations/instances_val2017.json
  - data/coco_mahjong_jp_v2/README.dataset.txt
  - data/coco_mahjong_jp_v2/README.roboflow.txt
  - data/coco_mahjong_jp_v2/train/_annotations.coco.json
  - data/coco_mahjong_jp_v2/valid/_annotations.coco.json
  - data/coco_mahjong_jp_v2/test/_annotations.coco.json
- **follow_up_candidates**: []
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-001

## Investigation scope

This investigation evaluates the tile-region detection stage of the realtime riichi tile recognition pipeline.

The primary question is whether a NanoDet-family single-class detector can return one usable bounding box for each visible physical mahjong tile with practical detection accuracy and execution cost.

The investigation covers:

- Existing `data/coco_mahjong` annotation structure and bounding-box quality.
- Mahjong-jp v2 annotation structure, split provenance, and bounding-box quality.
- Conversion of both datasets into one `mahjong_tile` class while retaining source provenance.
- Candidate NanoDet variants and input resolutions.
- Reproducible training, validation, evaluation, and performance measurement.
- Instance detection under dense placement, close alignment, partial overlap, clipping, rotation, perspective, background, and lighting changes.
- Bounding-box suitability for downstream tile-corner extraction.
- Detector-derived signals that can support realtime capture guidance.

## Out of scope

- Tile-type classification.
- Japanese and Chinese tile-style classification.
- Red-five classification.
- Final perspective-correction method.
- White-reference estimation.
- Shadow correction.
- Palette inference.
- Color quantization.
- Tiny CNN classification.
- Hand-validity checks.
- Temporal stabilization.
- Score calculation.
- Mobile application UI.

## Background

PRODUCT-ADR-RECOGNITION-001 adopts a staged realtime recognition pipeline.

The ADR assigns generic tile-region detection to a single-class NanoDet component.
The ADR identifies close alignment as an unverified risk. In this investigation, that risk means whether adjacent physical tiles remain separate detector instances; it does not make an expected tile count part of the detector contract.

The initial detector dataset has two sources.

| dataset | role in this investigation |
|---|---|
| `data/coco_mahjong_jp_v2` | Primary Japanese-riichi tile-region data. Its COCO annotation files index 13,236 images across train, valid, and test, despite the included README reporting 5,140 images. |
| `data/coco_mahjong` | Supplementary generic tile-region data. The dataset may contain Chinese tiles, flower tiles, and season tiles. |

All source classes can become one `mahjong_tile` detector class.
The conversion must retain the original dataset identity and category provenance.
Chinese and bonus-tile images remain valid only for generic tile-region detection in this investigation.

The two sources have different downstream eligibility:

- The tile-region detector may train from both `data/coco_mahjong` and `data/coco_mahjong_jp_v2`.
- Any later tile-type classifier may train only from `data/coco_mahjong_jp_v2`.
- Records derived from `data/coco_mahjong` must remain identifiable so that they cannot enter classifier training artifacts.

Tile-type classifier design and training remain outside this investigation.

## What was investigated

| investigation item | current state |
|---|---|
| Repository and authoring-policy inspection | Started. |
| `data/coco_mahjong` inventory and schema inspection | Initial analysis generated; review pending. |
| Mahjong-jp v2 inventory and schema inspection | Train, valid, and test analysis generated; detector-label interpretation reviewed. |
| Per-source bounding-box distribution analysis | Generated for both source datasets; comparative review pending. |
| Dense and adjacent instance coverage | Dataset observations generated; detector evaluation method pending. |
| Single-class annotation conversion | Completed. Generated train, val, and test COCO annotations and provenance were produced with the reviewed split policy. |
| NanoDet implementation and variant review | NanoDet v1.0.0 at commit `d3fb34fa91d6020f273d6d063bf324dcd97bac12` pinned and import verified. |
| Baseline experiment matrix | NanoDet-Plus-m 320 and 416 required; NanoDet-Plus-m-1.5x 416 conditional; seed policy reviewed; E1 and E2 server configs authored. |
| Detection and geometry metrics | Count-independent instance matching, standard COCO metrics, failure-mode metrics, threshold selection, and stratified reporting reviewed. |
| Training and validation execution | Training entrypoint and dependency compatibility verified; experiment execution pending. |
| Latency and model-size measurement | TBD |
| Realtime guidance signal evaluation | TBD |

## Findings

### Local Mahjong-jp v2 export metadata

The repository contains a Roboflow COCO export at `data/coco_mahjong_jp_v2`.

The included metadata states:

- Dataset name: `mahjong-jp v2`.
- Export date: 2023-02-10.
- Image count: 5,140.
- Annotation format: COCO.
- License: CC BY 4.0.
- Preprocessing: EXIF-aware auto-orientation with orientation metadata stripped.
- Generated augmentation: none.

The COCO annotation files contain a different inventory:

| split | images | annotations |
|---|---:|---:|
| train | 12,144 | 1,207,281 |
| valid | 722 | 71,745 |
| test | 370 | 36,925 |
| total | 13,236 | 1,315,951 |

The annotation files are the operative source for detector preparation; the README image count cannot describe the checked-in export as a whole.
All three splits define 75 category entries without duplicate category IDs, duplicate category names, or undefined referenced category IDs.
The inventory includes numeric category names and semantic Japanese tile names, but those distinctions are not part of the region-detector output contract.

### Detector responsibility and single-class label reduction

The NanoDet component is evaluated as a single-class instance detector.
Its responsibility is to return one usable bounding box for each visible physical mahjong tile.
Expected tile count, concealed-hand validity, open-meld structure, and other hand interpretation belong to downstream domain logic.

All tile annotations from both source datasets will therefore be converted to one generated detector category:

```yaml
categories:
  - id: 1
    name: mahjong_tile
```

Original source annotation files remain unchanged.
The generated conversion must retain source dataset, source split, original annotation identity, and original category identity as provenance.
Chinese, flower, and season tile annotations from `data/coco_mahjong` remain eligible for this generic region detector but remain ineligible for later tile-type classifier training.

Detector evaluation must not depend on an expected image-level tile count.
Relevant instance-level failure modes include missed tiles, background false positives, duplicate detections of one tile, merged boxes spanning multiple tiles, fragmented detections within one tile, and unusable localization.

### Generated detector dataset layout and split composition

The single-class detector dataset will be generated under `.local/recognition/nanodet_single_class_dataset`.
Source images will not be copied.
Generated COCO image records will use paths relative to the repository `data` directory, such as `coco_mahjong/train2017/<file>` and `coco_mahjong_jp_v2/train/<file>`.
NanoDet can therefore use `data` as its image root.

The baseline split composition is:

| generated split | source splits |
|---|---|
| train | `coco_mahjong/train2017` and `coco_mahjong_jp_v2/train` |
| val | `coco_mahjong/val2017` and `coco_mahjong_jp_v2/valid` |
| test | `coco_mahjong_jp_v2/test` |

Generated image IDs and annotation IDs are remapped independently within each generated split to avoid collisions.
The COCO files contain only the standard single detector category and do not depend on nonstandard provenance fields during training.

A separate `provenance.json` records each source annotation file, its SHA-256 digest, source dataset and split, generated ID ranges, source list-order mapping, original category inventory, category annotation counts, and tile-type-classifier eligibility.
This avoids a separate provenance entry for each of more than one million annotations while retaining deterministic reverse lookup into the unchanged source COCO files.

The reproducible builder is `tools/recognition/build_nanodet_single_class_coco_dataset.py`.
It produced:

| generated split | images | annotations |
|---|---:|---:|
| train | 13,853 | 1,218,462 |
| val | 1,149 | 75,097 |
| test | 370 | 36,925 |

The generated counts equal the sums of their configured source splits.
The provenance ID ranges are contiguous and non-overlapping within each generated split.
The generated annotation files occupy approximately 157.4 MB in total.

The builder writes:

```text
.local/recognition/nanodet_single_class_dataset/
├─ annotations/
│  ├─ instances_train.json
│  ├─ instances_val.json
│  └─ instances_test.json
└─ provenance.json
```

### Count-independent detector evaluation

Detector accuracy will be evaluated through one-to-one matching between predictions and ground-truth tile instances.
A prediction can match at most one ground-truth instance, and each ground-truth instance can match at most one prediction.
Additional predictions around an already matched tile remain false or duplicate detections rather than additional correct results.

The standard detector metrics are:

- COCO AP50:95.
- AP50.
- AP75.
- Precision.
- Recall.

Evaluation and post-processing must permit at least 150 detections per image because source images can contain more than 100 annotated tiles.
A 100-detection limit would cap recall independently of model quality on the densest images.

The investigation also requires explicit operational error metrics:

- `missed_tile_rate`: ground-truth tile instances without a matched prediction.
- `false_positive_per_image`: unmatched predictions per evaluated image.
- `duplicate_detection_rate`: additional predictions associated with an already matched ground-truth tile.
- `merged_detection_rate`: predictions that materially overlap multiple ground-truth tile instances.
- `fragmented_detection_rate`: multiple prediction fragments contained within one ground-truth tile instance.
- Matched bounding-box IoU p05 and median.

Merge and fragmentation classifications supplement standard COCO evaluation; their exact geometric classification rules must be implemented consistently in the evaluation tooling.

The confidence threshold will be selected by a validation-set sweep rather than fixed before evaluation.
Each candidate threshold must report precision, recall, missed-tile rate, false positives per image, and duplicate-detection rate.
The selected threshold is then frozen before final test evaluation.

Metrics must also be stratified by:

- Source dataset.
- Ground-truth instance density per image.
- Relative ground-truth bounding-box area.
- Tile adjacency or overlap condition.

These stratifications distinguish general detector quality from failures specific to small, dense, or closely adjacent tile instances.
Expected tile count is not used as an evaluation input or correctness rule.

### Baseline NanoDet experiment matrix

The minimum informative baseline compares input resolution before increasing model capacity.

| experiment | model | input size | execution condition | purpose |
|---|---|---:|---|---|
| E1 | NanoDet-Plus-m | 320 x 320 | Required | Establish the lowest-cost baseline. |
| E2 | NanoDet-Plus-m | 416 x 416 | Required | Isolate the benefit and cost of higher input resolution with model capacity held constant. |
| E3 | NanoDet-Plus-m-1.5x | 416 x 416 | Conditional | Test whether remaining accuracy limitations at 416 are caused by insufficient model capacity. |

E1 and E2 must use the same generated dataset, augmentation policy, epoch count, optimizer, scheduler, random seed, pretrained-weight policy, class count, and evaluation limits.
E3 is executed only when E2 remains materially insufficient and the observed errors plausibly indicate capacity rather than data or annotation problems.

The initial comparison uses one run per executed condition.
After selecting the best condition, only that condition receives two additional seeds, for three runs in total.
This investigation first determines whether the detector reaches a practical range; it does not spend three-seed cost on every clearly inferior configuration.

A 512 x 512 condition is not part of the initial matrix.
It becomes a follow-up candidate only if the 416 x 416 results show a resolution-limited failure pattern, particularly on small or adjacent tiles, rather than a general model-capacity or data-domain failure.

The authoritative experiment configs are:

- `tools/recognition/nanodet/configs/e1_nanodet_plus_m_320.yml`
- `tools/recognition/nanodet/configs/e2_nanodet_plus_m_416.yml`
- `tools/recognition/nanodet/configs/e1_nanodet_plus_m_320_stage100.yml`
- `tools/recognition/nanodet/configs/e2_nanodet_plus_m_416_stage100.yml`

All four configs use:

- One `mahjong_tile` class in the main and auxiliary heads.
- `/srv/data/mjtensu/data` as the image root.
- The generated single-class train and validation annotations under `/srv/data/mjtensu/.local/recognition/nanodet_single_class_dataset`.
- Batch size 96, 10 workers, FP32, AdamW, the official 300-epoch schedule, and seed 42 for the initial comparison.
- Resolution-matched official COCO weight files under `/srv/data/mjtensu/nanodet/pretrained`: Google Drive file ID `1Dq0cTIdJDUhQxJe45z6rWncbZmOyh1Tv` for NanoDet-Plus-m 320 and `1FN3WK3FLjBm7oCqiwUcD3m3MjfqxuzXe` for NanoDet-Plus-m 416. The downloaded files are approximately 4.9 MB each; SHA-256 is `2702bf130b47a78db20a3a14585ae17f635c1e6e704fbadc4f49d8680ddbed68` for 320 and `f416a46613cc7cad11736a742ba02053192555eb25d6522dc286b93a206b10c0` for 416.
- Separate E1 and E2 output directories under `/srv/data/mjtensu/.local/recognition/nanodet_runs`.

The source config does not expose a maximum-detections setting in YAML.
Inspection of NanoDet v1.0.0 found a hard-coded `max_num=100` in `nanodet/model/head/nanodet_plus_head.py`.
The NanoDet COCO evaluator also constructs `pycocotools.COCOeval` without overriding its default maximum-detection sequence, whose largest value is 100.
Thus, increasing only the model-head limit would still leave evaluation capped at 100 detections per image.

Both limits will be raised to 200 for this investigation.
The value exceeds the observed maximum source density of 118 while retaining bounded post-processing cost.
The reproducible, idempotent patcher is `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_max_detections_200.py`.
It modifies only the NanoDet-Plus head and COCO evaluator and refuses to patch an unexpected source layout.

The server clone was patched successfully. Verification found `max_num=200` in `nanodet_plus_head.py` and `coco_eval.params.maxDets = [1, 10, 200]` in `coco_detection.py`; both modified modules passed Python bytecode compilation.
The first epoch-10 validation exposed an additional pycocotools summary assumption. The standard bbox summary computes its first overall AP entry with a hard-coded default `maxDets=100`, while the remaining AP and AR entries use `self.params.maxDets[2]`. Because the configured sequence is `[1, 10, 200]`, the first entry had no matching 100-detection dimension and became `-1.000`, even though AP50, AP75, and AR@200 were valid. NanoDet maps this first `coco_eval.stats` entry to its `mAP` save key, so best-checkpoint selection would be invalid if training continued unchanged.
The reproducible patcher `tools/recognition/nanodet/patches/apply_pycocotools_max_detections_summary_200.py` changes only that first bbox summary call to use `self.params.maxDets[2]` and compiles the modified module. The stage-100 run must be resumed or restarted only after this patch is verified.
Before interruption, the epoch-10 E1 validation reported AP50@200 `0.990`, AP75@200 `0.990`, AP-small@200 `0.358`, AP-medium@200 `0.978`, AP-large@200 `0.638`, and AR@200 `0.979`. These are promising early signals, but the overall AP50:95 value is invalid until the summary patch is applied. The unusually high early AP50 and AP75 also justify a later train/validation duplicate and near-duplicate audit before treating the result as evidence of deployment-level generalization.
Results from this evaluator must be labeled as AP or mAP at a maximum of 200 detections per image rather than compared directly with official COCO results evaluated at 100.

### COCO pretrained-weight compatibility with a single-class head

NanoDet v1.0.0 loads configured pretrained weights through `load_model_weight` in `nanodet/util/check_point.py`.
The loader compares each checkpoint tensor shape with the corresponding model tensor shape before calling `load_state_dict`.
When a key exists but its shape differs, the loader reports that the parameter was skipped and substitutes the model's current initialized tensor.
Unknown checkpoint keys are dropped, and model keys absent from the checkpoint retain their current initialized values.

Therefore, the 80-class COCO checkpoint can warm-start the one-class E1 and E2 models without an additional selective-loading patch.
Class-dependent tensors in the main and auxiliary heads remain newly initialized, while shape-compatible backbone, FPN, and head tensors are loaded from the checkpoint.
Both downloaded weight files were deserialized successfully as dictionaries containing only a `state_dict` entry. Each state dict contains 645 parameter tensors and begins with `backbone.conv1.0.weight`, matching the format expected by NanoDet's loader.
The initial E1 preflight exposed that ShuffleNetV2 separately attempts to download its default ImageNet backbone weights while the model is being constructed, before NanoDet applies the configured full-model `load_model` weight. This download failed with a connection reset and is redundant for these experiments. Both authoritative configs therefore set `model.arch.backbone.pretrain: false`; initialization then proceeds directly to loading the verified NanoDet COCO weight file.
A subsequent preflight exposed a NanoDet v1.0.0 training-entrypoint defect: `tools/train.py` treats every checkpoint without `pytorch-lightning_version` as an old training checkpoint and calls `convert_old_model`, which requires `epoch` and `iter`. The official weight-only files contain only `state_dict`, so this path fails with `KeyError: 'epoch'` before `load_model_weight` is called. The reproducible patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_weight_only_load_model.py` changes the conversion condition to require `epoch` and `iter`, while allowing a weight-only `state_dict` file to pass directly to the existing shape-aware loader.
The first successful training startup log must be retained so the skipped parameter list can be audited.

The corrected E1 preflight then reached active GPU training successfully. At batch size 96 and FP32, the RTX 3090 used approximately 11.2 GiB and processed about 50 iterations in 50 to 53 seconds. With 144 iterations per epoch, this is approximately 2.4 to 2.5 minutes per E1 training epoch before validation overhead. Loss values decreased over the first three epochs, confirming that data loading, augmentation, pretrained-weight loading, forward/backward execution, and optimizer updates all operated successfully.

An otherwise matched AMP preflight used approximately 8.36 GiB and processed 50 iterations in 46 to 49 seconds. This reduced memory by about 25% but improved observed throughput by only approximately 5% to 8%. Because the initial E1/E2 comparison does not need the memory headroom and the official baseline uses FP32, AMP is not adopted for the required baseline runs solely as a speed optimization.

The server exposes 12 logical CPUs and approximately 62 GiB of RAM, with about 59 GiB available during the investigation. The baseline already uses 10 DataLoader workers, so increasing the worker count substantially above the CPU count would risk oversubscription rather than remove the bottleneck. The first DataLoader optimization therefore keeps `workers_per_gpu: 10` and adds `persistent_workers: true` plus `prefetch_factor: 4` to the training loader only. This keeps training workers alive across epochs and allows up to four prepared batches per worker while preserving the YAML-configured worker count. Validation workers remain non-persistent because validation occurs only every ten epochs.

Process inspection after the first patch showed the main training process at approximately one CPU core, ten training workers with a combined lifetime average near 1.2 cores, and another ten validation workers retained after validation. All worker processes were observed in `do_poll`, indicating that they were waiting on their queues rather than saturating the available CPUs. CPU affinity allowed cores 0 through 11. This evidence means the training prefetch queue was already keeping up and that offline augmentation or additional workers are unlikely to remove most GPU idle periods. The authoritative patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_dataloader_prefetch.py` was corrected to keep only training workers persistent and to remove the unnecessary persistent validation-worker pool on the next process restart.

A native `py-spy` profile of the active E1 main process collected 900 samples with 100% active time and reported `dynamic_k_matching` in `dsl_assigner.py` beneath 79% of sampled wall time, while the function's own Python-frame time was only 5%. This identifies NanoDet-Plus dynamic label matching as the dominant training-step bottleneck rather than the DataLoader. The low own-time share indicates that the cost is mainly in tensor operations, CUDA launches, and synchronization performed from that function rather than ordinary Python bytecode alone. The observed GPU pattern is consistent: convolution-heavy intervals reach 90% to 100% compute and 50% to 66% memory-controller utilization, while matching intervals remain near 30% compute with near-zero memory-controller utilization despite high clocks and about 200 W draw.

Source inspection found that `dynamic_k_matching` loops over every ground-truth instance, calls `dynamic_ks[gt_idx].item()`, then launches a separate `torch.topk` for that ground-truth column. Reading a CUDA tensor through `.item()` requires a host-visible scalar and therefore introduces a synchronization point for each ground-truth instance. With the dense mahjong images, this produces many synchronization points and small top-k launches per assignment call. The later `if prior_match_gt_mask.sum() > 0` condition introduces another host-visible CUDA scalar and synchronization per assignment call.

The reproducible optimization patcher is `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_vectorized_dynamic_k_matching.py`. It replaces the per-ground-truth loop with one batched top-k over all ground-truth columns, a rank mask derived from each dynamic k, and one scatter into the matching matrix. It also replaces the host-side conflict condition with branchless indexed conflict resolution. The companion `tools/recognition/nanodet/validate_dynamic_k_matching_vectorization.py` checks exact output equivalence on non-tied synthetic cases on CPU and CUDA and reports a representative 2,025-prior by 100-ground-truth CUDA microbenchmark.

The validator passed exact CPU and CUDA equivalence for 20 priors by 1 ground truth, 64 by 5, 256 by 20, and 2,025 by 100. On the RTX 3090, the representative 2,025-by-100 CUDA microbenchmark measured 6.414 ms per call for the original implementation and 0.387 ms per call for the vectorized implementation, a 16.57x matching-stage speedup. The active E1 run was stopped cleanly after 58 displayed epochs at global step 8,352, and the patch compiled successfully in the pinned NanoDet source.

E1 resumed successfully at displayed epoch 59 from the same checkpoint. Consecutive 50-iteration intervals within epochs completed in approximately 26 seconds, compared with the previous 50 to 53 seconds, yielding about a 1.9x to 2.0x end-to-end training throughput improvement. GPU traces changed from prolonged approximately 30% compute utilization to frequent 70% to 100% intervals, while short approximately 30% intervals remained. The observed loss values remained in the same range without an obvious discontinuity after resume. An approximately eight-second zero-utilization interval occurred at the epoch boundary and is treated separately from training-step throughput, likely reflecting checkpoint or epoch-transition work. Because `torch.topk` does not guarantee stable ordering for exact ties, the epoch-60 AP@200 validation result remains the next required behavioral check before adopting the patch as the investigation baseline.

A second native profile after vectorization collected 3,600 samples with 100% active time. `dynamic_k_matching` no longer appeared as the dominant frame, but `target_assign_single_img` covered 100% of the sampled training interval and `DSLAssigner.assign` remained beneath 74% of sampled wall time. `_single_tensor_adamw` accounted for approximately 3.47 seconds of the 20.87-second profile interval. This shows that the first synchronization defect was real and fixed, but most remaining training-step overhead still lies inside per-image target assignment rather than optimizer execution or input preparation.

Source inspection showed that `naive_collate` intentionally leaves NumPy arrays as Python lists. `target_assign_single_img` then converts each image's ground-truth boxes and labels separately, producing up to 192 small host-to-device transfers per iteration at batch size 96 before ignore targets. The assignment path also materializes repeated classification tensors, constructs a concatenated `[num_priors, num_gt, 4]` inside-box delta tensor, computes conflict argmin for every valid prior, and converts two CUDA scalar average factors through `.item()` per iteration. The reproducible patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_assignment_pipeline_optimizations.py` batches ground-truth box and label transfer once per tensor type, replaces repeated copies with broadcast views, removes the large concatenated inside-box temporary, restricts conflict argmin to conflicting priors, and keeps average factors as scalar tensors. `tools/recognition/nanodet/validate_assignment_pipeline_optimizations.py` verifies these transformations on CPU and CUDA and benchmarks the transfer and classification-cost stages before the patch is applied to the active training source.

The active E1 run stopped cleanly after 83 displayed epochs at global step 11,952. The assignment-pipeline validator passed CPU and CUDA equivalence. On the RTX 3090, representative packed ground-truth transfer time decreased from 1.634 ms to 0.181 ms, a 9.03x transfer-stage speedup, while the classification-cost broadcast change decreased 0.137 ms to 0.119 ms, a 1.15x speedup. The patch applied and compiled successfully. A scan of the patched training path found no remaining operational `.item()`, `.cpu()`, or `.numpy()` calls in the assigner or training-loss code; the remaining host conversions are confined to post-processing and evaluation paths. End-to-end iteration timing and a third native profile remain required to quantify how much of the remaining `DSLAssigner.assign` cost is removed.

A later native profile exposed DataLoader IPC overhead after the assignment path was accelerated. `rebuild_storage_fd` appeared beneath 40% of sampled wall time, multiprocessing queue `get` beneath 45%, and the pin-memory path beneath 11%. Source inspection confirmed that the dataset already returns each transformed image as a Tensor, but `naive_collate` leaves the batch as a list and `TrainingTask._preprocess_batch_input` performs one device transfer per image before stacking on the GPU. At batch size 96 this creates many shared-storage reconstructions, pin-memory operations, and host-to-device copies for a fixed-shape field. The server uses the `file_descriptor` sharing strategy and has a file-descriptor limit of 1,048,576, so this is overhead rather than descriptor exhaustion. The reproducible patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_hybrid_image_collate.py` stacks only the image field inside the DataLoader worker, preserves variable-length targets as lists, and transfers the resulting pinned batch tensor once with `non_blocking=True`. The companion `tools/recognition/nanodet/validate_hybrid_image_collate.py` verifies padded-image and metadata equivalence and benchmarks the complete DataLoader-to-device path. E1 was stopped cleanly after 94 displayed epochs at global step 13,536 before applying this patch. The validator passed padded-image and variable-length-target equivalence. In the representative CUDA pipeline benchmark, observed image storage units decreased from four to one, end-to-end time for ten measured batches decreased from 0.315 seconds to 0.216 seconds, throughput increased from 3,050.1 to 4,453.6 images per second, and the measured pipeline speedup was 1.46x. After resume, steady-state training remained approximately 25 seconds per 50 iterations. The observed six-to-eight-second GPU-idle interval occurred exactly at an epoch boundary: `training_epoch_end` synchronously writes the full `model_last.ckpt` every epoch. Intervals wholly within an epoch remained 25 seconds, while intervals crossing an epoch boundary took 31 to 32 seconds. The idle interval is therefore checkpoint-write overhead rather than continuing DataLoader starvation. The hybrid collate remains accepted for reducing IPC and transfer overhead, but checkpoint cadence is the appropriate control for the remaining periodic GPU idle time.

A native profile of the COCO validation phase showed `evaluateImg` beneath 100% of sampled wall time with the GIL held for 100% of samples. The pinned pycocotools implementation computes IoUs once, then evaluates 1,149 images across one category and four area ranges through a serial list comprehension, yielding 4,596 independent `evaluateImg` tasks. The reproducible patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_parallel_coco_evaluation.py` adds a `ParallelCOCOeval` subclass and switches NanoDet to it. The parent preserves upstream preparation and IoU computation, while a ten-process `spawn` pool evaluates the independent tasks. `Pool.map` preserves the category-area-image order required by `accumulate`. `spawn` is selected instead of `fork` because evaluation is launched from a process that has already initialized CUDA. The companion `tools/recognition/nanodet/validate_parallel_coco_evaluation.py` compares every `evalImgs` record, accumulated precision/recall/score arrays, and all twelve summary statistics against the upstream serial evaluator before the parallel implementation is adopted. The full validation passed for all 4,596 `evalImgs` records, the accumulated precision, recall, and score arrays, and all twelve summary statistics. On the server, serial per-image evaluation took 62.246 seconds and ten-process parallel evaluation took 19.758 seconds, a 3.15x speedup. Accumulation remained approximately 0.5 seconds in both cases, confirming that only `evaluateImg` required optimization. The parallel evaluator is therefore accepted for the required NanoDet comparison runs.

E1 completed the full 100-epoch comparison stage successfully, and the epoch-100 validation produced a new best checkpoint. At `maxDets=200`, overall COCO AP was 0.978774, AP50 was 0.990089, AP75 was 0.990055, and AR was 0.987. The reported area-stratified AP values were 0.434679 for small objects, 0.988706 for medium objects, and 0.701503 for large objects. The corresponding area-stratified recall values were 0.525, 0.999, and 0.754. The class table reported 99.0 AP50 and 97.9 mAP for `mahjong_tile`. The final parallel per-image evaluation completed in 30.30 seconds in the live training process, followed by 0.46 seconds for accumulation. The model artifacts were saved under `/srv/data/mjtensu/.local/recognition/nanodet_runs/E1_plus_m_320_seed42/model_best`, and Lightning stopped normally because `max_epochs=100` was reached. These results establish that the 320-pixel NanoDet-Plus-m condition is already highly accurate overall, while small-object and large-object strata remain the main localization weaknesses to compare against E2.

The matched E2 FP32 preflight used approximately 18.9 GiB. After the accepted assignment, collate, and evaluation optimizations, the live E2 stage-100 run stabilized at approximately 34 to 36 seconds per 50 iterations within an epoch. Intervals crossing an epoch boundary remained approximately 40 to 45 seconds because the full resume checkpoint was still written every epoch. The E2 losses decreased normally through the initial warm-up and early cosine schedule, with memory stabilizing at approximately 18.9 GiB.

The remaining steady-state GPU-utilization troughs correlate with the per-image assignment phase. The current head still invokes `target_assign_single_img` through Python `multi_apply` once per image, while `bbox_overlaps` already supports batch dimensions. The reproducible patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_chunked_batch_assignment.py` therefore adds a single-class, no-ignore fast path controlled by `NANODET_ASSIGN_CHUNK`. It pads only each configured image chunk, performs inside-box tests, IoU calculation, classification cost, dynamic-k matching, conflict resolution, and target construction in batched CUDA operations, and retains the existing per-image implementation as the fallback. `tools/recognition/nanodet/validate_chunked_batch_assignment.py` compares integer assignments exactly, floating overlaps and all generated targets within tight FP32 tolerance, and benchmarks batch latency and peak allocated memory for each chunk size against the legacy loop. E2 stopped cleanly after 16 displayed epochs at global step 2,304 before applying the patch. CPU and CUDA equivalence passed for the initial synthetic cases. On an E2-shaped CUDA benchmark with batch size 96, 3,549 priors, sampled mean 83.83 ground truths, and sampled maximum 118, the first run measured 97.978 ms per batch for the legacy per-image loop. Chunk 4 took 44.204 ms, a 2.22x speedup; chunk 8 took 33.728 ms, a 2.90x speedup; and chunk 16 took 29.253 ms, a 3.35x speedup. The tightened validator then confirmed exact integer assignment and tolerance-bounded floating target equivalence for the actual batch-96, 3,549-prior dimensions at chunk sizes 16, 32, 48, and 96. In the repeated benchmark, the legacy loop took 96.244 ms with a 74.8 MiB assignment peak. Chunk 16 took 29.071 ms with a 297.3 MiB peak, a 3.31x speedup; chunk 32 took 27.017 ms with a 526.6 MiB peak, a 3.56x speedup; chunk 48 took 26.215 ms with a 756.5 MiB peak, a 3.67x speedup; and chunk 96 took 25.502 ms with a 1,446.9 MiB peak, a 3.77x speedup. Because chunk 96 improved assignment latency by only 3.569 ms over chunk 16 while requiring approximately 1.15 GiB more assignment peak memory, chunk 16 was selected as the balanced default. After resuming E2 at epoch 17 with chunk 16, memory remained approximately 19.0 GiB, loss values continued smoothly, and intervals wholly within an epoch decreased to 29 to 32 seconds per 50 iterations, centered near 30 seconds. This is approximately four to six seconds, or 12% to 17%, faster than the prior 34-to-36-second E2 steady state and is consistent with the assignment microbenchmark's predicted savings.

The reproducible patcher `tools/recognition/nanodet/patches/apply_nanodet_v1_0_0_training_overhead_reductions.py` additionally makes full resume-checkpoint cadence configurable through `NANODET_CHECKPOINT_INTERVAL`, always saves the final epoch, and removes duplicate `.item()` calls for each logged training loss. A five-epoch checkpoint interval is the initial long-run candidate and limits crash recovery loss to fewer than five completed epochs. Live E2 timing after the patch showed that intervals crossing non-checkpoint epoch boundaries still took approximately 34 to 37 seconds versus about 30 seconds wholly within an epoch. The earlier attribution of the entire epoch-boundary delay to checkpoint writing was therefore incomplete: checkpoint writes explain an additional periodic stall, but approximately four to seven seconds of epoch-transition overhead remains even when no full checkpoint is written.

The required comparison will therefore use staged execution. `e1_nanodet_plus_m_320_stage100.yml` and `e2_nanodet_plus_m_416_stage100.yml` stop each condition at epoch 100 while retaining `CosineAnnealingLR.T_max: 300`. The epoch-100 validation results and learning curves are compared, and only the selected condition is resumed from its `model_last.ckpt` through epoch 300. This avoids retraining the selected condition's first 100 epochs and reduces the expected initial decision cost to about nine GPU-hours.

The first chained stage-100 driver was interrupted after the E1 run had reached epoch 38 because the custom-maxDets summary defect was discovered. PyTorch Lightning handled SIGINT as a graceful shutdown and returned success, so the shell driver immediately started E2 despite `set -euo pipefail`. Further staged runs must be launched separately, or the driver must explicitly verify completion at the intended final epoch before starting the next experiment.

The interrupted E1 run retained a valid 65 MB `model_last.ckpt`. Its Lightning checkpoint metadata reports zero-based `epoch: 37` and `global_step: 5472`; with 144 iterations per epoch, this equals 38 completed displayed epochs. The checkpoint includes model state, optimizer state, learning-rate scheduler state, callbacks, and training-loop state, so E1 can resume at the following epoch without repeating the first 38 epochs.

### Verified NanoDet training environment

The training environment was verified on the Linux server without Docker and without changing the host CUDA installation.
The canonical server project root is `/srv/data/mjtensu`; `/srv/bugrat/data-lv/mjtensu` resolves to that location.
The authoritative Investigation and tooling sources remain in the Windows repository at `C:\Users\imved\projects\mjtensu`; the server is used as an execution environment.

| component | verified value |
|---|---|
| NanoDet source | v1.0.0, commit `d3fb34fa91d6020f273d6d063bf324dcd97bac12` |
| Environment manager | uv 0.12.1 |
| Python | 3.10 virtual environment |
| PyTorch | 1.13.1+cu117 |
| PyTorch CUDA runtime | 11.7 |
| PyTorch Lightning | 1.9.5 |
| Setuptools | 80.9.0, retained because Lightning 1.9.5 imports `pkg_resources` |
| GPU | NVIDIA GeForce RTX 3090 |
| Reported GPU memory | 23.56 GiB |
| Host CUDA capability | 13.0 as reported by the server environment; left unchanged |

`torch.cuda.is_available()` returned true and the RTX 3090 was detected successfully.
`uv pip check` reported all 60 installed packages as compatible.
NanoDet, PyTorch Lightning, and PyTorch imported successfully, and `python tools/train.py --help` reached the training command-line parser.

The single-class COCO builder was then executed on the server against `/srv/data/mjtensu` and produced the same counts as the Windows-side artifact:

| generated split | images | annotations |
|---|---:|---:|
| train | 13,853 | 1,218,462 |
| val | 1,149 | 75,097 |
| test | 370 | 36,925 |

The server-side generated artifact is available at `/srv/data/mjtensu/.local/recognition/nanodet_single_class_dataset`.
The remaining `pkg_resources` deprecation warning is expected from Lightning 1.9.5 and does not block execution while Setuptools remains below version 81.

This verifies that a CUDA 11.7 PyTorch runtime can execute through the existing host NVIDIA driver without a CUDA 11.7 host toolkit or container.
The environment versions must be recorded with each experiment result.

## Cross-cutting observations

Source provenance is not optional when the two datasets are combined.
The region detector can consume both sources, while the later classifier is restricted to Japanese-source records.
A flattened generated dataset that removes source identity would make that boundary difficult to enforce and audit.

An image-level expectation such as fourteen tiles is not a detector acceptance criterion.
Dense or aligned arrangements are useful only as conditions under which instance separation and localization can be measured.

## Follow-up judgment candidates

- Review and prototype a vectorized `dynamic_k_matching` implementation that replaces per-ground-truth top-k launches with one batched top-k plus masking, verify assignment equivalence on real batches, and benchmark it from the same checkpoint before deciding whether to alter the required E1/E2 runs.
- Define exact geometric rules for merge, fragmentation, and adjacency classification in the evaluation implementation.

## Recommendation

Use the generated source-preserving single-class COCO dataset for NanoDet experiments.
Run NanoDet-Plus-m at 320 x 320 and 416 x 416 as the required baseline comparison. Run NanoDet-Plus-m-1.5x at 416 x 416 only when the smaller 416 model remains materially capacity-limited. Re-run only the selected best condition with two additional seeds.
Evaluate through one-to-one, count-independent instance matching, standard COCO AP, explicit operational failure metrics, stratified reporting, and validation-based confidence-threshold selection.
Use the verified NanoDet v1.0.0 Python 3.10 environment with PyTorch 1.13.1+cu117 and PyTorch Lightning 1.9.5 for training. Preserve the host CUDA 13.0 environment and do not require Docker.

## Follow-up artifact candidates

- Reproducible NanoDet training environment and generated experiment configurations.
- Detector evaluation report with instance-level error samples and runtime measurements.

## Open questions

- Which execution runtime should define the practical deployment latency and memory acceptance boundary.
- Which numeric acceptance thresholds should be set after baseline measurements expose the attainable range.
